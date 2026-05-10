import time, torchvision, argparse, logging, sys, os
import torch, random, collections
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
from utils.UTILS1 import compute_psnr
from utils.UTILS import AverageMeters, print_args_parameters, compute_ssim
from datasets.datasets_pairs import my_dataset_eval
from networks.MaeVit_arch import MaskedAutoencoderViT
from networks.NAFNet_arch import NAFNet
from networks.Split_images import split_image, merge, process_split_image_with_model_parallel
from networks.gaussian_even import gaussian_shuffle, gaussian_inverse_shuffle
from functools import partial
from networks.image_utils import splitimage, mergeimage

sys.path.append(os.getcwd())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:', device)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--vit_patch_size', type=int, default=8)
parser.add_argument('--vit_embed_dim', type=int, default=256)
parser.add_argument('--vit_depth', type=int, default=6)
parser.add_argument('--vit_num_heads', type=int, default=8)
parser.add_argument('--vit_decoder_embed_dim', type=int, default=256)
parser.add_argument('--vit_decoder_depth', type=int, default=6)
parser.add_argument('--vit_decoder_num_heads', type=int, default=8)
parser.add_argument('--vit_mlp_ratio', type=int, default=4)
parser.add_argument('--vit_img_size', type=int, default=352)
parser.add_argument('--vit_grid_type', type=str, default='4x4')
parser.add_argument('--overlap_size', type=int, default=176)
parser.add_argument('--Crop_patches', type=int, default=352)

# path setting
parser.add_argument('--experiment_name', type=str, default="test")
parser.add_argument('--result_path', type=str, default='./results/')
parser.add_argument('--eval_in_path', type=str, default='./test_images/')
parser.add_argument('--eval_gt_path', type=str, default='')

# load pre-trained model
parser.add_argument('--pre_model_0', type=str, default='./ckpt/best1.pth')
parser.add_argument('--pre_model_1', type=str, default='./ckpt/best2.pth')

# model setting
parser.add_argument('--base_channel', type=int, default=24)
parser.add_argument('--num_res', type=int, default=6)
parser.add_argument('--img_channel', type=int, default=3)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1])
parser.add_argument('--global_residual', type=str2bool, default=True)

# Gaussian Even
parser.add_argument('--use_even', type=str2bool, default=True)
parser.add_argument('--inputs_ensemble', type=str2bool, default=False)

args = parser.parse_args()
if not args.eval_gt_path:
    args.eval_gt_path = args.eval_in_path
print_args_parameters(args)

log_dir = args.result_path + '/log_file/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

trans_eval = transforms.Compose([transforms.ToTensor()])
results_mertircs = log_dir + args.experiment_name + '.txt'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test_even(net, net_1, eval_loader, Dname='S', save_result=True):
    """Algorithm 2: Dynamic Even Transformer During Testing.

    Uses Gaussian shuffling + overlapped patches + mean merge.
    """
    net.eval()
    net_1.eval()
    with torch.no_grad():
        Avg_Meters_evaling = AverageMeters()
        st = time.time()
        for index, (data_in, label, name) in enumerate(eval_loader, 0):
            inputs = Variable(data_in).to(device)
            labels = Variable(label).to(device)
            B, C, H, W = inputs.shape

            if index == 0:
                print(f" val_input.size: {data_in.size()}, gt.size: {label.size()}")

            # Step 1: Gaussian Shuffling
            if args.use_even:
                shuffled, h_idx, w_idx = gaussian_shuffle(inputs)
            else:
                shuffled = inputs

            # Step 2: Overlapped patch splitting (Algorithm 2)
            split_data, starts = splitimage(shuffled, crop_size=args.Crop_patches, overlap_size=args.overlap_size)
            for i, data in enumerate(split_data):
                output = net(data)
                output = net_1(output)
                split_data[i] = output

            # Step 3: Mean-based merge for overlapping regions
            outputs = mergeimage(split_data, starts, crop_size=args.Crop_patches,
                                 resolution=(B, C, H, W), is_mean=True)

            # Step 4: Gaussian Inverse Shuffling
            if args.use_even:
                outputs = gaussian_inverse_shuffle(outputs, h_idx, w_idx)

            out_psnr = compute_psnr(outputs, labels)
            out_psnr_wClip = compute_psnr(torch.clamp(outputs, 0., 1.), labels)
            out_ssim = compute_ssim(outputs, labels)
            in_psnr = compute_psnr(inputs, labels)
            in_ssim = compute_ssim(inputs, labels)

            Avg_Meters_evaling.update({
                'eval_output_psnr': out_psnr,
                'eval_output_psnr_wClip': out_psnr_wClip,
                'eval_input_psnr': in_psnr,
                'eval_output_ssim': out_ssim,
                'eval_input_ssim': in_ssim
            })

            content = f'index: {index} | name: {name[0]} | [in_psnr:{in_psnr:.3f}, in_ssim:{in_ssim:.4f} | out_psnr:{out_psnr:.3f}, out_psnr_clip:{out_psnr_wClip:.3f}, out_ssim:{out_ssim:.4f}]'
            print(content)
            with open(results_mertircs, 'a') as file:
                file.write(content + '\n')

            if save_result:
                save_result_path = args.result_path + '/'
                os.makedirs(save_result_path, exist_ok=True)
                torchvision.utils.save_image(
                    [torch.clamp(outputs, 0., 1.).cpu().detach()[0]],
                    save_result_path + name[0], nrow=1, padding=0)

        Final_output_PSNR = Avg_Meters_evaling['eval_output_psnr']
        Final_output_PSNR_wclip = Avg_Meters_evaling['eval_output_psnr_wClip']
        Final_input_PSNR = Avg_Meters_evaling['eval_input_psnr']
        Final_output_SSIM = Avg_Meters_evaling['eval_output_ssim']
        Final_input_SSIM = Avg_Meters_evaling['eval_input_ssim']

        content_ = (f"Dataset:{Dname} || [Num_eval:{len(eval_loader)} "
                     f"In_PSNR:{Final_input_PSNR:.3f} / In_SSIM:{Final_input_SSIM:.3f} || "
                     f"Out_PSNR:{Final_output_PSNR:.3f} | Out_PSNR_wclip:{Final_output_PSNR_wclip:.3f} / "
                     f"OUT_SSIM:{Final_output_SSIM:.3f}] cost time: {time.time() - st:.1f}s")
        print(content_)
        with open(results_mertircs, 'a') as file:
            file.write(content_ + '\n')


def get_eval_data(val_in_path=args.eval_in_path, val_gt_path=args.eval_gt_path, trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label=val_gt_path, transform=trans_eval, fix_sample=500)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader


def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))


if __name__ == '__main__':
    # Build Stage 1: Vision Transformer
    net = MaskedAutoencoderViT(
        patch_size=args.vit_patch_size, embed_dim=args.vit_embed_dim, depth=args.vit_depth,
        num_heads=args.vit_num_heads,
        decoder_embed_dim=args.vit_decoder_embed_dim, decoder_depth=args.vit_decoder_depth,
        decoder_num_heads=args.vit_decoder_num_heads,
        mlp_ratio=args.vit_mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    # Build Stage 2: Compact NAFNet (without global residual)
    net_1 = NAFNet(img_channel=args.img_channel, width=args.base_channel, middle_blk_num=args.num_res,
                   enc_blk_nums=args.enc_blks, dec_blk_nums=args.dec_blks, global_residual=False)

    # Load weights
    net.load_state_dict(torch.load(args.pre_model_0, map_location=device), strict=True)
    net_1.load_state_dict(torch.load(args.pre_model_1, map_location=device), strict=True)
    print('-----' * 20, 'successfully load pre-trained weights!')

    net.to(device)
    net_1.to(device)
    print_param_number(net)
    print_param_number(net_1)

    eval_loader = get_eval_data(val_in_path=args.eval_in_path, val_gt_path=args.eval_gt_path)

    test_even(net=net, net_1=net_1, eval_loader=eval_loader, Dname=args.experiment_name, save_result=True)

    with open(results_mertircs, 'a') as file:
        file.write('-=-=' * 50 + '\n')
