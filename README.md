# EvenFormer: Dynamic Even Transformer for Real-World Image Restoration

<a href='https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Lu_EvenFormer_Dynamic_Even_Transformer_for_Real-World_Image_Restoration_CVPRW_2025_paper.pdf'><img src='https://img.shields.io/badge/Paper-CVPRW2025-b31b1b.svg'></a> &nbsp;&nbsp;

## :trophy: Runner-Up of the NTIRE 2025 Image Shadow Removal Challenge & 2nd Highest PSNR of the NTIRE 2025 Ambient Lighting Normalization Challenge

Our team achieved the **Runner-Up Award** in the [NTIRE 2025 Image Shadow Removal Challenge](https://cvlai.net/ntire/2025/) and the **2nd highest PSNR** in the NTIRE 2025 Ambient Lighting Normalization Challenge, with less than 1/10 of the parameters of the first-place solution (23M vs. 235M+).

This is the official PyTorch implementation of the paper:

>**EvenFormer: Dynamic Even Transformer for Real-World Image Restoration**<br>
>Xin Lu, Yuanfei Bao, Jiarong Yang, Anya Hu, Jie Xiao, Kunyu Wang, Dong Li, Senyan Xu, Kean Liu, Xueyang Fu<sup>&dagger;</sup>, Zheng-Jun Zha<br>
>University of Science and Technology of China (USTC)<br>
>CVPR Workshop 2025

![pipeline](assets/pipeline.png)


## :wrench: Dependencies and Installation

```bash
git clone https://github.com/fanzh03/EvenFormer.git
cd EvenFormer
pip install -r requirements.txt
```

**Main dependencies:** PyTorch >= 1.10, torchvision, numpy, Pillow, timm, tensorboard, lpips


## :file_folder: Project Structure

```
EvenFormer/
    ├── ckpt/                         # Pre-trained checkpoints
    │   ├── best1.pth                 # Stage 1: ViT model weights
    │   └── best2.pth                 # Stage 2: NAFNet refinement weights
    ├── datasets/                     # Dataset loading
    │   └── datasets_pairs.py
    ├── loss/                         # Loss functions
    │   ├── losses.py                 # Charbonnier, FFT, SSIM, LPIPS losses
    │   └── ...
    ├── networks/                     # Model architectures
    │   ├── MaeVit_arch.py            # Stage 1: ViT encoder-decoder with U-Net skip connections
    │   ├── NAFNet_arch.py            # Stage 2: NAFNet refinement network
    │   ├── gaussian_even.py          # Gaussian Even Mechanism (shuffle & inverse shuffle)
    │   ├── Split_images.py           # Image splitting & merging (4x4 grid)
    │   └── ...
    ├── utils/
    │   └── UTILS.py                  # Metrics & utilities
    ├── TEST.py                       # Inference script (Algorithm 2)
    └── train_evenformer.py           # Training script (three-step strategy)
```


## :surfer: Quick Start

**Step 1: Download Checkpoints**

Download the pre-trained checkpoints and place them in the `ckpt/` directory:
- `best1.pth` — Stage 1 ViT model
- `best2.pth` — Stage 2 NAFNet refinement model

**Step 2: Run Testing**

```bash
python TEST.py \
    --eval_in_path ./test_images/ \
    --result_path ./results/ \
    --use_even True
```

The restored results will be saved in `./results/`. A log file at `./results/log_file/test.txt` records per-image PSNR/SSIM metrics.

**Note:** Ensure both paths end with `/`.


## :muscle: Train

**Step 1: Prepare Data**

Prepare training pairs (degraded / ground-truth images). We use the NTIRE 2025 Ambient Lighting Normalization dataset and NTIRE 2025 Image Shadow Removal dataset.

**Step 2: Three-step Training**

Our training follows a three-step strategy with the Gaussian Even Mechanism:

1. **Step 1** — Train ViT with Charbonnier + FFT loss (Adam, lr=4e-4, batch=4, patch=512, 1000 epochs):
```bash
python train_evenformer.py \
    --experiment_name step1_vit \
    --unified_path ./experiments/ \
    --training_path_txt data/train_list.txt \
    --eval_in_path /PATH/val_input/ \
    --eval_gt_path /PATH/val_gt/ \
    --training_step 1 \
    --BATCH_SIZE 4 \
    --Crop_patches 512 \
    --learning_rate 0.0004 \
    --EPOCH 1000 \
    --base_loss char \
    --addition_loss fft \
    --addition_loss_coff 0.02 \
    --use_even True
```

2. **Step 2** — Freeze ViT, train NAFNet with Charbonnier + SSIM loss (Adam, lr=4e-5, batch=1, patch=750, 300 epochs):
```bash
python train_evenformer.py \
    --experiment_name step2_nafnet \
    --unified_path ./experiments/ \
    --training_step 2 \
    --load_pre_model True \
    --pre_model_0 ./experiments/step1_vit/best_vit.pth \
    --BATCH_SIZE 1 \
    --Crop_patches 750 \
    --learning_rate 0.00004 \
    --EPOCH 300 \
    --base_loss char \
    --addition_loss ssim \
    --addition_loss_coff 0.2 \
    --use_even True
```

3. **Step 3** — Fine-tune both stages with Charbonnier + LPIPS loss (SGD, lr=1e-5, batch=2, patch=1000):
```bash
python train_evenformer.py \
    --experiment_name step3_finetune \
    --unified_path ./experiments/ \
    --training_step 3 \
    --load_pre_model True \
    --pre_model_0 ./experiments/step2_nafnet/best_vit.pth \
    --pre_model_1 ./experiments/step2_nafnet/best_nafnet.pth \
    --BATCH_SIZE 2 \
    --Crop_patches 1000 \
    --learning_rate 0.00001 \
    --base_loss char \
    --addition_loss lpips \
    --addition_loss_coff 0.6 \
    --optim sgd \
    --use_even True
```


## :book: Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@InProceedings{Lu_2025_CVPR,
    author    = {Lu, Xin and Bao, Yuanfei and Yang, Jiarong and Hu, Anya and Xiao, Jie and Wang, Kunyu and Li, Dong and Xu, Senyan and Liu, Kean and Fu, Xueyang and Zha, Zheng-Jun},
    title     = {EvenFormer: Dynamic Even Transformer for Real-World Image Restoration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025}
}
```


## :postbox: Contact

Please feel free to contact us if there is any question (luxion@mail.ustc.edu.cn).
