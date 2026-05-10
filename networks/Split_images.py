from torchvision import transforms
import torch
from PIL import Image
import os
from functools import partial
from networks.image_utils import splitimage, mergeimage


def split_image(image_input, grid_type):
    image = image_input

    def split_image(image, rows, cols):
        n, c, h, w = image.shape
        h_stride, w_stride = h // rows, w // cols
        images = []
        positions = []
        for i in range(rows):
            for j in range(cols):
                sub_image = image[:, :, i * h_stride:(i + 1) * h_stride, j * w_stride:(j + 1) * w_stride]
                images.append(sub_image)
                positions.append((i * h_stride, j * w_stride, h_stride, w_stride))
        return images, positions

    if grid_type == '2x2':
        return split_image(image, 2, 2)
    elif grid_type == '4x4':
        return split_image(image, 4, 4)
    elif grid_type == '8x8':
        return split_image(image, 8, 8)
    else:
        raise ValueError("Unsupported grid type")


def merge(sub_images, positions):
    num_sub_images = len(sub_images)
    n, c, h, w = sub_images[0].shape

    max_h = max(pos[0] + pos[2] for pos in positions)
    max_w = max(pos[1] + pos[3] for pos in positions)

    image = torch.zeros((sub_images[0].size()[0], sub_images[0].size()[1], max_h, max_w))

    for img, pos in zip(sub_images, positions):
        x, y, h, w = pos
        image[:, :, x:x + h, y:y + w] = img

    return image


def process_split_image_with_model(sub_images, model):
    processed_sub_images = [model(sub_image) for sub_image in sub_images]
    return processed_sub_images


def process_split_image_with_model_parallel(sub_images, model):
    n, c, h, w = sub_images[0].shape
    L = len(sub_images)
    merged_tensor = torch.stack(sub_images, dim=0)
    reshaped_tensor = merged_tensor.view(n * L, c, h, w)
    processed_sub_images = model(reshaped_tensor)
    image = processed_sub_images.view(L, n, c, h, w)
    images = [image[i] for i in range(L)]
    return images


def process_split_image_with_model_1(net, net_0, outputs, name, inputs):
    if name:
        split_data, starts = splitimage(inputs, 352, 176)
        for i, data in enumerate(split_data):
            output = net(data)
            output = net_0(output)
            split_data[i] = output
        output = mergeimage(split_data, starts, 352, resolution=(1, 3, 1440, 1920), is_mean=False)
        outputs = output
    return outputs
