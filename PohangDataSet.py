from torch.utils.data import Dataset
import os
import numpy as np
import torch
import random
import torchvision.transforms as tf
from PIL import Image
from PIL import ImageEnhance


def random_roate(img1, img2, mask):
    angle = tf.RandomRotation.get_params([-180, 180])
    img1 = img1.rotate(angle)
    img2 = img2.rotate(angle)
    mask = mask.rotate(angle)
    return img1,img2, mask


def enhance_feature(image):
    if random.random() > 0.5:
        enh_image = ImageEnhance.Brightness(image)
        brightness = 1.5
        image = enh_image.enhance(brightness)
    if random.random() > 0.5:
        enh_col = ImageEnhance.Color(image)
        color = 1.5
        image = enh_col.enhance(color)
    if random.random() > 0.5:
        enh_con = ImageEnhance.Contrast(image)
        contrast = 1.5
        image = enh_con.enhance(contrast)
    return image


class PohangDataset(Dataset):
    def __init__(self, root, isTrain=False):

        self.isTrain = isTrain
        self.sync_img_mask = []

        img_sar_dir = os.path.join(root, 'sar')
        img_opt_dir = os.path.join(root, 'opt')
        mask_dir = os.path.join(root, 'lbl')

        for img_filename in os.listdir(img_sar_dir):
            img_mask_pair = (os.path.join(img_sar_dir, img_filename),
                             os.path.join(img_opt_dir, img_filename),
                             os.path.join(mask_dir, img_filename.replace('jpg', 'png')))
            self.sync_img_mask.append(img_mask_pair)

        self.trans_opt = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.trans_sar = tf.Compose([
            tf.ToTensor(),
            tf.Normalize([0.5], [0.5])
        ])

        if (len(self.sync_img_mask)) == 0:
            print("Found 0 dataset, please check your dataset!")

    def __getitem__(self, index):
        img_sar_path, img_opt_path, mask_path = self.sync_img_mask[index]
        img_sar = Image.open(img_sar_path)
        img_opt = Image.open(img_opt_path)
        mask = Image.open(mask_path).convert('L')

        if self.isTrain:
            img_sar, img_opt, mask = random_roate(img_sar, img_opt, mask)
            img_opt = enhance_feature(img_opt)

        img_RGB = self.trans_opt(img_opt)
        img_sar = self.trans_sar(img_sar)
        mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long()

        return img_sar, img_RGB, mask

    def __len__(self):
        return len(self.sync_img_mask)
