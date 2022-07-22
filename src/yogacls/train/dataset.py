import random
from pathlib import Path
from typing import List

import torch
from pathlib import Path
import torch.nn.functional as F
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
import transformers
from src.util.logger import getLogger

logger = getLogger()


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def load_datalist(data_dir):
    data_dir = Path(data_dir)
    pose_list = data_dir.glob("*")
    class_dic = {}
    class_id = 0
    dataset = []
    for pose in pose_list:
        if not pose.is_dir():
            continue
        pose_name = pose.stem
        class_dic[class_id] = pose_name
        for fn in pose.glob("*.png"):
            dataset.append([str(fn.absolute()),class_id ])
        class_id += 1
    return dataset, class_dic

def compose_augmentation():
    transform = [
        # リサイズ
        A.Resize(230, 230),
        A.RandomCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # ぼかし
        A.Blur(blur_limit=15, p=0.5),
        # 明るさ、コントラスト
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=0.5),
        # 回転
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5),
        #Random Erasing
        #A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
        # 正規化
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
    ]
    return A.Compose(transform)

def compose_transform():
    transform = [
        A.Resize(230, 230),
        A.CenterCrop(224, 224),
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2()
        
    ]
    return A.Compose(transform)

class YogaPoseDataset(Dataset):
    def __init__(self, cfg, data_list, is_aug=False) -> None:
        super().__init__()
        #self.cfg = cfg.ml
        self.data_list = data_list
        self.is_aug = is_aug  # bool
        
        self.transform = compose_transform()
        if is_aug:
            self.transform = compose_augmentation()
    
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        fp, cid = self.data_list[idx]
        image = Image.open(fp).convert("RGB")
        image = np.array(image)
        #print(image.shape)
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, cid
    
if __name__ == "__main__":
    from src.util.plot import show_img_imnet
    d, c = load_datalist("/mnt/yogapose/")
    print(len(d), d[:2])
    print(len(c))
    dataset = YogaPoseDataset(None, d, is_aug=True)
    show_img_imnet(dataset)
    
    