import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
# from sympy.codegen import Print
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation

defult_cfg={


    "inputs": "rgbd",

    "dataset": "irseg",
    "root": "D:\github\SEGMENT\SGFNet-main_DOP\dataset\data_",
    "n_classes": 1,
    "id_unlabel": -1,
    "brightness": 0.5,
    "contrast": 0.5,
    "saturation": 0.5,
    "p": 0.5,
    "scales_range": "0.5 2.0",
    "crop_size": "800 640",
    "eval_scales": "0.5 0.75 1.0 1.25 1.5 1.75",
    "eval_flip": "true",

    "ims_per_gpu": 4,
    "num_workers": 1,
    "lr_start": 5e-5,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_power": 0.9,
    "epochs": 400,

    "loss": "crossentropy",
    "class_weight": "enet"
}

class IRSeg(data.Dataset):

    def __init__(self, cfg=defult_cfg,root=defult_cfg['root'], mode='trainval', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449], [0.226]), 
        ])


        self.root = root
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        self.mode = mode
        self.do_aug = do_aug

        if self.mode == 'train':
            self.rgb_image_dir = os.path.join(self.root, 'S0') 
            self.DOP_image_dir = os.path.join(self.root, 'DOP_Gray')
            self.label_dir = os.path.join(self.root, 'gtmask')
        elif self.mode == 'test':
            self.rgb_image_dir = os.path.join(self.root, 'S0_test')
            self.DOP_image_dir = os.path.join(self.root, 'DOP_Gray_test')
            self.label_dir = os.path.join(self.root, 'gtmask_test') 

        self.image_files = [f for f in os.listdir(self.rgb_image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index].strip()

        image = Image.open(os.path.join(self.root, self.rgb_image_dir, image_path))
        DOP = Image.open(os.path.join(self.root, self.DOP_image_dir, image_path))
        label = Image.open(os.path.join(self.label_dir, image_path))

        resize_transform = transforms.Resize((800, 640)) 

        image = resize_transform(image)
        DOP = resize_transform(DOP)
        label = resize_transform(label)

        sample = {
            'image': image,
            'DOP': DOP,
            'label': label,
        }

        if self.mode in ['train', 'trainval'] and self.do_aug:
            sample = self.aug(sample)

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['DOP'] = self.dp_to_tensor(sample['DOP'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        return sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),
            (255,255,255),
        ]


