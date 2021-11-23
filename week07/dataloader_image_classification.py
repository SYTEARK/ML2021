from PIL import Image
from tqdm import tqdm
import random
import os.path as osp
import glob

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

class ImageTransform():
    """
    이미지 전처리 클래스. 훈련 시와 검증 시의 동작이 다르다.
    이미지 크기를 Resize하고 색상을 Normalize한다.
    훈련 시에는 RandomResizeCrop과 RandomHorizontalFlip으로 Data Augmentation을 수행한다.
    (개미와 벌 데이터는 수가 굉장히 적어서 Augmentation 필요)

    Attributes
    ----------
    resize: int
    mean: (R,G,B)
    std: (R,G,B)
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(       # 그림의 랜덤한 부분을 잘라냄
                    resize, scale=(0.5, 1.0)
                ),
                transforms.RandomHorizontalFlip(), # 랜덤하게 좌우반전
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

def make_datapath_list(data_path, phase="train"):
    """
    데이터의 경로를 저장한 리스트 작성
    """
    root_path = "hymenoptera_data/"
    target_path = osp.join(data_path, root_path+phase+'/**/*.jpg') # *.jpg: 뒤가 .jpg로 끝나는 모든 파일들
    print(target_path)
    
    path_list = []
    
    for path in glob.glob(target_path): # glob.glob(pattern) => pattern에 해당하는 파일 list를 작성
        path_list.append(path)
    
    return path_list

class HymenopteraDataset(Dataset): # torch.utils.data.Dataset을 상속
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        img_transformed = self.transform(
            img, self.phase
        )
        
        i = 0
        split_path = img_path.split('/')
        for s in split_path:
            if s == self.phase:
                i += 1
                break
            i += 1
        label = split_path[i]
            
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1
            
        return img_transformed, label
