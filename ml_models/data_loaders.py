import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms


def default_loader(path):
    return Image.open(path).convert('RGB')

def get_transforms(is_train=False):
    if is_train:
        data_transforms = transforms.Compose([
            transforms.Scale(266),
            transforms.CenterCrop((400, 266)),
            # transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Scale(266),
            transforms.CenterCrop((400, 266)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return data_transforms


class ImageURLDataset(data.Dataset):

    def __init__(self, image_url, transform=None, target_transform=None, loader=default_loader):

        super().__init__()

        self.image_url = image_url
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img = self.loader(self.image_url)
        target = 0
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return 1
