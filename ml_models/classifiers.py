import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision

from .data_loaders import ImageURLDataset, get_transforms


class ImagenetModel(object):

    def __init__(self, arch="resnet18"):
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()

    def predict(self, image_filename, use_gpu=None):
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        image_dset = ImageURLDataset(image_filename, transform=get_transforms(is_train=False))

        dset_loader = data.DataLoader(image_dset, shuffle=False)

        results = {}
        for batch_idx, dset in enumerate(dset_loader):
            # Get the inputs
            inputs, labels = dset
            # Wrap them in Variable
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs = Variable(inputs)
            labels = Variable(labels)

            # Forward
            outputs = self.model(inputs)
            preds_proba, preds = outputs.data.max(1)
            pred_idx = int(preds.cpu().numpy().flatten()[0])
            pred_proba = float(np.exp(preds_proba.cpu().numpy().flatten()[0]))
            results = {
                "pred_idx": pred_idx,
                "pred_prob": round(pred_proba, 2)
            }
        return results
