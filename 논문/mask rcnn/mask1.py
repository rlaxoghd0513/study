import os
import glob
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# 데이터셋 만들기

# classes = (
#     'top', 'blouse', 't-shirt', 'Knitted fabri', 'shirt', 'bra top', 
#     'hood', 'blue jeans', 'pants', 'skirt', 'leggings', 'jogger pants', 
#     'coat', 'jacket', 'jumper', 'padding jacket', 'best', 'kadigan', 
#     'zip up', 'dress', 'jumpsuit')
classes = ('레트로','로맨틱','리조트')
# num_classes = 21
num_classes = len(classes)
class FashionDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.coco = COCO(path)
        self.image_ids = list(self.coco.imgToAnns.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        file_name = f'D:\coco\원천데이터\원천데이터_1/{file_name}'
        image = Image.open(file_name).convert('RGB')

        annot_ids = self.coco.getAnnIds(imgIds=image_id)
        annots = [x for x in self.coco.loadAnns(annot_ids) if x['image_id'] == image_id]
        
        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int32)
        masks = np.array([self.coco.annToMask(annot) for annot in annots], dtype=np.uint8)

        area = np.array([annot['area'] for annot in annots], dtype=np.float32)
        iscrowd = np.array([annot['iscrowd'] for annot in annots], dtype=np.uint8)

        target = {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd}
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.uint8)            

        return image, target
# 데이터 어그멘테이션 만들기

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(
                image, target)

        return image, target


class Resize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):
        w, h = image.size
        image = image.resize(self.size)

        _masks = target['masks'].copy()
        masks = np.zeros((_masks.shape[0], self.size[0], self.size[1]))
        
        for i, v in enumerate(_masks):
            v = Image.fromarray(v).resize(self.size, resample=Image.BILINEAR)
            masks[i] = np.array(v, dtype=np.uint8)

        target['masks'] = masks
        target['boxes'][:, [0, 2]] *= self.size[0] / w
        target['boxes'][:, [1, 3]] *= self.size[1] / h
        
        return image, target
        

class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        
        return image, target
# Mask R-CNN 모형 만들기

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, len(classes)+1)
# 모형 학습하기

batch_size = 32
lr = 1e-3
max_size = 800
num_workers = 12
num_epochs = 5
device = 'cuda:0'

transforms_train = Compose([
    Resize((max_size, max_size)),
    ToTensor()])


def collate_fn(batch):
    return tuple(zip(*batch))


dataset = FashionDataset('D:/coco/라벨링데이터/레트로/268.json', transforms=transforms_train)
train_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, 
    num_workers=num_workers, collate_fn=collate_fn)


model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(
    params, lr=lr, weight_decay=1e-5)


def train_fn():
    model.train()
    for epoch in range(num_epochs):
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
            
            print(
                f"{epoch}, {i}, C: {losses['loss_classifier'].item():.5f}, M: {losses['loss_mask'].item():.5f}, "\
                f"B: {losses['loss_box_reg'].item():.5f}, O: {losses['loss_objectness'].item():.5f}, T: {loss.item():.5f}")
            loss.backward()
            optimizer.step()