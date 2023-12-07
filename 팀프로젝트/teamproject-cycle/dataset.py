import os

from PIL import Image
from torch.utils.data import Dataset


class ImagetoImageDataset(Dataset):
    def __init__(self, domainA_dir, domainB_dir, transforms=None):#이미지파일이 위치한 경로와 이미지에 적용할 변환을 지정하는 매개변수
        self.imagesA = [os.path.join(domainA_dir, x) for x in os.listdir(domainA_dir) if #os.path.join domainA_dir안에 x라는 리스트를 만들거다
                        x.endswith('.png') or x.endswith('jpg')] #domainA_dir에 있는 파일들의 경로를 리스트로 저장.
        self.imagesB = [os.path.join(domainB_dir, x) for x in os.listdir(domainB_dir) if
                        x.endswith('.png') or x.endswith('jpg')]

        self.transforms = transforms

    def __len__(self):
        return min(len(self.imagesA), len(self.imagesB))

    def __getitem__(self, idx):
        imageA = Image.open(self.imagesA[idx])#인덱스는 데이터셋에서 가져올 데이터의 위치를 지정하는 역할
        imageB = Image.open(self.imagesB[idx])

        if self.transforms is not None:
            imageA = self.transforms(imageA)
            imageB = self.transforms(imageB)

        return imageA, imageB
