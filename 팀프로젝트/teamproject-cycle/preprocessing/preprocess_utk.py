import os
import shutil
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir',
                    default='D:/UTK/archive/UTKFace',
                    help='The UTKFace aligned images dir')
parser.add_argument('--output_dir',
                    default='D:/UTK',
                    help='The directory to write processed images')


def main():
    args = parser.parse_args()
    image_names = [x for x in os.listdir(args.data_dir) if x.endswith('.jpg')] #os,listdir 이름을 반환하는 함수
    print(f"Total images found: {len(image_names)}")

    ages = [int(x.split('_')[0]) for x in image_names]

    ages_to_keep_a = [x for x in range(18, 29)]
    ages_to_keep_b = [x for x in range(40, 120)]

    domainA, domainB = [], []
    for image_name, age in zip(image_names, ages):
        if age in ages_to_keep_a:
            domainA.append(image_name)
        elif age in ages_to_keep_b:
            domainB.append(image_name)

    N = min(len(domainA), len(domainB))
    domainA = domainA[:N]
    domainB = domainB[:N]

    print(f"Image in A: {len(domainA)} and B: {len(domainB)}")

    domainA_dir = os.path.join(args.output_dir, 'trainA') #output-dir에 trainA라는 폴더를 만들거고 그 경로는 domainA_dir로 지정한다
    domainB_dir = os.path.join(args.output_dir, 'trainB')

    os.makedirs(domainA_dir, exist_ok=True)
    os.makedirs(domainB_dir, exist_ok=True)

    for imageA, imageB in zip(domainA, domainB):
        shutil.copy(os.path.join(args.data_dir, imageA), os.path.join(domainA_dir, imageA))
        shutil.copy(os.path.join(args.data_dir, imageB), os.path.join(domainB_dir, imageB))
        #shutil.copy(os.path.join(args.data_dir, imageA), os.path.join(domainA_dir, imageA))는 
        # args.data_dir에 있는 imageA 파일을 domainA_dir로 복사하는 코드입니다. 
        # os.path.join() 함수를 사용하여 원본 파일 경로와 복사할 경로를 결합합니다. shutil.copy() 함수를 사용하여 파일을 복사합니다


if __name__ == '__main__':
    main()
