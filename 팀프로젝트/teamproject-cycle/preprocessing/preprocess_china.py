import os
import shutil
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir',
                    default='D:\짱깨\All-Age-Faces Dataset\original images',
                    help='The china aligned images dir')
parser.add_argument('--output_dir',
                    default='D:\짱깨\class',
                    help='The directory to write processed images')


def main():
    args = parser.parse_args()
    image_names = [x for x in os.listdir(args.data_dir) \
                   if x.endswith('.jpg')] 
    print(f"Total images found: {len(image_names)}")

    # ages = [int(x.split('_')[0]) for x in image_names]
    # ages = int(image_names.split('.')[0][-2:])
    # ages = [int(x.split('A')[0]) for x in image_names]
    ages = [int(x.split('A')[1].split('.')[0]) if x.split('A')[1].endswith('.jpg') else 0 for x in image_names]

    ages_to_keep_10 = [x for x in range(10, 20)]
    ages_to_keep_20 = [x for x in range(20, 30)]
    ages_to_keep_30 = [x for x in range(30, 40)]
    ages_to_keep_40 = [x for x in range(40, 50)]
    ages_to_keep_50 = [x for x in range(50, 60)]
    ages_to_keep_60 = [x for x in range(60, 70)]
    ages_to_keep_70 = [x for x in range(70, 80)]
    ages_to_keep_80 = [x for x in range(80, 90)]

    domain10, domain20, domain30, domain40, domain50,\
          domain60, domain70, domain80, domain_else \
            = [], [], [], [], [], [], [], [], []
    
    for image_name, age in zip(image_names, ages):
        if age in ages_to_keep_10:
            domain10.append(image_name)
        elif age in ages_to_keep_20:
            domain20.append(image_name)
        elif age in ages_to_keep_30:
            domain30.append(image_name)
        elif age in ages_to_keep_40:
            domain40.append(image_name)
        elif age in ages_to_keep_50:
            domain50.append(image_name)
        elif age in ages_to_keep_60:
            domain60.append(image_name)
        elif age in ages_to_keep_70:
            domain70.append(image_name)
        elif age in ages_to_keep_80:
            domain80.append(image_name)
        else:
            domain_else.append(image_name)

    # N = min(len(domain10), len(domainB))
    # domainA = domainA[:N]
    # domainB = domainB[:N]

    print(f"Image in 10: {len(domain10)} and 20: {len(domain20)} and 30: {len(domain30)}, and 40: {len(domain40)}, and 50: {len(domain50)}, and 60: {len(domain60)}, and 70: {len(domain70)}, and 80: {len(domain80)}")

    domain10_dir = os.path.join(args.output_dir, '10') #output-dir에 trainA라는 폴더를 만들거고 그 경로는 domainA_dir로 지정한다
    domain20_dir = os.path.join(args.output_dir, '20')
    domain30_dir = os.path.join(args.output_dir, '30')
    domain40_dir = os.path.join(args.output_dir, '40')
    domain50_dir = os.path.join(args.output_dir, '50')
    domain60_dir = os.path.join(args.output_dir, '60')
    domain70_dir = os.path.join(args.output_dir, '70')
    domain80_dir = os.path.join(args.output_dir, '80')

    os.makedirs(domain10_dir, exist_ok=True)
    os.makedirs(domain20_dir, exist_ok=True)
    os.makedirs(domain30_dir, exist_ok=True)
    os.makedirs(domain40_dir, exist_ok=True)
    os.makedirs(domain50_dir, exist_ok=True)
    os.makedirs(domain60_dir, exist_ok=True)
    os.makedirs(domain70_dir, exist_ok=True)
    os.makedirs(domain80_dir, exist_ok=True)

    # for image10, image20, image30, image40, image50, image60, image70, image80 in zip(domain10, domain20, domain30, domain40, domain50, domain60, domain70, domain80):
    #     shutil.copy(os.path.join(args.data_dir, image10), os.path.join(domain10_dir, image10))
    #     shutil.copy(os.path.join(args.data_dir, image20), os.path.join(domain20_dir, image20))
    #     shutil.copy(os.path.join(args.data_dir, image30), os.path.join(domain30_dir, image30))
    #     shutil.copy(os.path.join(args.data_dir, image40), os.path.join(domain40_dir, image40))
    #     shutil.copy(os.path.join(args.data_dir, image50), os.path.join(domain50_dir, image50))
    #     shutil.copy(os.path.join(args.data_dir, image60), os.path.join(domain60_dir, image60))
    #     shutil.copy(os.path.join(args.data_dir, image70), os.path.join(domain70_dir, image70))
    #     shutil.copy(os.path.join(args.data_dir, image80), os.path.join(domain80_dir, image80))
    #     #shutil.copy(os.path.join(args.data_dir, imageA), os.path.join(domainA_dir, imageA))는 
        # args.data_dir에 있는 imageA 파일을 domainA_dir로 복사하는 코드입니다. 
        # os.path.join() 함수를 사용하여 원본 파일 경로와 복사할 경로를 결합합니다. shutil.copy() 함수를 사용하여 파일을 복사합니다

    for image10 in domain10:
        shutil.copy(os.path.join(args.data_dir, image10), os.path.join(domain10_dir, image10))

    for image20 in domain20:
        shutil.copy(os.path.join(args.data_dir, image20), os.path.join(domain20_dir, image20))

    for image30 in domain30:
        shutil.copy(os.path.join(args.data_dir, image30), os.path.join(domain30_dir, image30))

    for image40 in domain40:
        shutil.copy(os.path.join(args.data_dir, image40), os.path.join(domain40_dir, image40))

    for image50 in domain50:
        shutil.copy(os.path.join(args.data_dir, image50), os.path.join(domain50_dir, image50))

    for image60 in domain60:
        shutil.copy(os.path.join(args.data_dir, image60), os.path.join(domain60_dir, image60))

    for image70 in domain70:
        shutil.copy(os.path.join(args.data_dir, image70), os.path.join(domain70_dir, image70))

    for image80 in domain80:
        shutil.copy(os.path.join(args.data_dir, image80), os.path.join(domain80_dir, image80))



if __name__ == '__main__':
    main()