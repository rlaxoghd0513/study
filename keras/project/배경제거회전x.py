import os
import numpy as np
from PIL import Image, ExifTags
from rembg import remove

input_dir = "d:/study_data/_data/project/project/Training/[원천]EMOIMG_중립_TRAIN_04/"
output_dir = "d:/study_data/_data/project/project/Training_x/[원천]EMOIMG_중립_TRAIN_04/"

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        
        img_path = os.path.join(input_dir, filename)

        img = Image.open(img_path)

        # 이미지 회전 정보 가져오기
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            exif=dict(img._getexif().items())
            if exif[orientation] == 3:
                img=img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img=img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img=img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError, TypeError):
            # 예외 처리: 이미지에 회전 정보가 없는 경우 무시
            pass

        img = img.convert('RGBA')

        out = remove(img)

        out = out.convert('RGBA')

        arr = np.array(out)

        arr[(arr[:,:,0] < 1) & (arr[:,:,1] < 1) & (arr[:,:,2] < 1)] = [0, 0, 0, 0]

        out = Image.fromarray(arr)

        output_path = os.path.join(output_dir, filename.split(".")[0] + "_transparent.png")
        out.save(output_path)