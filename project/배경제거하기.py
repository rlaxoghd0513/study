import os
import numpy as np
from PIL import Image
from rembg import remove

input_dir = "d:/study_data/_data/project/test/"
output_dir = "d:/study_data/_data/test1/"

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGBA')

        out = remove(img)

        out = out.convert('RGBA')

        arr = np.array(out)

        arr[(arr[:,:,0] < 1) & (arr[:,:,1] < 1) & (arr[:,:,2] < 1)] = [0, 0, 0, 0]

        # 투명한 부분을 아예 없애기
        # arr = arr[(arr[:,:,3] > 0)]

        out = Image.fromarray(arr)

        output_path = os.path.join(output_dir, filename.split(".")[0] + "_transparent.png")
        out.save(output_path)