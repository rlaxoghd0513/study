#깨진 이미지 찾기.


from PIL import Image
import os

checkdir = os.path.join('d:/study_data/_data/project/project/Training/')
files = os.listdir(checkdir)
format = [".jpg", ".jpeg"]

for(path, dirs, f) in os.walk(checkdir):
    for file in f:
        if file.endswith(tuple(format)):
            try:
                image = Image.open(path+"/"+file).load()
            except Exception as e:
                print('An exception is raised:', e)
                print(file)
                
                os.remove(os.path.join(path, file))
                print(f'{file} has been deleted.')
                
                