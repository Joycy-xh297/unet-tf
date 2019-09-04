import cv2
import numpy as np
import glob, os, os.path

def resize_img(path, size):
    for img in glob.glob(path+"/*.png"):
        image = cv2.imread(img)
        name = img
        print(name)
        image = cv2.resize(image, size)
        cv2.imwrite(name, image)
        print('writing img')

def main():
    general_path = "data1/RR_mint/"
    sub_paths = ["train/","test/","ref/"]
    dir_list = [general_path + sub_path for sub_path in sub_paths]
    # print(dir_list)
    size = (512, 512)
    for path in dir_list:
        # print("is the path existing? ", os.path.exists(path))
        print(os.listdir(path))
        for item in os.listdir(path):
            print(item)
            resize_img(path+item, size)

if __name__ == "__main__":
    main()
    




