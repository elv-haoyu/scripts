import os
import json
from pathlib import Path

import cv2
# import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# from collections import defaultdict

def rm_space_title(file_path:str):
    "file_path = './TL_title_png'"
    path = Path(file_path)
    p_lst = path.glob('*.png')
    for p in p_lst:
        t = str(p).replace(' ', '_').replace('&', '_')
        os.system(f"mv \"{p}\" {t}")
        
def read_csv_images(csv):
    with open(csv, 'r') as f:
        files = [f.strip().replace(",","") for f in f.readlines()]
    return files

def getImg(i, img_files, cvt=True):
    img = cv2.imread(str(img_files[i]))
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_img_cv(img_files, cvt=True):
    img = cv2.imread(str(img_files))
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def display_single_cv(img_path:str, cvt=True):
    img = imread_img(str(img_path), cvt)
    plt.imshow(img)
    plt.title(f"Image {img_path}")
    plt.show()
    
def get_images_plt(i, files):
    img = plt.imread(files[i])
    return img

def display_single_plt(img_path):
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title(f"Image {img_path}")
    plt.show()
    
    
def displayFiles(i, files, cvt=True):
    plt.imshow(getImg(i, files, cvt))
    plt.title('Image {}'.format(i))
    plt.show()

    
def display_files(i, files):
    img = plt.imread(files[i])
    plt.imshow(img)
    plt.title('Image {}'.format(i))
    plt.show()
    
    
def displayImg(img_path, cvt=True):
    img = cv2.imread(str(img_path))
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('Image {}'.format(img_path))
    plt.show()
    
    
def display_image(img_path):
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title('Image {}'.format(img_path))
    plt.show()

def display_all(file_lst, cvt):
    img_data = [getImg(i, file_lst,cvt) for i in range(len(file_lst))]

    fig = plt.figure(figsize=(20, 20*(len(file_lst)//16+1)))
    grid = ImageGrid(
        fig, 111,  # similar to subplot(111)
        nrows_ncols=(len(file_lst)//4+1, 4),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
        )

    for idx, (ax, im) in enumerate(zip(grid, img_data)):
        ax.imshow(im)
        ax.title.set_text('Image {}'.format(idx))

    plt.show()

    
def display_cv2plt(cv_img, img_path):
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"Image {img_path}")
    plt.show()
