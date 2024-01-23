import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import sys, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import tqdm
import json

def parse_data(source_file, start, end=None):
    with open(source_file, "r") as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        name_url_list = []
        for row in datareader:
            count += 1
            if count < start:
                continue
            if count % 10000 == 0:
                print('count', count)
            p_info = row[0].split()[:3]
            person_id = p_info[0]
            if person_id not in giphy_names:
                continue
            person_ind = p_info[1]
            link = p_info[2]
            #person_name = id_to_name[person_id]
            #person_name = id_to_name[person_id].split('/')[0]
            name_url_list.append([person_id, person_ind, link])
    return name_url_list

def download_image(name_url):
    f_name = '_'.join(' '.join(name_url[:2]).split())
    name = name_url[0]
    url = name_url[-1]
    if not os.path.exists(dl_dir+name):
        os.mkdir(dl_dir+name)
    filename = os.path.join(dl_dir + name_url[0], f_name + '.jpg')
    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return

    try:
        response = request.urlopen(url, timeout=10)
        image_data = response.read()
    except:
        print('Warning: Fail to download image from {}'.format(url))
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(f_name))
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(f_name))
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return
    print('Image {} completecd.'.format(url))
    return
#dl_dir = '/pool0/ml/elv-xuwen/MS-Celeb-1M/raw_dataset/'
dl_dir = '/pool0/ml/elv-xuwen/MS-Celeb-1M/vcl/'
source_file = '/pool0/ml/elv-youliangyu/data/MS-Celeb-1M/FaceImageCroppedWithOutAlignment.tsv'
start = 0
#end = 7000000

#giphy_names = np.load('./giphy_names.npy', allow_pickle=True)
giphy_names = np.load('./vcl.npy', allow_pickle=True)
name_url_list = parse_data(source_file, start)
print('name_url_list length', len(name_url_list))
pool = multiprocessing.Pool(processes=16)  # Num of CPUs
# pool.map(download_image, name_url_list)
tqdm.tqdm(pool.map(download_image, name_url_list))
# failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, name_url_list), total=len(name_url_list)))

pool.close()
pool.terminate()