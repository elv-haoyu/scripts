import cv2
import numpy as np
from pathlib import Path, PurePath
import os
import matplotlib.pyplot as plt


haarcascades = Path("/home/elv-haoyu/elv-ml/opencv/data/haarcascades")
def clean_pool(img_folder_lst, ii):
    nonside_gts = []
    nonside_ids = []
    face_cascade = cv2.CascadeClassifier(str(haarcascades/ 'haarcascade_frontalface_default.xml'))
#     face_cascade = cv2.CascadeClassifier('/home/elv-haoyu/elvTaggerP0/celeb-tagger/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml') 
    eye_cascade = cv2.CascadeClassifier(str(haarcascades/ 'haarcascades/haarcascade_eye.xml'))
    
    for image_folder_path in img_folder_lst:
        nonside_gt = []
        nonside_id = []
        img_lst = os.listdir(image_folder_path)
        for im_idx, im_id in enumerate(img_lst):
            im_path = os.path.join(image_folder_path, im_id)
            img = cv2.imread(im_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # tune scaleFactor and minNeighbors 
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) != 1:
                continue
            (x,y,w,h) = faces[0]
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # tune scaleFactor and minNeighbors , scaleFactor=1.1, minNeighbors=1
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=2)
            if len(eyes) >= 2:                
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
                nonside_gt.append(im_id)
                nonside_id.append(im_idx)
            if im_idx==ii:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.title(f"Image {im_path}")
                plt.show()
        nonside_gts.append(nonside_gt)
        nonside_ids.append(nonside_id)

    return nonside_gts, nonside_ids

def sample_face(gts, embbs):
    gts_sampler = []
    embbs_sampler = []
    for i in range(gts.shape[0]):
        n = len(gts[i])
        size = int(max(0.2*n, 20))
        if size < n:
            idx = np.random.choice(n, size, replace=False)
            idx.sort()
        
            gts_sampler.append(gts[i][idx])
            embbs_sampler.append(embbs[i][idx].astype('float16'))
        if size >= n:
            gts_sampler.append(gts[i])
            embbs_sampler.append(embbs[i].astype('float16'))

    return gts_sampler, embbs_sampler
    
    
def delete_features(gts_ibc, embb_ibc, rm_ids:list):
    idx_to_remove = np.concatenate([np.where(gts_ibc == id)[0] for id in rm_ids])
    print("previous", idx_to_remove.shape, gts_ibc.shape, embb_ibc.shape)
    gts_ibc = np.delete(gts_ibc, idx_to_remove, axis=0)
    embb_ibc = np.delete(embb_ibc, idx_to_remove, axis=0)
    print("current", gts_ibc.shape, embb_ibc.shape)
    return gts_ibc, embb_ibc

def replace_features(gts_ibc, gts_sampler: list, embb_ibc, embbs_sampler: list):
    celebs = set(np.concatenate(gts_sampler))
    idx_to_remove = [] 
    for c in celebs:
        idx_to_remove.append(np.where(gts_ibc==c)[0])
    idx_to_remove = np.array(idx_to_remove)
    print(len(idx_to_remove))
    gts_ibc = np.delete(gts_ibc, idx_to_remove, axis=0)
    embb_ibc = np.delete(embb_ibc, idx_to_remove, axis=0)
    print(len(np.array(gts_sampler)))
    return np.concatenate([gts_ibc, np.array(gts_sampler)]), \
            np.concatenate([embb_ibc, np.array(embbs_sampler)])