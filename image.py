import os
from PIL import Image
import numpy as np
import h5py
import cv2


def load_data(img_path,train = True):
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    index = int(img_name.split('.')[0])

    prev_index = int(max(1,index-5))
    post_index = int(min(150,index+5))

    prev_img_path = os.path.join(img_folder,'%03d.jpg'%(prev_index))
    post_img_path = os.path.join(img_folder,'%03d.jpg'%(post_index))

    gt_path = img_path.replace('.jpg','_resize.h5')

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')
    post_img = Image.open(post_img_path).convert('RGB')

    # resize image to 640*360 as previous work
    prev_img = prev_img.resize((640,360))
    img = img.resize((640,360))
    post_img = post_img.resize((640,360))

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    gt_file.close()
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

    return prev_img,img,post_img,target

