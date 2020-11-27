import  h5py
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import json
from image import *

#set the root to the path of FDST dataset you download
root = ''

#now generate the FDST's ground truth
train_folder = os.path.join(root,'train_data')
test_folder = os.path.join(root,'test_data')
path_sets = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder,f))]+[os.path.join(test_folder,f) for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder,f))]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print (img_path)
    gt_path = img_path.replace('.jpg','.json')
    with open (gt_path,'r') as f:
        gt = json.load(f)

    anno_list = gt.values()[0]['regions']
    img= plt.imread(img_path)
    k = np.zeros((360,640))
    rate_h = img.shape[0]/360.0
    rate_w = img.shape[1]/640.0
    for i in range(0,len(anno_list)):
        y_anno = min(int(anno_list[i]['shape_attributes']['y']/rate_h),360)
        x_anno = min(int(anno_list[i]['shape_attributes']['x']/rate_w),640)
        k[y_anno,x_anno]=1
    k = gaussian_filter(k,3)
    with h5py.File(img_path.replace('.jpg','_resize.h5'), 'w') as hf:
            hf['density'] = k
            hf.close()
