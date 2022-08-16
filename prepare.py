import argparse
import os
import sys
import imageio
import glob
import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',required=True,help='path to dataset')
parser.add_argument('--output',required=True,help='output path for .h5 file')
parser.add_argument('--sigma',type=float,default=3,help='Gaussian kernel size in pixels')
args = parser.parse_args()

images = []
transforms = []
counts = []
gts = []
densities = []
attentions = []

def load_data(dataset_path,names,sigma):
    data = []

    for name in names:
        image_path = os.path.join(dataset_path,'images',name + '.tif')
        image = imageio.imread(image_path)
        
        csv_path = os.path.join(dataset_path,'csv',name + '.csv')
        if os.path.exists(csv_path):
            points = np.loadtxt(csv_path,delimiter=',',skiprows=1).astype('int')
            if len(points.shape)==1:
                points = points[None,:]
            
            gt = np.zeros(image.shape[:2],dtype='float32')
            gt[points[:,1],points[:,0]] = 1
        
            distance = distance_transform_edt(1-gt).astype('float32')
            confidence = np.exp(-distance**2/(2*sigma**2))
        else:
            gt = np.zeros(image.shape[:2],dtype='float32')
            confidence = np.zeros(image.shape[:2],dtype='float32')
            
        confidence = confidence[...,None]

        attention = confidence>0.001
        attention = confidence.astype('float32')

        data.append({
            'name':name,
            'image':image,
            'gt':gt,
            'confidence':confidence,
            'attention':attention
        })
    
    return data

def augment_images(images):
    """ Augment by rotating and flipping """
    """ Adapted from https://github.com/juglab/n2v/blob/master/n2v/internals/N2V_DataGenerator.py """
    augmented = np.concatenate((images,
                              np.rot90(images, k=1, axes=(1, 2)),
                              np.rot90(images, k=2, axes=(1, 2)),
                              np.rot90(images, k=3, axes=(1, 2))))
    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented
    
def read_names(split):
    return [name.rstrip() for name in open(os.path.join(args.dataset,split+'.txt'),'r')]
train_names,val_names,test_names = [read_names(split) for split in ['train','val','test']]

train_data,val_data,test_data = [load_data(args.dataset,names,args.sigma) for names in [train_names,val_names,test_names]]

def add_data_to_h5(f,data,split,augment=False):
    names = np.array([d['image'] for d in data])
    images = np.stack([d['image'] for d in data],axis=0)
    gt = np.stack([d['gt'] for d in data],axis=0)
    confidence = np.stack([d['confidence'] for d in data],axis=0)
    attention = [d['attention'] for d in data]
    
    if augment:
        names = np.repeat(names,8)
        images = augment_images(images)
        gt = augment_images(gt)
        confidence = augment_images(confidence)
        attention = augment_images(attention)

    f.create_dataset(f'{split}/names',data=names)
    f.create_dataset(f'{split}/images',data=images)
    f.create_dataset(f'{split}/gt',data=gt)
    f.create_dataset(f'{split}/confidence',data=confidence)
    f.create_dataset(f'{split}/attention',data=attention)

with h5py.File(args.output,'w') as f:
    add_data_to_h5(f,train_data,'train',augment=True)
    add_data_to_h5(f,val_data,'val')
    add_data_to_h5(f,test_data,'test')

