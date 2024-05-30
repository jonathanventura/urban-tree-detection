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
parser.add_argument('dataset',help='path to dataset')
parser.add_argument('output',help='output path for .h5 file')
parser.add_argument('--train',default='train.txt')
parser.add_argument('--val',default='val.txt')
parser.add_argument('--test',default='test.txt')
parser.add_argument('--augment',action='store_true')
parser.add_argument('--sigma',type=float,default=3,help='Gaussian kernel size in pixels')
parser.add_argument('--bands',default='RGBN',help='description of bands in input raster (RGB or RGBN)')
args = parser.parse_args()

images = []
transforms = []
counts = []
gts = []
densities = []
attentions = []

def load_data(dataset_path,names,sigma):
    data = []

    pbar = tqdm.tqdm(total=len(names))
    for name in names:
        image = None
        for suffix in ['.tif','.tiff','.png']:
            image_path = os.path.join(dataset_path,'images',name + suffix)
            if os.path.exists(image_path):
                image = imageio.imread(image_path)
                if suffix == '.png' or args.bands == 'RGB':
                    image = image[...,:3]
                break
        if image is None:
            raise RuntimeError(f'could not find image for {name}')
        
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
        attention = attention.astype('float32')

        data.append({
            'name':name,
            'image':image,
            'gt':gt,
            'confidence':confidence,
            'attention':attention
        })
        
        pbar.update(1)
    
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
    
def read_names(filename):
    return [name.rstrip() for name in open(os.path.join(args.dataset,filename),'r')]
train_names,val_names,test_names = [read_names(split) for split in [args.train,args.val,args.test]]

train_data,val_data,test_data = [load_data(args.dataset,names,args.sigma) for names in [train_names,val_names,test_names]]

def add_data_to_h5(f,data,split,augment=False):
    if len(data)==0: return
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
    add_data_to_h5(f,train_data,'train',augment=args.augment)
    add_data_to_h5(f,val_data,'val')
    add_data_to_h5(f,test_data,'test')
    f.attrs['bands'] = args.bands

