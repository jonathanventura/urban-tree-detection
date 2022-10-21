""" Compute metrics on test set. """
import numpy as np
import argparse
import os
import h5py as h5
import yaml
from utils.evaluate import evaluate, make_figure
from models import SFANet
from utils.preprocess import *
import imageio
import matplotlib as mpl
mpl.use('Agg')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', help='path to data hdf5 file')
    parser.add_argument('log', help='path to log directory')
    parser.add_argument('--max_distance', type=float, default=10, help='max distance from gt to pred tree (in pixels)')

    args = parser.parse_args()

    params_path = os.path.join(args.log,'params.yaml')
    if os.path.exists(params_path):
        with open(params_path,'r') as f:
            params = yaml.safe_load(f)
            mode = params['mode']
            min_distance = params['min_distance']
            threshold_abs = params['threshold_abs'] if mode == 'abs' else None
            threshold_rel = params['threshold_rel'] if mode == 'rel' else None
    else:
        print(f'warning: params.yaml missing -- using default params')
        min_distance = 1
        threshold_abs = None
        threshold_rel = 0.2
    
    f = h5.File(args.data,'r')
    images = f[f'test/images'][:]
    gts = f[f'test/gt'][:]

    bands = f.attrs['bands']
    
    preprocess = eval(f'preprocess_{bands}')
    training_model, model = SFANet.build_model(
        images.shape[1:],
        preprocess_fn=preprocess)

    weights_path = os.path.join(args.log,'weights.best.h5')
    training_model.load_weights(weights_path)

    print('----- getting predictions from trained model -----')
    preds = model.predict(images,verbose=True,batch_size=1)[...,0]

    print('----- calculating metrics -----')
    results = evaluate(
        gts=gts,
        preds=preds,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        max_distance=args.max_distance,
        return_locs=True)

    with open(os.path.join(args.log,'results.txt'),'w') as f:
        f.write('precision: '+str(results['precision']))
        f.write('recall: '+str(results['recall']))
        f.write('fscore: '+str(results['fscore']))
        f.write('rmse [px]: '+str(results['rmse']))

    print('------- results for: ' + args.log + ' ---------')
    print('precision: ',results['precision'])
    print('recall: ',results['recall'])
    print('fscore: ',results['fscore'])
    print('rmse [px]: ',results['rmse'])
        
    fig = make_figure(images,results)
    fig.savefig(os.path.join(args.log,'figure.pdf'))

if __name__ == '__main__':
    main()
