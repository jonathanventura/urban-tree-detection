import numpy as np

from models import SFANet
from utils.preprocess import preprocess
from utils.inference import run_tiled_inference

import argparse
import os
import sys

import rasterio

import tqdm
from tqdm import trange

import glob

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='path to input tiff file or directory')
    parser.add_argument('--output', required=True, help='path to output json file or directory')
    parser.add_argument('--log', required=True, help='path to log directory')
    parser.add_argument('--tile_size', type=int, default=2048, help='tile size')
    parser.add_argument('--overlap', type=int, default=32, help='overlap between tiles')

    args = parser.parse_args()
    
    weights_path = os.path.join(args.log,'weights.best.h5')
    padded_size = args.tile_size + args.overlap*2
    training_model, model = SFANet.build_model((padded_size,padded_size,4),preprocess_fn=preprocess)
    training_model.load_weights(weights_path)
    
    if os.path.isdir(args.input):
        os.makedirs(args.output,exist_ok=True)
        paths = sorted(glob.glob(os.path.join(args.input,'*.tif')) + glob.glob(os.path.join(args.input,'*.tiff')))
        pbar = tqdm.tqdm(total=len(paths))
        for input_path in paths:
            output_path = os.path.join(args.output,os.path.basename(input_path).split('.')[0]+'.json')
            if not os.path.exists(output_path):
                run_tiled_inference(model,input_path,output_path,min_distance=1,threshold_abs=None,threshold_rel=.2)
            pbar.update(1)
    else:
        run_tiled_inference(model,args.input,args.output,min_distance=1,threshold_abs=None,threshold_rel=.2)

if __name__ == '__main__':
    main()
