import numpy as np

import argparse
import os
import sys

import rasterio
import rasterio.transform

from skimage.feature import peak_local_max

import tempfile

import geopandas as gpd

def _tiled_inference(model,input_path,output_path,tile_size,overlap):
    with rasterio.open(input_path,'r') as src:
        meta = src.meta
        height = meta['height']
        width = meta['width']
        nodata = meta['nodata']
        
        padded_size = tile_size+overlap*2
        
        meta['count'] = 1
        meta['dtype'] = 'float32'
        with rasterio.open(output_path,'w',**meta) as dest:
        
            for row in range(overlap,height-overlap,tile_size):
                for col in range(overlap,width-overlap,tile_size):
                    window = rasterio.windows.Window(col-overlap,row-overlap,padded_size,padded_size)
                    image = src.read(window=window)
                    image = np.expand_dims(np.transpose(image,[1,2,0]),axis=0)
                    
                    down_pad = max(0,padded_size-image.shape[1])
                    right_pad = max(0,padded_size-image.shape[2])
                    image = np.pad(image,((0,0),(0,down_pad),(0,right_pad),(0,0)))
            
                    output = model.predict(image,verbose=False)
                    
                    # zero out "no data" pixels
                    mask = np.all(image==nodata,axis=-1)
                    output[mask] = 0

                    output_crop = output[0,overlap:-overlap,overlap:-overlap,0]

                    h = min(height-row,output_crop.shape[0])
                    w = min(width-col,output_crop.shape[1])
                    window = rasterio.windows.Window(col,row,w,h)
                    dest.write(output_crop[None,:h,:w],window=window)

def _tiled_peak_finding(path,input_size,overlap,min_distance,threshold_abs,threshold_rel):
    with rasterio.open(path,'r') as f:
        meta = f.meta
        height = meta['height']
        width = meta['width']
        
        padded_size = input_size+overlap*2
        
        all_indices = []
        
        for row in range(overlap,height-overlap,input_size):
            for col in range(overlap,width-overlap,input_size):
                window = rasterio.windows.Window(col-overlap,row-overlap,padded_size,padded_size)
                image = np.squeeze(f.read(1,window=window))
                
                indices = peak_local_max(image,min_distance=min_distance,threshold_abs=threshold_abs,threshold_rel=threshold_rel)
                
                good = np.all(np.stack([
                    indices[:,0] >= overlap,
                    indices[:,0] < overlap+input_size,
                    indices[:,1] >= overlap,
                    indices[:,1] < overlap+input_size],
                    axis=-1),axis=-1)
                indices = indices[good]
                indices[:,0] += row-overlap
                indices[:,1] += col-overlap
                
                all_indices.append(indices)
        all_indices = np.concatenate(all_indices,axis=0)
        return all_indices

def run_tiled_inference(model,input_path,output_path,min_distance,threshold_abs,threshold_rel):
    temp_path = tempfile.NamedTemporaryFile(suffix='.tif').name
    _tiled_inference(
        model=model,
        input_path=input_path,
        output_path=temp_path,
        tile_size=2048,
        overlap=32)

    with rasterio.open(temp_path,'r') as f:
        meta = f.meta
        epsg = meta['crs'].to_epsg()
        crs = f'EPSG:{epsg}'
        transform = meta['transform']

    indices = _tiled_peak_finding(temp_path,input_size=256,overlap=32,min_distance=min_distance,threshold_abs=threshold_abs,threshold_rel=threshold_rel)

    x,y = rasterio.transform.xy(transform,indices[:,0],indices[:,1])

    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x,y),crs=crs)
    gdf.to_file(output_path,driver='GeoJSON')

    os.remove(temp_path)

