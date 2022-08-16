import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import glob
import numpy as np

from models import SFANet
from utils.preprocess import preprocess

import argparse
import os
import sys

import h5py as h5

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, help='path to training data hdf5 file')
    parser.add_argument('--log', required=True, help='path to log directory')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    args = parser.parse_args()

    f = h5.File(args.data,'r')
    train_images = f['train/images'][:]
    train_confidence = f['train/confidence'][:]
    train_attention = f['train/attention'][:]
    val_images = f['val/images'][:]
    val_confidence = f['val/confidence'][:]
    val_attention = f['val/attention'][:]

    model, testing_model = SFANet.build_model(
        train_images.shape[1:],
        preprocess_fn=preprocess)
    opt = Adam(args.lr)
    model.compile(optimizer=opt, loss=['mse','binary_crossentropy'], loss_weights=[1,0.1])

    print(model.summary())
    
    os.makedirs(args.log,exist_ok=True)

    callbacks = []

    weights_path = os.path.join(args.log, 'weights.best.h5')
    callbacks.append(ModelCheckpoint(
            filepath=weights_path,
            monitor='val_loss',
            verbose=True,
            save_best_only=True,
            save_weights_only=True,
            ))
    weights_path = os.path.join(args.log, 'weights.latest.h5')
    callbacks.append(ModelCheckpoint(
            filepath=weights_path,
            monitor='val_loss',
            verbose=True,
            save_best_only=False,
            save_weights_only=True,
            ))
    tensorboard_path = os.path.join(args.log,'tensorboard')
    os.system("rm -rf " + tensorboard_path)
    callbacks.append(tf.keras.callbacks.TensorBoard(tensorboard_path))

    y_train = (train_confidence, train_attention)
    y_val = (val_confidence, val_attention)

    model.fit(
            train_images, y_train,
            validation_data=(val_images,y_val),
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=True,
            callbacks=callbacks)

if __name__ == '__main__':
    main()
