## Urban Tree Detection ##

This repository provides code for training and evaluating a convolutional neural network (CNN) to detect tree in urban environments with aerial imagery.   The CNN takes multispectral imagery as input and outputs a confidence map indicating the locations of trees. The individual tree locations are found by local peak finding. In our study site in Southern California, we determined that, using our trained model, 73.6% of the detected trees matched to actual trees, and 73.3% of the trees in the study area were detected.

### Installation ###

The model is implemented with Tensorflow 2.4.1.  We have provided an `environment.yml` file so that you can easily create a conda environment with the dependencies installed:

    conda env create 
    conda activate urban-tree-detection

### Dataset ###

The data used in our paper can be found in [a separate Github repository](https://github.com/jonathanventura/urban-tree-detection-data/).

To prepare a dataset for training and testing, run the `prepare.py` script.  You can specify the bands in the input raster using the `--bands` flag (currently `RGB` and `RGBN` are supported.)

    python3 -m scripts.prepare <path to dataset> <path to hdf5 file> --bands RGBN

### Training ###

To train the model, run the `train.py` script.

    python3 -m scripts.train <path to hdf5 file> <path to log directory>

### Hyperparameter tuning ###

The model outputs a confidence map, and we use local peak finding to isolate individual trees.  We use the Optuna package to determine the optimal parameters of the peaking finding algorithm.  We search for the best of hyperparameters to maximize F-score on the validation set.

    python3 -m scripts.tune <path to hdf5 file> <path to log directory>

### Evaluation on test set ###

Once hyperparameter tuning finishes, use the `test.py` script to compute evaluation metrics on the test set.

    python3 -m scripts.test <path to hdf5 file> <path to log directory> 

### Inference on a large raster ###

To detect trees in rasters and produce GeoJSONs containing the geo-referenced trees, use the `inference.py` script.  The script can process a single raster or a directory of rasters.

    python3 -m scripts.inference <input tiff or directory> \
                                 <output json or directory> \
                                 <path to log directory>

### Pre-trained weights ###

[Pre-trained weights](https://cpslo-my.sharepoint.com/:u:/g/personal/jventu09_calpoly_edu/EUMJLY3xLt1KnJA-gu6T1boBdWzzPHzJSKoUxRNKyiZDrg?e=vGfvFn) for a model trained on 60cm NAIP 2020 imagery from Southern California are available.  The `pretrained` directory should be used as the log directory for the `inference` script.  

We also provide an [example NAIP 2020 tile from Los Angeles](https://cpslo-my.sharepoint.com/:i:/g/personal/jventu09_calpoly_edu/EU1xfporUiBDvT2ZOpW0raEBOqJcJQpqcOv1lKNMCgbCdQ?e=zsgxXs) and an [example GeoJSON predictions file](https://cpslo-my.sharepoint.com/:u:/g/personal/jventu09_calpoly_edu/EaoRmlVJ4hRKhf2-LLeg-r4BwDM4bSUz5NI3P3ydIWs7kA?e=ZvbPFT).

You can explore a [map of predictions for the entire urban reserve of California](https://jventu09.users.earthengine.app/view/urban-tree-detector) (based on NAIP 2020 imagery) created using this pre-trained model.

### Using your own data ###

To train on your own data, you will need to organize the data into the format expected by `prepare.py`.

* The image crops (or "chips") should all be the same size and the side length should be a multiple of 32.
* The code is currently designed for three-band (RGB) or four-band (red, green, blue, near-IR) imagery.  To handle more bands, you would need to add an appropriate preprocessing function in `utils/preprocess.py`.  If RGB are not in the bands, then `models/VGG.py` would need to be modified, as the code expects the first three bands to be RGB to match the pre-trained weights.
* Store the images as TIFF or PNG files in a subdirectory called `images`.
* For each image, store a csv file containing x,y coordinates for the tree locations in a file `<name>.csv` where `<name>.tif`, `<name>.tiff`, or `<name>.png` is the corresponding image. The csv file should have a single header line.
* Create the files `train.txt`, `val.txt`, and `test.txt` to specify the names of the files in each split.

### Citation ###

If you use or build upon this repository, please cite our paper:

J. Ventura, C. Pawlak, M. Honsberger, C. Gonsalves, J. Rice, N.L.R. Love, S. Han, V. Nguyen, K. Sugano, J. Doremus, G.A. Fricker, J. Yost, and M. Ritter. ["Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery."](https://doi.org/10.48550/arXiv.2208.10607)  arXiv:2208.10606 [cs], Oct. 2022.

### Acknowledgments ###

This project was funded by CAL FIRE (award number: 8GB18415) the US Forest Service (award number: 21-CS-11052021-201), and an incubation grant from the Data Science Strategic Research Initiative at California Polytechnic State University.
