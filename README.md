## Urban Tree Detection ##

This repository provides code for training and evaluating a convolutional neural network (CNN) to detect tree in urban environments with aerial imagery.   The CNN takes multispectral imagery as input and outputs a confidence map indicating the locations of trees. The individual tree locations are found by local peak finding. In our study site in Southern California, we determined that, using our trained model, 73.6% of the detected trees matched to actual trees, and 73.3% of the trees in the study area were detected.

### Installation ###

The model is implemented with Tensorflow 2.4.1.  We have provided an `environment.yml` file so that you can easily create a conda environment with the dependencies installed:

    conda env create 
    conda activate urban-tree-detection

### Dataset ###

The data used in our paper can be found in [a separate Github repository](https://github.com/jonathanventura/urban-tree-detection-data/).

To prepare a dataset for training and testing, run the `prepare.py` script.

    python3 -m scripts.prepare --dataset <path to dataset> \
                       --output <path to hdf5 file>

### Training ###

To train the model, run the `train.py` script.

    python3 -m scripts.train --data <path to hdf5 file> \
                     --log <path to log directory> \

### Hyperparameter tuning ###

The model outputs a confidence map, and we use local peak finding to isolate individual trees.  We use the Optuna package to determine the optimal parameters of the peaking finding algorithm.  We search for the best of hyperparameters to maximize F-score on the validation set.

    python3 -m scripts.tune --data <path to hdf5 file> \
                    --log <path to log directory>

### Evaluation on test set ###

Once hyperparameter tuning finishes, use the `test.py` script to compute evaluation metrics on the test set.

    python3 -m scripts.test --data <path to hdf5 file> \
                    --log <path to log directory> 

### Using your own data ###

To train on your own data, you will need to organize the data into the format expected by `prepare.py`.

* The image crops (or "chips") should all be the same size and the side length should be a multiple of 32.
* The code is currently designed for four-band imagery (red, green, blue, near-IR).  The code would need to be modified to handle a different set of bands.
* Store the images as .tif files in a subdirectory called `images`.
* For each image, store a csv file containing x,y coordinates for the tree locations in a file `<name>.csv` where `<name>.tif` is the corresponding image.
* Create the files `train.txt`, `val.txt`, and `test.txt` to specify the names of the files in each split.

### Citation ###

If you use or build upon this repository, please cite our paper:

Ventura, J., Honsberger, M., Gonsalves, C., Rice, J., Pawlak, C., Love, N., Han, S., Nguyen, V., Sugano, K., Doremus, J., Fricker, G. A., Yost, J., Ritter, M. (2022). Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery.  (In preparation)

### Acknowledgments ###

This project was funded by CAL FIRE (award number: 8GB18415) the US Forest Service (award number: 21-CS-11052021-201), and an incubation grant from the Data Science Strategic Research Initiative at California Polytechnic State University.

