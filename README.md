## Urban Tree Detection ##

This repository provides code for training and evaluating a convolutional neural network (CNN) to detect tree in urban environments with aerial imagery.   The CNN takes multispectral imagery as input and outputs a confidence map indicating the locations of trees. The individual tree locations are found by local peak finding. In our study site in Southern California, we determined that 73.6% of the predicted trees were correct, and the method found 73.3% of the trees in the study area.

### Installation ###

The model is implemented with Tensorflow 2.4.1.  We have provided an `environment.yml` file so that you can easily create a conda environment with the dependencies installed:

    conda env create 
    conda activate urban-tree-detection

### Citation ###

If you use or build upon this repository, please cite our paper:

Ventura, J., Honsberger, M., Gonsalves, C., Rice, J., Pawlak, C., Love, N., Han, S., Nguyen, V., Sugano, K., Doremus, J., Fricker, G. A., Yost, J., Ritter, M. (2022). Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery.  (In preparation)

### Acknowledgments ###

This project was funded by CAL FIRE (award number: 8GB18415) the US Forest Service (award number: 21-CS-11052021-201), and an incubation grant from the Data Science Strategic Research Initiative at California Polytechnic State University.

