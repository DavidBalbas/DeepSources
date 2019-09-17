# DeepSources
## Code of the project *A Deep Learning approach for the detection of point sources in the Comsic Microwave Background*

**The code is being updated. The final version will be available by Thursday 19 the latest.**

### Overview

The code is divided in three main packages, which correspond to different sections/chapters of the written report. These are: simulations of CMB and Gaussian noise at 5 arcmin (sim5min), simulations of CMB and Gaussian noise in the plane at 5 min, which may include foregrounds (sim5minplane), and realistic simulations of the 143 GHz Planck channel at 7.22 arcmin (sim7min).

Each of the modules contains its own functions, scripts and already trained CNN models. They are designed to work independently from each other, so parts of our code are often repeated in different files.

For any questions, commentaries, requests or suggestions, email me at dbg47@alumnos.unican.es.

### Required data

The script *create_folders.py* shall be run initially. This script will create the folders required to store the simulations and the patches that the training generator modules will later produce. In addition, some functionalities require a .fits map of the Galactic foregrounds, realistic instrumental noise or the actual Planck data. These can mostly be found at the [Planck Legacy Archive](https://pla.esac.esa.int/#home), although they can also be provided upon request by email.

### Method explanation and user guide

There are some common .py files in each of the packages; their behaviour differs according to the type of simulations, but the main features are equal.

##### training_generator.py:
Generates a complete training set from the CMB power spectrum coefficients and the foregrounds, if applicable. The main method is _generate__training__set()_, which produces all the patches and the full maps with the selected parameters.

##### detection_module_(stats).py:
Loads the patches from the data folders and evaluates the performance of two models, statistically. The thresholds of the methods shall be modified individually inside he code.

##### detection_module_sphere.py:
Loads the patches of a full map from the data folders and evaluates the performance of a CNN model over the sphere, drawing the locations of the detection, spurious detections and undetected sources above the chosen flux.

##### cnn.py (or related modules):
Loads the patches from the data folders and evaluates the performance of two models, statistically. The thresholds of the methods shall be modified individually inside he code.
