# DeepSources
## Code of the project *A Deep Learning approach for the detection of point sources in the Comsic Microwave Background*

### Overview

The code is divided in three main packages, which correspond to different sections/chapters of the written report. These are: simulations of CMB and Gaussian noise at 5 arcmin (sim5min), simulations of CMB and Gaussian noise in the plane at 5 min, which may include foregrounds (sim5minplane), and realistic simulations of the 143 GHz Planck channel at 7.22 arcmin (sim7min).

Each of the modules contains its own functions, scripts and already trained CNN models. They are designed to work independently from each other, so parts of our code are often repeated in different files.

The file _matched__filter__diego.py_ is the local matched filter implementation kindly provided by Diego Herranz, used with permission. We have introduced slight modifications in the code, but the core algorithm remains unchanged.

For any questions, commentaries, requests or suggestions, please email me at dbg47@alumnos.unican.es.

### Required data

The script *create_folders.py* shall be run initially. This script will create the folders required to store the simulations and the patches that the training generator modules will later produce. In addition, some functionalities require a .fits map of the Galactic foregrounds, realistic instrumental noise or the actual Planck data. These can mostly be found at the [Planck Legacy Archive](https://pla.esac.esa.int/#home), although they can also be provided upon request by email.

### Method explanation and user guide

There are some common .py files in each of the packages; their behaviour differs according to the type of simulations, but the main features are equal. All the methods correspond to one of these descriptions or a variation:

##### training_generator.py:
Generates a complete training set from the CMB power spectrum coefficients and the foregrounds, if applicable. The main method is _generate__training__set()_, which produces all the patches and the full maps with the selected parameters.

##### detection_module_(stats).py:
Loads the patches from the data folders and evaluates the performance of two models, statistically. The thresholds of the methods shall be modified individually inside he code.

##### detection_module_sphere.py:
Loads the patches of a full map from the data folders and evaluates the performance of a CNN model over the sphere, drawing the locations of the detection, spurious detections and undetected sources above the chosen flux.

##### cnn.py (or related modules):
Loads the patches from the data folders and trains the defined CNN models. The architecture should be changed in the script if required.

##### detection_module_planck.py:
Similar to the detection in the sphere but with the Planck data. The detections are no longer catalogued as spurious or true, only the total number of sources and the location map is given.

### Other files

The file _COM__PWR__CMB_ includes the coefficients (Dl) Planck best-fit CMB power spectrum. The file _foreorder.npy_ is a list of the pixel index of the HEALPix Nside=16 patches, ordered by foreground intensity. Finally, the different .hdf files are the different trained CNN modules.
