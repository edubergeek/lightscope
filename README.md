# lightscope
Exoplanet Finder using AI to Find Eclipses in Light Curves

# Files
## KeplerETL.py
Extract Transform Load (ETL) Kepler light curve fits files as a Tensorflow dataset

## LightScope.ipynb
Train an AI model to find exoplanet signatures in a light curve.
Predict the period of the exoplanet's revolution around the main star.

## KerasModel.py
Keras/TF neural network model base class

## KeplerModel.py
Kepler neural network model derived class
Provides Compile, Train, Evaluate methods

## MLPModel.ipynb
A basic fully connected (MLP) model

## CNNModel.ipynb
A basic Conv1D (CNN) model

## Train.ipynb
An interactive notebook to manually run a training workflow.

# ToDo List

- Right now our training set only contains the Kepler confirmed light curves
    - Add an equal sized download of light curves having no exoplanet detections
- Train a binary classifier model
- Train a Transformer model
- Train a YOLO model for each period bounding box
- Train an AutoEncoder model
- Hyperparameter search
    - Batch size
    - Learning Rate
    - Dropout
    - Loss function
    - Chunk size
- Model architecture search

