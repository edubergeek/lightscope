# lightscope
Exoplanet Finder using AI to Find Eclipses in Light Curves

# Files
## KeplerETL.py
Extract Transform Load (ETL) Kepler light curve fits files as a Tensorflow dataset

## LightScope.ipynb
Train an AI model to find exoplanet signatures in a light curve.
Predict the period of the exoplanet's revolution around the main star.

# ToDo List

- Right now we only build the training set from a small sample
    - Process all Kepler llc light curve fits files
- Right now our training set only contains the Kepler confirmed light curves
    - Add an equal sized download of light curves having no exoplanet detections
- Train a MLP binary classifier model
- Train a Conv1D binary classifier model
- Train a Transformer binary classifier model
- Train a MLP regression model for period
- Train a Conv1D regression model for period
- Train a Transformer regression model for period
- Train a YOLO model for each period bounding box

