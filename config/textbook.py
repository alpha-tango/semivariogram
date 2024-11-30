"""
Config file for running the Co-Kriging example from
"An Introduction to Appled Geostatistics" by Isaaks + Srivastava,
p. 405 the example begins. 
I wrote this to test the cokriging.py, so it may not work
with other drivers in this repo. 
"""

import numpy as np
import pandas as pd

from scripts.models import SphericalModel

####################################
# Data
####################################

# the column name in the raw dataset of the primary variable
PRIMARY_VAR_NAME = 'U'

# the column name in the raw dataset of the secondary variable
SECONDARY_VAR_NAME = 'V'

def raw_data():
    """
    See Fig 17.1 on p 405.
    """
    columns = ['id', 'x', 'y', 'V', 'U']
    data = [
    [1, -3, 6, 414, 741],
    [2, -8, -5, 386, 504],
    [3, 3, -3, 56, None]  # Sample 3 has no "U" data
    ]
    return pd.DataFrame(data, columns=columns)

def primary_data():
    """
    Return a dataframe containing just the primary variable
    """
    columns = ['id', 'x', 'y', PRIMARY_VAR_NAME]
    return raw_data()[columns].dropna()

def secondary_data():
    """
    Return a dataframe containing just the secondary variable
    """
    columns = ['id', 'x', 'y', SECONDARY_VAR_NAME]
    return raw_data()[columns].dropna()


####################################
# Model
####################################

# Textbook unhelpful
# Hardcoding values from Table 17.1

class DummyModel:
    
    def init(self):
        pass

    def fit_h(self):
        pass


class PrimaryModel(DummyModel):

    def fit_h(self, h):
        if len(h[0]) == 1:
            # it's the target
            return np.matrix([
                [134229],
                [102334]
                ])
        return np.matrix([
        [605000,99155],
        [99155,605000]
        ])


class SecondaryModel(DummyModel):

    def fit_h(self, h):
        return np.matrix([
        [107000, 49623, 57158],
        [49623, 107000, 45164],
        [57158, 45164, 107000]
        ])


class CrossModel(DummyModel):

    def fit_h(self, h):
        if len(h[0]) == 1:
            # it's the target
            return np.matrix([
                [70210],
                [52697],
                [75887]
                ])
        return np.matrix([
        [137000, 49715, 57615],
        [49715, 137000, 45554]
        ])


PRIMARY_MODEL = PrimaryModel()
SECONDARY_MODEL = SecondaryModel()
CROSS_MODEL = CrossModel()

####################################
# Kriging
####################################

# specify target x coordinates to be used in a Numpy meshgrid
# therefore whatever generates the variable needs to be
# compatible with meshgrid.
# Numpy linspace is another good option if you have a total
# resolution you want rather than a specific step size. 
TARGET_X_COORDS = np.arange(start=0, stop=1, step=1)  

# specify target y coordinates to be used in a Numpy meshgrid
TARGET_Y_COORDS = np.arange(start=0, stop=1, step=1) 

####################################
# Plots
####################################