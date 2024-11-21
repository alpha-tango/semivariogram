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
    [2, -8, 5, 386, 504],
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

# models are given on p. 408
# I'm ignoring the anistropy so these are not quite the same
# also I'm very confused by the ranges

primary_sill = 80000  # between 70000 and 95000
primary_range = 30
primary_nugget = 440000

PRIMARY_MODEL = SphericalModel(fit_range=primary_range, sill=primary_sill, nugget=primary_nugget)

secondary_sill = 42000  # between 40000 and 45000
secondary_range = 30
secondary_nugget = 22000

SECONDARY_MODEL = SphericalModel(fit_range=secondary_range, sill=secondary_sill, nugget=secondary_nugget)

cross_sill = 45000  # between 50000 and 40000
cross_range = 30
cross_nugget = 47000

CROSS_MODEL = SphericalModel(fit_range=cross_range, sill=cross_sill, nugget=cross_nugget)

####################################
# Kriging
####################################

# specify target x coordinates to be used in a Numpy meshgrid
# therefore whatever generates the variable needs to be
# compatible with meshgrid.
# Numpy linspace is another good option if you have a total
# resolution you want rather than a specific step size. 
TARGET_X_COORDS = np.arange(start=-1, stop=1, step=1)  

# specify target y coordinates to be used in a Numpy meshgrid
TARGET_Y_COORDS = np.arange(start=-1, stop=1, step=1) 

####################################
# Plots
####################################