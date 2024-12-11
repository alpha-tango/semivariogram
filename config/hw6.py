"""
Config settings for berea dataset
"""

import scripts.models as models

######################################
# image settings and functions
######################################

IM_TAG = 'hw6'

def custom_raw_plots(raw_df):
    pass


######################################
# Preparing raw data for pairwise-df
######################################

def raw_data():
    """
    Return a dataframe with the following columns:
    `id`: an id used for row counting in creating pairs.
    `x`: x location.
    `y`: y location.
    `primary`: the primary parameter to look at. 
    `secondary`: (optional) the secondary parameter.
    """
    import duckdb

    sample_select = """
        SELECT
        id,
        x,
        y,
        na_mgl AS primary
        FROM read_csv('data/hw6.csv', normalize_names=True)
        ORDER BY id ASC
        """
    return duckdb.sql(sample_select).df()

######################################
# Model
######################################

RANGE = 10
SILL = 15000

model_class = models.get_model('hw6')
MODEL = model_class(fit_range=RANGE, sill=SILL)

######################################
# Kriging
######################################

# use these coordinates for testing
TEST_COORDS = [[7,14]]
# TEST_COORDS = [[5,5], [7, 14], [100,100], [1000,1000]]
# TEST_COORDS = [[7, 14], [1,2], [21,5], [24,24], [15,15], [10,21], [100,100], [1000,1000]]


######################################
# Binning data
######################################

# set a list of max bins for custom bins
# BIN_MAXES = []



