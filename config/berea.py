"""
Config settings for berea dataset
"""

######################################
# Preparing raw data for pairwise-df
######################################

import duckdb

# 

def raw_data():
    """
    Return a dataframe with the following columns:
    `id`: an id used for row counting in creating pairs.
    `x`: x location.
    `y`: y location.
    `primary`: the primary parameter to look at. 
    `secondary`: (optional) the secondary parameter.
    """
    berea_select = """
        SELECT
        row_number() OVER () AS id,
        x_mm AS x,
        y_mm AS y,
        permeability_md AS primary,
        water_resisitivity_ohmm AS secondary
        FROM read_csv('data/berea_subset2.csv', normalize_names=True)
        ORDER BY x, y ASC
        """
    return duckdb.sql(berea_select).df()


######################################
# Binning data
######################################

# set a list of max bins for custom bins
BIN_MAXES = [3,5,10,15,20,25,30,40,500]



