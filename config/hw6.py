"""
Config settings for berea dataset
"""

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

    berea_select = """
        SELECT
        id,
        x,
        y,
        na_mgl AS primary
        FROM read_csv('data/hw6.csv', normalize_names=True)
        ORDER BY x, y ASC
        """
    return duckdb.sql(berea_select).df()


######################################
# Binning data
######################################

# set a list of max bins for custom bins
# BIN_MAXES = []



