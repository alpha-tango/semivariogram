"""
Config settings for berea dataset
"""

import scripts.models as models
import scripts.stats as stats

import numpy as np

######################################
# image settings and functions
######################################

IM_TAG = 'log_pfaa'
H_UNITS = 'km'
PRIMARY_DISPLAY_NAME = 'log(PFAA) + 10'
SECONDARY_DISPLAY_NAME = 'log(TOC) + 10'
UNITS = 'ng/kg'

######################################
# Preparing raw data for pairwise-df
######################################

# set the distance. 
# options are 'euclidean' or 'geographic'
# default is euclidean
DISTANCE_METHOD = 'geographic'

def distance_matrix(v, u):
    # v is a list of [lon, lat] coords
    # u is a list of [lon, lat] coords
    n = len(v)
    m = len(u)
    H = np.ones((n, m))
    for i in range(n):
        for j in range(m):
            H[i,j] = stats.geographic_distance(v[i][0], u[j][0], v[i][1], u[j][1])
    return H


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

    pfas_select = """
        SELECT
        ROW_NUMBER() OVER () AS id,
        CONCAT(data.sample_id, sub_id) AS sample_id,
        property_latitude AS y,
        property_longitude AS x,
        sum_PFAA as sum_PFAA,
        LOG(sum_PFAA) + 10 AS primary,
        LOG(toc) + 10 AS secondary,
        CASE WHEN data.rain IS NOT NULL THEN TRUE
        ELSE FALSE END AS rain,
        sampling_date,
        property_name
        FROM read_csv('data/VT_pfas_data_clean.csv', normalize_names=True) AS data
        INNER JOIN read_csv('data/VT_pfas_ID.csv', normalize_names=True) AS sample_id
        ON data.sample_id = sample_id.sample_id
        WHERE data.sub_id IS NULL
        """
    return duckdb.sql(pfas_select).df()

#######################################
# Variogram settings
#######################################

BIN_METHOD = 'equal_points'

#######################################
# Model
#######################################

RANGE = 10  # set it higher to get some result
SILL = .06
NUGGET = .01

MODEL = models.ExponentialModel(fit_range=RANGE, sill=SILL, nugget=NUGGET)

CROSS_RANGE = 5
CROSS_SILL = 2.9
CROSS_NUGGET = 0

CROSS_MODEL = models.ExponentialModel(fit_range=CROSS_RANGE, sill=CROSS_SILL, nugget=CROSS_NUGGET)


########################################
# Kriging 
########################################

TEST_COORDS = [
[-72.0, 44.0],
[-73.0, 45.0],
[-71.5, 42.5],
[-71.5,45.0]
]

def transform(c):
    return np.exp(c - 10)




