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
PRIMARY_VAR_NAME = 'total PFAA'
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

# def raw_data():
#     import duckdb

#     pfas_select = """
#         SELECT
#         ROW_NUMBER() OVER () AS id,
#         CONCAT(data.sample_id, sub_id) AS sample_id,
#         property_latitude AS x,
#         property_longitude AS y,
#         CASE WHEN data.rain IS NOT NULL THEN TRUE
#         ELSE FALSE END AS rain,
#         sampling_date,
#         property_name,
#         PFPeA,


#         FROM read_csv('data/VT_pfas_data_clean.csv', normalize_names=True) AS data
#         INNER JOIN read_csv('data/VT_pfas_ID.csv', normalize_names=True) AS sample_id
#         ON data.sample_id = sample_id.sample_id
#         WHERE 
#         data.PFOA NOT IN ('ND', '<MDL', '<RL')
#         AND data.sample_id NOT IN ('J6', 'B2', 'E5', 'K6')
#         """
#     return duckdb.sql(pfas_select).df()


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

RANGE = 50  # set it higher to get some result
SILL = .1
NUGGET = 10

MODEL = models.ExponentialModel(fit_range=RANGE, sill=SILL, nugget=10)

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




