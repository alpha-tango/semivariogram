"""
Config settings for berea dataset
"""

import scripts.models as models
import scripts.stats as stats

import numpy as np

######################################
# image settings and functions
######################################

IM_TAG = 'pfaa'
H_UNITS = 'km'
PRIMARY_VAR_NAME = 'PFAA'

def custom_raw_plots(raw_df):
    import matplotlib.pyplot as plt

    # Plot location vs concentration
    fig, ax = plt.subplots()
    ax.scatter(raw_df['x'], raw_df['primary'])
    ax.set_title("PFAA concentration by latitude")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    plt.show()
    fig.savefig(f'images/{IM_TAG}_concentration_by_lat.png')

    fig, ax = plt.subplots()
    ax.scatter(raw_df['y'], raw_df['primary'])
    ax.set_title("PFAA concentration by longitude")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    plt.show()
    fig.savefig(f'images/{IM_TAG}_concentration_by_long.png')

    # Plot concentration vs date sampled
    fig, ax = plt.subplots()
    ax.scatter(raw_df['sampling_date'], raw_df['primary'])
    ax.set_title("PFAA concentration by sampling date")
    ax.set_xlabel("Sampling Date")
    ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    plt.show()
    fig.savefig(f'images/{IM_TAG}_concentration_by_date.png')

    # Plot concentration vs rain
    fig, ax = plt.subplots()
    ax.scatter(raw_df['rain'], raw_df['primary'])
    ax.set_title("PFAA concentration by previous rain")
    ax.set_xlabel("Rain")
    ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    plt.show()
    fig.savefig(f'images/{IM_TAG}_concentration_by_rain.png')

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
        -- LOG(
        --     CASE WHEN data.PFOA IN ('ND', '<MDL', '<RL')
        --     THEN 0.001
        --     ELSE CAST(PFOA AS FLOAT)
        --     END
        -- ) AS primary,
        -- LOG(sum_PFAA) AS primary,
        sum_PFAA as primary,
        LOG(sum_PFAA) as log_sum_PFAA,
        CASE WHEN data.rain IS NOT NULL THEN TRUE
        ELSE FALSE END AS rain,
        sampling_date,
        property_name
        FROM read_csv('data/VT_pfas_data_clean.csv', normalize_names=True) AS data
        INNER JOIN read_csv('data/VT_pfas_ID.csv', normalize_names=True) AS sample_id
        ON data.sample_id = sample_id.sample_id
        -- WHERE 
            -- data.sample_id LIKE 'E1%'
        -- data.sample_id NOT IN ('J6', 'B2', 'E5')
        -- AND data.sample_id NOT LIKE 'K6%'
        """
    return duckdb.sql(pfas_select).df()

#######################################
# Variogram settings
#######################################

#     parser.add_argument('--bin_method', choices=['config', 'fixed_width', 'equal_points', 'raw'],
        # help="""
        # Choose a method to bin the data before modeling semivariance.
        # `config`: in the appropriate config file, set a list of bin maxes.
        # `fixed_width`: bins are of fixed width. Must be used with `--bin_width` argument.
        # `equal_points`: bins have an equal number of points. Defaults to square 
        #     root of the number of data points, or can be used with `--points_per_bin` argument.
        # `raw`: does not bin the data at all.
        # """
        # )

BIN_METHOD = 'equal_points'

# set a list of max bins for custom bins
# BIN_MAXES = [3,5,10,15,20,25,30,40,500]

#######################################
# Model
#######################################

RANGE = 0.3
SILL = 0.1 * 10**8

MODEL = models.ExponentialModel(fit_range=RANGE, sill=SILL)

########################################
# Kriging 
########################################

TEST_COORDS = [
[-72.0, 44.0],
[-73.0, 45.0],
[-71.5, 42.5],
[-71.5,45.0]
]




