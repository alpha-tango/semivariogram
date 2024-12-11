"""
Config settings for berea dataset
"""

import scripts.stats as stats
from scripts.models import ExponentialModel

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

######################################
# image settings and functions
######################################

IM_TAG = 'pfaa'
H_UNITS = 'km'
UNITS = 'ng/kg'

def custom_raw_plots(raw_df):
    import matplotlib.pyplot as plt

    # # Plot location vs concentration
    # fig, ax = plt.subplots()
    # ax.scatter(raw_df['x'], raw_df['primary'])
    # ax.set_title("PFAA concentration by latitude")
    # ax.set_xlabel("Latitude")
    # ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    # plt.show()
    # fig.savefig(f'images/{IM_TAG}_concentration_by_lat.png')

    # fig, ax = plt.subplots()
    # ax.scatter(raw_df['y'], raw_df['primary'])
    # ax.set_title("PFAA concentration by longitude")
    # ax.set_xlabel("Longitude")
    # ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    # plt.show()
    # fig.savefig(f'images/{IM_TAG}_concentration_by_long.png')

    # # Plot concentration vs date sampled
    # fig, ax = plt.subplots()
    # ax.scatter(raw_df['sampling_date'], raw_df['primary'])
    # ax.set_title("PFAA concentration by sampling date")
    # ax.set_xlabel("Sampling Date")
    # ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    # plt.show()
    # fig.savefig(f'images/{IM_TAG}_concentration_by_date.png')

    # # Plot concentration vs rain
    # fig, ax = plt.subplots()
    # ax.scatter(raw_df['rain'], raw_df['primary'])
    # ax.set_title("PFAA concentration by previous rain")
    # ax.set_xlabel("Rain")
    # ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    # plt.show()
    # fig.savefig(f'images/{IM_TAG}_concentration_by_rain.png')

    # Plot PFAA vs TOC
    fig, ax = plt.subplots()
    ax.scatter(raw_df['secondary'], raw_df['primary'])
    ax.set_title("PFAA concentration by TOC concentration")
    ax.set_xlabel("TOC (%)")
    ax.set_ylabel("PFAA Concentration (ng/kg dry soil weight)")

    #calculate equation for trendline
    z = np.polyfit(raw_df['secondary'], raw_df['primary'], 1)
    p = np.poly1d(z)

    #add trendline to plot
    plt.plot(raw_df['secondary'], p(raw_df['secondary']), color='black')

    # add textbox in upper left with line equation
    textstr = f"PFAA={z[0]:.2f}(TOC)+{z[1]:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.show()
    fig.savefig(f'images/{IM_TAG}_pfaa_by_toc.png')

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

# the column name in the raw dataset of the primary variable 
PRIMARY_VAR_NAME = 'sum_pfaa'

# the column name in the raw dataset of the secondary variable (for Co-Kriging)
SECONDARY_VAR_NAME = 'toc'

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
        sum_PFAA as primary,
        sum_PFAA,
        TOC as secondary,
        TOC,
        CASE WHEN data.rain IS NOT NULL THEN TRUE
        ELSE FALSE END AS rain,
        sampling_date,
        property_name
        FROM read_csv('data/VT_pfas_data_clean.csv', normalize_names=True) AS data
        INNER JOIN read_csv('data/VT_pfas_ID.csv', normalize_names=True) AS sample_id
        ON data.sample_id = sample_id.sample_id
        WHERE 
        data.sub_id IS NULL
        """
    return duckdb.sql(pfas_select).df()

def primary_data():
    """
    Return a dataframe containing just the primary variable.
    Filter out half of results so we can do cross-validation.
    """

    # define which columns we need
    columns = ['id', 'x', 'y', PRIMARY_VAR_NAME]

    # select just those column and drop missing values
    print(raw_data())
    primary = raw_data()[columns].dropna()

    # filter out half the results (by row number) 
    # so that we artificially have more secondary data
    # and so we can later do cross-validation.
    return primary[primary['id']%2 == 0]

def primary_transform(v):
    """
    PFAA: Take in data and apply a log transform for normality.
    Then add ten so values are more similar in magnitude
    and we have fewer tiny values that could cause
    matrix multiplication issues.
    """
    return np.log(v) + 10

def primary_backtransform_error(v):
    return np.exp(v - 10)

def secondary_data():
    """
    Return a dataframe containing just the secondary variable,
    TOC.
    """
    columns = ['id', 'x', 'y', SECONDARY_VAR_NAME]
    return raw_data()[columns].dropna()

def secondary_transform(v):
    """
    TOC: Apply a log transform for normality.
    """
    return np.log(v) + 10

# no need for secondary inverse transform
# since we are only estimating the primary


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

# RANGE = 0.3
# SILL = 0.1 * 10**8

# copy co-kriging
RANGE = 30
SILL = 0.06
NUGGET = 0.01

MODEL = ExponentialModel(fit_range=RANGE, sill=SILL, nugget=NUGGET)

########################################
# Kriging 
########################################

# specify particular coordinates to be used with "--test" command line option
# rather than a full meshgrid.
TEST_COORDS = [
[-72.0, 44.0],
[-73.0, 45.0],
[-71.5, 42.5],
[-71.5,45.0]
]

# specify target x coordinates to be used in a Numpy meshgrid
# therefore whatever generates the variable needs to be
# compatible with meshgrid.
# Numpy linspace is another good option if you have a total
# resolution you want rather than a specific step size. 
TARGET_X_COORDS = np.arange(start=-73.5, stop=-71.25, step=.1)  

# specify target y coordinates to be used in a Numpy meshgrid
TARGET_Y_COORDS = np.arange(start=42.5, stop=45.25, step=.1) 

def kriging_validation_plots(raw_df, validation_data):
    vermont_shape = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/vermont-counties.geojson")
    # Convert data to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        raw_df,
        geometry=gpd.points_from_xy(raw_df['x'], raw_df['y']),
        crs="EPSG:4326"
    )

    ############### Co-Kriged Estimates with Vermont map and sample points overlaid #########################
    # Plot the Vermont map
    fig, ax = plt.subplots(figsize=(10, 10))
    vermont_shape.plot(ax=ax, color='white', edgecolor='black')

    # Plot original points for reference
    ax.scatter(raw_df['x'], raw_df['y'], c='black', marker='x', label='Kriged Primary Points')
    plt.scatter(validation_data['x'], validation_data['y'], c=validation_data['difference'], label='Difference Between Estimated and Actual', cmap='YlOrRd')
    plt.colorbar(label=f"Difference ({UNITS})")
    plt.legend()

    # Add titles and labels
    plt.title(f"Sum PFAA: Kriged Estimate vs. Actual")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    fig.savefig(f"images/{IM_TAG}_kriging_custom_validation.png")



##########################################
# Co-Kriging
##########################################

# Be sure that models here correspond with the data transforms
# Defined above -- sill and range will be different than if untransformed.

# gamma = np.sum((x - y)**2) / (2*n)
# C = np.sum((x - mean_x)*(y- mean_y)) / n

class CovarianceExponentialModel:
    
    def __init__(self, a:float, w:float, z:float):
        self.a = a  # this is the range
        self.w = w  # this is the sill
        self.z = z  # this is the nugget

    def fit_h(self, h):
        """
        De Marsily, Semivariance:
        gamma(h) = w * ( 1 - exp(-h/a)) + nugget

        Rizzo, Covariance. 

        c1 = sill - nugget
        C(h) = c1 * exp(-h/a) 


        See scripts/models.py for Semivariogram implementation.
        """
        c1 = self.w - self.z

        return c1 * np.exp(-h/self.a)


# the range, sill, and nugget are based on semivariogram choices
# for the transformed data: log(pfaa) + 10
primary_range = 30
primary_sill = 0.06
primary_nugget = 0.01
PRIMARY_MODEL = CovarianceExponentialModel(a=primary_range, w=primary_sill, z=primary_nugget)


# the range, sill, and nugget are based on semivariogram choices
# for the transformed data: log(toc) + 10
secondary_range = 30
secondary_sill = 0.01  
secondary_nugget = 0.0
SECONDARY_MODEL = CovarianceExponentialModel(a=secondary_range, w=secondary_sill, z=secondary_nugget)

# the range, sill, and nugget are based on semivariogram choices
# for the transformed data
cross_range = 5
cross_sill = 2.90
cross_nugget = 0.0
CROSS_MODEL = CovarianceExponentialModel(a=cross_range, w=cross_sill, z=cross_nugget)

def co_kriging_custom_plots(known_sample_primary, known_sample_secondary, target_X, target_Y, E, R):

    vermont_shape = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/vermont-counties.geojson")
    # Convert data to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        known_sample_primary,
        geometry=gpd.points_from_xy(known_sample_primary['x'], known_sample_primary['y']),
        crs="EPSG:4326"
    )

    ############### Co-Kriged Estimates with Vermont map and sample points overlaid #########################
    # Plot the Vermont map
    fig, ax = plt.subplots(figsize=(10, 10))
    vermont_shape.plot(ax=ax, color='white', edgecolor='black')

    # Plot the interpolated grid using contourf
    contour = ax.contourf(
        target_X, target_Y, E,
        levels=10,         # Number of contour levels
        cmap='YlOrRd',     # Colormap
        alpha=0.5          # Transparency
    )
    ax.clabel(contour, fontsize=4, colors='dimgray')

    # Add a color bar for soil values
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Sum PFAA')

    # Plot original points for reference
    # ax.scatter(known_sample_primary['x'], known_sample_secondary['y'], c='np.log(merged_df['PFOA']+100)', label='Primary Sample Points', cmap='YlOrRd')
    ax.scatter(known_sample_primary['x'], known_sample_primary['y'], c='black', marker='x', label='Primary Sample Points')
    ax.scatter(known_sample_secondary['x'], known_sample_secondary['y'], c='red', marker='+', label='Secondary Sample Points')
    plt.legend()

    # Add titles and labels
    plt.title("Sum PFAA Values Contour Map - Vermont")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    fig.savefig(f"images/{IM_TAG}_co_kriging_custom_vt.png")

    ############### Co-Kriged Error Variance with Vermont map and sample points overlaid #########################
    # Plot the Vermont map
    fig, ax = plt.subplots(figsize=(10, 10))
    vermont_shape.plot(ax=ax, color='white', edgecolor='black')

    # Plot the interpolated grid using contourf
    contour = ax.contourf(
        target_X, target_Y, R,
        levels=10,         # Number of contour levels
        cmap='YlOrRd',     # Colormap
        alpha=0.5          # Transparency
    )
    ax.clabel(contour, fontsize=4, colors='dimgray')

    # Add a color bar for soil values
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Error Variance')

    # Plot original points for reference
    # ax.scatter(known_sample_primary['x'], known_sample_secondary['y'], c='np.log(merged_df['PFOA']+100)', label='Primary Sample Points', cmap='YlOrRd')
    ax.scatter(known_sample_primary['x'], known_sample_primary['y'], c='black', marker='x', label='Primary Sample Points')
    ax.scatter(known_sample_secondary['x'], known_sample_secondary['y'], c='red', marker='+', label='Secondary Sample Points')
    plt.legend()

    # Add titles and labels
    plt.title("Error Variance Contour Map - Vermont")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    fig.savefig(f"images/{IM_TAG}_co_kriging_custom_error_vt.png")

#########################################################
# Validation
#########################################################

def validation_data():
    """
    Return a dataframe containing just the primary variable
    validation set.
    This one uses the other half of data removed from the 
    primary_data() function above. The secondary_data()
    function pulled in all the sample points, so we do
    already have secondary data at every single one of these
    validation points.
    """

    # define which columns we need
    columns = ['id', 'x', 'y', PRIMARY_VAR_NAME]

    # select just those column and drop missing values
    primary = raw_data()[columns].dropna()

    # filter out the half of the results
    # that we used to implement the co-kriging
    return primary[primary['id']%2 == 1]

def co_kriging_validation_plots(known_sample_primary, known_sample_secondary, validation_data):

    vermont_shape = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/vermont-counties.geojson")
    # Convert data to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        known_sample_primary,
        geometry=gpd.points_from_xy(known_sample_primary['x'], known_sample_primary['y']),
        crs="EPSG:4326"
    )

    ############### Co-Kriged Estimates with Vermont map and sample points overlaid #########################
    # Plot the Vermont map
    fig, ax = plt.subplots(figsize=(10, 10))
    vermont_shape.plot(ax=ax, color='white', edgecolor='black')

    # Plot original points for reference
    ax.scatter(known_sample_primary['x'], known_sample_primary['y'], c='black', marker='x', label='Kriged Primary Points')
    plt.scatter(validation_data['x'], validation_data['y'], c=validation_data['difference'], label='Difference Between Estimated and Actual', cmap='YlOrRd')
    plt.colorbar(label=f"Difference ({UNITS})")
    plt.legend()

    # Add titles and labels
    plt.title(f"Sum PFAA: Co-Kriged Estimate vs. Actual")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    fig.savefig(f"images/{IM_TAG}_co_kriging_custom_validation.png")








