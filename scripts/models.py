"""
SemivarianceModel is the general model class that the specific models
(Spherical, Gaussian, etc.) inherit from.
Each specific model has its own class.

This is a little different (worse, probably) from normal model implementations,
such as Sklearn, in that I include the dataframe in the model. 
"""

import duckdb
import numpy as np
from sklearn.isotonic import isotonic_regression


#############################
# Functions
#############################

def get_model(model):
    return SUPPORTED_MODELS[model]

#############################
# Classes
#############################


class SemivarianceModel:
    """
    Fit a model to (h, semivariance) pairs within the range specified.
    Takes in a dataframe (df) with the following columns:
    - `h`: the lag distance
    - `semivariance`: the semivariance
    - `n`: number/count of points represented by that (h, semivariance) pair. 
           If using the raw data, this could be one. If using binned data,
           this would be > 1.
    You must also pass in a range (`a` in the De Marsily text) within which you 
    want to fit the data.
    """

    def __init__(self, data_df, fit_range):
        # set the params
        self.data = data_df
        self.a = fit_range

        # get the sill, bc it's needed for almost every model
        # and we only want to calculate it once
        self.sill = self.get_sill(data_df)
        self.name = self.get_name()
        
        # TODO: add a column validity check above the sill setter

    def get_name(self):
        """
        Name of the model -- this is a placeholder for specific 
        model classes
        """
        pass

    def get_sill(self, data_df):
        """
        Calculate the sill from the data.
        """
        sill_select = f"""
        SELECT
            AVG(semivariance) AS sill
        FROM data_df
        WHERE h >= {self.a}
        """

        return duckdb.sql(sill_select).fetchnumpy()['sill'][0]

    def max_h(self, data_df):
        """
        Find the maximum lag in the data
        """
        max_lag_select = f"""
        SELECT
            MAX(h) as max_h
        FROM data_df
        WHERE h >= {self.a}
        """
        return duckdb.sql(max_lag_select).fetchnumpy()['max_h'][0]

    def fit_h(self, h):
        """
        This is specific to the model, so this is just a placeholder.
        Comment needed. 
        `h` is a vector of lags. If no argument is passed, then 
        the model's data is used.
        """
        pass 

    def plottable(self):
        """
        Fit a series of `h` values.
        Returns a dataframe with columns `h` and `semivariance`,
        with the same number of datapoints as the input h.
        """
        plottable_h = np.linspace(0, self.max_h(self.data), 100)
        return {
            'h': plottable_h,
            'semivariance': self.fit_h(plottable_h)
        }


class ExponentialModel(SemivarianceModel):
    """
    Exponential model that inherits from the general Semivariance Model
    functions.
    """

    def fit_h(self, h):
        """
        De Marsily:
        w * ( 1 - exp(-h/a))
        Vectorized implementation.
        Default is to use the model data, but you can pass in 
        any vector of distances. 
        """
        return self.sill * (1 - np.exp( -h / self.a))

    def get_name(self):
        return 'Exponential'




def get_fit_info(fit_dataframe, h_col, gamma_col, fit_range):
    """
    Get info / stats to fit the models with
    """
    fit_info_select = f"""
        SELECT
            AVG({gamma_col}) AS sill,
            MAX({h_col}) AS max_h
        FROM fit_dataframe
        WHERE {h_col} >= {fit_range}
        """

    return duckdb.sql(fit_info_select).fetchnumpy()

def fit_model(model_type, fit_dataframe, h_col, gamma_col, fit_range):
    """
    Fit a model on the given x, y data, cutting off at the range.
    Return plottable x, y values.

    fit_dataframe is a dataframe that you want to fit. 
    h_col is the name of the column that contains the lags.
    gamma_col is the name of the column that contains the individual semivariances.
    fit_range is a float.
    """

    ######################################
    # get needed info for fitting models
    ######################################

    fit_info = get_fit_info(fit_dataframe, h_col, gamma_col, fit_range)
    
    sill = fit_info['sill'][0]
    max_h = fit_info['max_h'][0]

    ##################################
    # actual model code 
    ##################################
    def spherical(h):
        """
        De Marsily:
        w * ((3/2) * (h/a) - (1/2)*(h/a)**3)  for h < a
        w   for h > a (I implement as h >= a)
        where w = sill
        a = range,
        h = lag.
        """
        if h >= fit_range:
            return sill
        else:
            return sill * ((3/2) * (h/fit_range) - (1/2)*(h/fit_range)**3)

    def exponential(h):
        """
        De Marsily:
        w * ( 1 - exp(-h/a))
        """
        return sill * (1 - np.exp(-h/fit_range))

    def gaussian(h):
        """
        De Marsily:
        w * (1 - exp(-(h/a)**2))
        """
        return sill * (1 - np.exp(-(h/fit_range)**2))

    ####################################################
    # create x,y vectors to plot the modeled line with
    ####################################################

    # create a list of 100 evenly-spaced x values to plot our model line against
    # match the max lag of the raw semivariances for nice plotting
    # TODO: is 0 always a good start point?
    fit_x = np.linspace(0, max_h, 100)

    # make a corresponding list of y-values of the modeled line

    if model_type == 'spherical':
        fit_y = list(map(spherical, fit_x))
    elif model_type == 'exponential':
        fit_y = list(map(exponential, fit_x))
    elif model_type == 'gaussian':
        fit_y = list(map(gaussian, fit_x))
    elif model_type == 'isotonic':
        # we want just the lags
        fit_x = fit_dataframe[h_col]
        fit_y = isotonic_regression(
                y=fit_dataframe[gamma_col],
                sample_weight=fit_dataframe['n'],  # this is only true in Berea!! 
                y_min=0,
                y_max=sill,
                increasing=True)  # this is default but just to be clear
    else:
        print("bad outcome")

    return {'fit_x': fit_x,
            'fit_y': fit_y,
            'sill': sill,
            'range': fit_range}

#############################
# variables
#############################

SUPPORTED_MODELS = {
    'exponential': ExponentialModel
}
