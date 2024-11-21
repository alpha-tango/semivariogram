"""
SemivarianceModel is the general model class that the specific models
(Spherical, Gaussian, etc.) inherit from.
Each specific model has its own class.

This is a little different (worse, probably) from normal model implementations,
such as Sklearn, in that I include the dataframe in the model. 
"""

import duckdb
import numpy as np
from sklearn.isotonic import IsotonicRegression


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

    def __init__(self, fit_range, sill, nugget = 0):
        # set the params
        self.a = fit_range

        # get the sill, bc it's needed for almost every model
        # and we only want to calculate it once
        self.sill = sill
        self.name = self.get_name()
        self.nugget = 0
        
        # TODO: add a column validity check above the sill setter

    def set_data(self, data_df):
        self.data = data_df

    def get_name(self):
        """
        Name of the model -- this is a placeholder for specific 
        model classes
        """
        pass

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


class LinearModel(SemivarianceModel):
    """
    """

    def fit_h(self, h):
        """
        """
        linear_model = np.polyfit(self.data['h'], self.data['semivariance'], 1)
        linear_model_fn = np.poly1d(linear_model)
        return linear_model_fn(h)

    def get_name(self):
        return 'Linear'


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

class SphericalModel(SemivarianceModel):

    def fit_h(self, h):
        """
        De Marsily:
        w * ((3/2) * (h/a) - (1/2)*(h/a)**3)  for h < a
        w   for h > a (I implement as h >= a)
        where w = sill
        a = range,
        h = lag.
        """
        if h >= self.range:
            return self.sill + self.nugget
        else:
            return self.sill * ((3/2) * (h/self.a) - (1/2) * (h/self.a)**3) + nugget


###########################
# Code to deprecate
###########################

    # def spherical(h):

    #     if h >= fit_range:
    #         return sill
    #     else:
    #         return sill * ((3/2) * (h/fit_range) - (1/2)*(h/fit_range)**3)

    # def gaussian(h):
    #     """
    #     De Marsily:
    #     w * (1 - exp(-(h/a)**2))
    #     """
    #     return sill * (1 - np.exp(-(h/fit_range)**2))



class HW6Model(ExponentialModel):
    """
    Exponential model that inherits from the general Exponential Model
    functions.
    """

    def fit_h(self, h):
        """
        De Marsily:
        w * ( 1 - exp(-h/a))
        Vectorized implementation.
        Default is to use the model data, but you can pass in 
        any vector of distances. 

        HW6:
        15000 * (1 - exp(-3|h| / 10))
        """
        return self.sill * (1 - np.exp( -3*h / self.a))


#############################
# variables
#############################

SUPPORTED_MODELS = {
    'exponential': ExponentialModel,
    'hw6': HW6Model,
    'linear': LinearModel
}
