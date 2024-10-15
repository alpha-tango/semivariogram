"""
TODO: make all these things class based to reduce duplicate code
"""

import duckdb
import numpy as np
from sklearn.isotonic import isotonic_regression

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

    ##################################
    # get needed info for fitting models
    ##################################

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