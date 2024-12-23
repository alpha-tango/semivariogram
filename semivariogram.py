#!/usr/bin/python3
import argparse
import duckdb
import importlib
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from typing import Union, List

import scripts.binner as binners
import scripts.models as models
import scripts.plots as plots
import scripts.stats as stats


def main():
    ###################################
    # parse command-line arguments
    ###################################

    parser = argparse.ArgumentParser(
                    prog='Semivariogram',
                    description='Build a semivariogram',
                    epilog="""
                    If neither bin_width nor custom_bins is set, the default behavior
                    is to use equal points per bin.
                    """)

    parser.add_argument(
        'config_file',
        type=str,
        help="""
        Specify the name of the config file to use. Ex: if you have 'config/myfile.py',
        this argument should be `myfile`
        """)
    
    parser.add_argument(
        'plot',
        choices=['raw_histogram', 'raw_semivariogram', 'semivariogram'],
        help="""
        Specify which plot you want to build.
        Raw histogram displays a histogram of the raw data.
        Raw semivariogram displays semivariance by distance, with multiple binning methods supported.
        Semivariogram displays a semivariogram using the selected model, or all models.
        """)

    parser.add_argument("--cross", action=argparse.BooleanOptionalAction,
        help="""
        Make a cross semivariogram using the primary and secondary variables specified in the config file.
        """)

    parser.add_argument("--bin_width", type=float, 
        help="""
        If set, this forces the semivariogram to use the specified, fixed bin width.
        """)

    parser.add_argument("--points_per_bin", type=int,
        help="""
        If used with `--bin_method=equal_points`, this forces the semivariogram to use the
        specified number of points per bin.
        """)
    # TODO: add option to run all models


    try:
        options = parser.parse_args()
    except:
        parser.print_help()
        return 1

    workflow_config = importlib.import_module(f'config.{options.config_file}')
    print(workflow_config)

    if workflow_config.BIN_METHOD == 'fixed_width':
        assert options.bin_width is not None
    elif workflow_config.BIN_METHOD == 'config':
        assert workflow_config.BIN_MAXES is not None

    ################################
    # get dataset
    ################################

    # get the raw data from the config file
    raw_df = workflow_config.raw_data()
    sample_count = raw_df['id'].count()
    print(f"{sample_count} samples found in raw data")

    ########################################
    # find pairwise lags and semivariances
    ########################################

    # are we using secondary variables?
    if options.cross:
        raw_df['secondary']
        secondary_str = """
        near.secondary AS near_secondary,
        far.secondary AS far_secondary,
        """
    else:
        secondary_str = ""

    # join data to itself
    pair_join = f"""
    SELECT
        near.x AS near_x,
        near.y AS near_y,
        far.x AS far_x,
        far.y AS far_y,
        {secondary_str}
        near.primary AS near_primary,
        far.primary AS far_primary
    FROM 
        raw_df AS near
    JOIN
        raw_df AS far
    ON near.id < far.id
    """

    pair_df = duckdb.sql(pair_join).df()

    # check that the number of pairs is as expected
    expected_pair_count = sample_count * (sample_count - 1) // 2
    pair_count = pair_df['near_x'].count()
    print(f"Got {pair_count} pairs, expected {expected_pair_count}")

    # find lag distance for each pair

    try:
        # if the user has set a distance method, use their preferred method
        if workflow_config.DISTANCE_METHOD == 'euclidean':
            pair_df['h'] = stats.euclidean_distance_2d(
                                    pair_df['near_x'],
                                    pair_df['far_x'],
                                    pair_df['near_y'],
                                    pair_df['far_y'])
        elif workflow_config.DISTANCE_METHOD == 'geographic':
            pair_df['h'] = stats.geographic_distance(
                                    pair_df['near_x'],
                                    pair_df['far_x'],
                                    pair_df['near_y'],
                                    pair_df['far_y'])
    except AttributeError:
        # if the user hasn't set a distance method, default to using
        # Euclidean distance
        pair_df['h'] = stats.euclidean_distance_2d(
                                            pair_df['near_x'],
                                            pair_df['far_x'],
                                            pair_df['near_y'],
                                            pair_df['far_y'])

    # find individual semivariance values for the pairs
    if options.cross:
            pair_df['semivariance'] = stats.raw_semivariance(
                                        pair_df['near_primary'],
                                        pair_df['far_secondary'])
    else:
        pair_df['semivariance'] = stats.raw_semivariance(
                                            pair_df['near_primary'],
                                            pair_df['far_primary'])

    # sort df by distance
    pair_df.sort_values(by=['h'], axis=0, inplace=True)


    ########################################################################
    # Plot option 1: show histogram of raw pairs (helps with binning choice)
    ########################################################################

    if options.plot == 'raw_histogram':

        if options.cross:
            print("Not yet implemented.")
            return 1
        
        #########################################
        # Custom plots
        ##########################################

        workflow_config.custom_raw_plots(raw_df)

        ##########################################
        # Boilerplate plots
        ##########################################

        # Plot x,y of sample locations
        plot = plots.SampleLocations(imname=workflow_config.IM_TAG, raw_df=raw_df)
        plot.show_and_save()

        # Plot distances vs semivariances
        plot = plots.RawPairData(imname=workflow_config.IM_TAG, pair_df=pair_df)
        plot.show_and_save()

        # Plot a histogram of distances vs. number of points at that distance
        plot = plots.RawHistogram(imname=workflow_config.IM_TAG, pair_df=pair_df)
        plot.show_and_save()

        # Plot smoothed raw pairs
        plot = plots.IsotonicSmooth(imname=workflow_config.IM_TAG, pair_df=pair_df)
        plot.show_and_save()

        return 0  # don't do any of the following stuff, end work here

    ############################################################
    # Calculate the bins using the various methods
    ############################################################

    # TODO: add a warning for bins with n < 10.
    # TODO: option - if n < 10 in a bin, fit on raw data or isotonic model instead.
    # TODO: add option to fit fully on raw data.

    # First: Bin the data according to the user's choice

    if workflow_config.BIN_METHOD == 'fixed_width':  # use a fixed fixed bin size (HW5 Q2.ii)
        binner = binners.EqualWidthBinner(bin_width=options.bin_width)
        bins_df = binner.bins(pair_df)

    elif workflow_config.BIN_METHOD == 'config':  # use a custom set of bins (HW5 Q2.iii)
        binner = binners.CustomBinner(workflow_config.BIN_MAXES)
        bins_df = binner.bins(pair_df=pair_df)
        
    elif workflow_config.BIN_METHOD == 'equal_points':  # each bin has a fixed number of points (HW5 Q2.i)
        binner = binners.EqualPointBinner(points_per_bin=options.points_per_bin)
        bins_df = binner.bins(pair_df=pair_df)

    elif workflow_config.BIN_METHOD == 'raw':
        binner = binners.RawBinner()
        bins_df = binner.bins(pair_df=pair_df)

    # check work
    print(bins_df.head(22))
    
    # TODO: checkpoint the data and implement loading

    #######################################################################
    # Plot option 2: show raw pairs with binned average lags, semivariances
    #######################################################################

    if options.plot == 'raw_semivariogram':
        # Plot the raw semivariance
        # See details in the class description

        n_bins = len(set(bins_df['bin']))

        if options.cross:
            plot = plots.RawCrossSemivariogram(imname=workflow_config.IM_TAG,
                                pair_df=pair_df,
                                avg_df=bins_df,
                                n_bins=n_bins,
                                h_units=workflow_config.H_UNITS)
        else:
            plot = plots.RawSemivariogram(imname=workflow_config.IM_TAG,
                                            pair_df=pair_df,
                                            avg_df=bins_df,
                                            n_bins=n_bins,
                                            h_units=workflow_config.H_UNITS)

        plot.show_and_save()

        return 0


    ###########################################################
    # Plot option 3: semivariogram with a fitted model
    ###########################################################

    # check that the user has given a range to fit the model on
    if not workflow_config.RANGE:
        print("Must specify range.")
        return 1

    # model the binned data
    if not workflow_config.MODEL:
        print("You must specify a model")
        return 1

    # fit the model to the data
    if options.cross:
        model = workflow_config.CROSS_MODEL
    else:
        model = workflow_config.MODEL

    # make semivariogram
    if options.plot == 'semivariogram':

        plot_model = model.plottable(bins_df['h'])
        if options.cross:
            plot = plots.CrossSemivariogram(
                        a=workflow_config.CROSS_RANGE,
                        omega=workflow_config.CROSS_SILL,
                        nugget=workflow_config.CROSS_NUGGET,
                        model_name=model.name,
                        model_lag=plot_model['h'],
                        model_semivariance=plot_model['semivariance'],
                        raw_df=bins_df,
                        display_primary_var=workflow_config.PRIMARY_DISPLAY_NAME,
                        display_secondary_var=workflow_config.SECONDARY_DISPLAY_NAME,
                        imname=workflow_config.IM_TAG,
                        h_units=workflow_config.H_UNITS
                        )
        else:
            plot = plots.Semivariogram(
                        a=workflow_config.RANGE,
                        omega=workflow_config.SILL,
                        nugget=workflow_config.NUGGET,
                        model_name=model.name,
                        model_lag=plot_model['h'],
                        model_semivariance=plot_model['semivariance'],
                        raw_df=bins_df,
                        display_var_name=workflow_config.PRIMARY_DISPLAY_NAME,
                        imname=workflow_config.IM_TAG,
                        h_units=workflow_config.H_UNITS
                        )

        plot.show_and_save()

        # TODO: plots overwrite each other, fix this.
        return 0


    return 0


if __name__ == "__main__":
    main()
