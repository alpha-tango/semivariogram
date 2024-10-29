#!/usr/bin/python3
import argparse
import duckdb
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List

import scripts.binner as binners
import scripts.models as models
import scripts.plots as plots
import scripts.stats as stats

import config.berea as workflow_config


def main():
    ###################################
    # parse command-line arguments
    ###################################

    parser = argparse.ArgumentParser(
                    prog='Semivariogram - Berea',
                    description='Build a semivariogram using the Berea subset',
                    epilog="""
                    If neither bin_width nor custom_bins is set, the default behavior
                    is to use equal points per bin.
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

    parser.add_argument('--bin_method', choices=['config', 'fixed_width', 'equal_points', 'raw'],
        help="""
        Choose a method to bin the data before modeling semivariance.
        `config`: in the appropriate config file, set a list of bin maxes.
        `fixed_width`: bins are of fixed width. Must be used with `--bin_width` argument.
        `equal_points`: bins have an equal number of points. Defaults to square 
            root of the number of data points, or can be used with `--points_per_bin` argument.
        `raw`: does not bin the data at all.
        """
        )

    parser.add_argument("--bin_width", type=int, 
        help="""
        If set, this forces the semivariogram to use the specified, fixed bin width.
        """)

    parser.add_argument("--points_per_bin", type=int,
        help="""
        If used with `--bin_method=equal_points`, this forces the semivariogram to use the
        specified number of points per bin.
        """)

    parser.add_argument("--range", type=float,
        help="""
        If used in 'semivariogram' setting, this sets the range the model will
        be fitted on.
        """)

    parser.add_argument("--model",
        choices=['spherical', 'exponential', 'gaussian', 'isotonic'],
        help="""
        Select a model to fit the semivariogram on, when used in 'semivariogram' mode.
        Choose a single model or 'all'. Currently must be used with "--range" setting.
        """)
    # TODO: add option to run all models

    try:
        options = parser.parse_args()
    except:
        parser.print_help()
        return 1

    if options.bin_method == 'fixed_width':
        assert options.bin_width is not None
    elif options.bin_method == 'equal_points':
        assert options.points_per_bin is not None
    elif options.bin_method == 'config':
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
    if raw_df['secondary'].empty:
        secondary_str = ""
    else:
        secondary_str = """
        near.secondary AS near_secondary,
        far.secondary AS far_secondary,
        """

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
    pair_df['h'] = stats.euclidean_distance_2d(
                                        pair_df['near_x'],
                                        pair_df['far_x'],
                                        pair_df['near_y'],
                                        pair_df['far_y'])

    # find individual semivariance values for the pairs
    pair_df['semivariance'] = stats.raw_semivariance(
                                        pair_df['near_primary'],
                                        pair_df['far_primary'])

    # TODO: move a lot of the above junk into a config file per dataset


    ########################################################################
    # Plot option 1: show histogram of raw pairs (helps with binning choice)
    ########################################################################

    # TODO: also plot raw data and trendlines

    if options.plot == 'raw_histogram':
        
        # Plot x,y of sample locations
        fig, ax = plt.subplots()
        ax.scatter(berea_df['x_mm'], berea_df['y_mm'])
        ax.set_title(f"Location of {sample_count} samples")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.show()
        fig.savefig('images/berea_sample_locations.png')

        # Plot location vs permeability
        fig, ax = plt.subplots()
        ax.scatter(berea_df['x_mm'], berea_df['permeability_md'])
        ax.set_title("Permeability by x location")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Permeability (md)")
        plt.show()
        fig.savefig('images/berea_permeability_by_x.png')

        fig, ax = plt.subplots()
        ax.scatter(berea_df['y_mm'], berea_df['permeability_md'])
        ax.set_title("Permeability by y location")
        ax.set_xlabel("Y (mm)")
        ax.set_ylabel("Permeability (md)")
        plt.show()
        fig.savefig('images/berea_permeability_by_y.png')

        # Plot distances vs semivariances
        fig, ax = plt.subplots()
        ax.scatter(pair_df['h'], pair_df['semivariance'], s=0.2)
        ax.set_title("Semivariance by lag distance (omnidirectional)")
        ax.set_xlabel("lag distance (mm)")
        ax.set_ylabel("semivariance")
        
        # add in the Ns
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((
            f"sample count = {sample_count}",
            f"pair count = {pair_count}"
            ))

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.show()
        fig.savefig('images/berea_raw_pair_data.png')

        # Plot a histogram of distances vs. number of points at that distance
        fig, ax = plt.subplots()
        ax.hist(pair_df['h'])
        ax.set_title("Count of sample pairs by lag distance (omnidirectional)")
        ax.set_xlabel("lag distance (mm)")
        ax.set_ylabel("count of pairs")
        plt.show()
        fig.savefig('images/berea_raw_histogram.png')

        return 0  # don't do any of the following stuff, end work here

    ############################################################
    # Calculate the bins using the various methods
    ############################################################

    # TODO: add a warning for bins with n < 10.
    # TODO: option - if n < 10 in a bin, fit on raw data or isotonic model instead.
    # TODO: add option to fit fully on raw data.

    # First: Bin the data according to the user's choice

    if options.bin_method == 'fixed_width':  # use a fixed fixed bin size (HW5 Q2.ii)

        binner = binners.EqualWidthBinner(bin_width=options.bin_width)
        bins_df = binner.bins(pair_df)

    elif options.bin_method == 'config':  # use a custom set of bins (HW5 Q2.iii)

        binner = binners.CustomBinner(workflow_config.BIN_MAXES)
        bins_df = binner.bins(pair_df=pair_df)
        
    elif options.bin_method == 'equal_points':  # each bin has a fixed number of points (HW5 Q2.i)
        binner = binners.EqualPointBinner(points_per_bin=options.points_per_bin)
        bins_df = binner.bins(pair_df=pair_df)

    else:  # options.bin_method == 'raw' based on the argument options
        print('not supported')
        return 1

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

        plot = plots.RawSemivariogram(imname='berea',
                                        pair_df=pair_df,
                                        avg_df=bins_df,
                                        n_bins=n_bins)
        plot.show_and_save()


    ###########################################################
    # Plot option 3: semivariogram with a fitted model
    ###########################################################

    elif options.plot == 'semivariogram':

        # check that the user has given a range to fit the model on
        if not options.range:
            print("Must specify range.")
            return 1

        # model the binned data
        if not options.model:
            print("You must specify a model")
            return 1

        # fit and plot model
        elif options.model in ('spherical', 'exponential', 'gaussian', 'isotonic'):

            model_class = models.get_model(options.model)
            model = model_class(data_df=to_model_df, fit_range=options.range)
            print(model.name)
            plot_model = model.plottable()

            plot = plots.Semivariogram(
                        a=options.range,
                        omega=model.sill,
                        model_name=model.name,
                        model_lag=plot_model['h'],
                        model_semivariance=plot_model['semivariance'],
                        raw_df=bins_df,
                        imname='berea'
                        )

            plot.show_and_save()

        else:
            print("Model not implemented yet, sorry.")
            return 1

    # TODO: plots overwrite each other, fix this. 

    return 0


if __name__ == "__main__":
    main()
