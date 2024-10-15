#!/usr/bin/python3
import argparse
import duckdb
import math
import matplotlib.pyplot as plt
from typing import Union, List

import scripts.berea as berea
import scripts.models as models
import scripts.stats as stats


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

    parser.add_argument("--bin_width", type=int, 
        help="""
        If set, this forces the semivariogram to use the specified, fixed bin width.
        """)

    parser.add_argument("-bx", "--bin_max", type=float, action='append',
        help="""
        A list of floats or ints to use for raw semivariogram bin maxes.
        Use like: `-bx 10 -bx 20.0 -bx 30`
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

    ################################
    # get dataset
    ################################

    berea_select = """
    SELECT
    row_number() OVER () AS id,
    x_mm,
    y_mm,
    permeability_md,
    water_resisitivity_ohmm AS resistivity_ohmm
    FROM read_csv('data/berea_subset2.csv', normalize_names=True)
    ORDER BY x_mm, y_mm ASC
    """
    berea_df = duckdb.sql(berea_select).df()
    sample_count = berea_df['id'].count()

    ########################################
    # find pairwise lags and semivariances
    ########################################

    # join data to itself
    pair_join = """
    SELECT
        near.x_mm AS near_x_mm,
        near.y_mm AS near_y_mm,
        far.x_mm AS far_x_mm,
        far.y_mm AS far_y_mm,
        near.resistivity_ohmm AS near_resistivity,
        far.resistivity_ohmm AS far_resistivity,
        near.permeability_md AS near_permeability,
        far.permeability_md AS far_permeability
    FROM 
        berea_df AS near
    JOIN
        berea_df AS far
    ON near.id < far.id
    """

    pair_df = duckdb.sql(pair_join).df()

    # check that the number of pairs is as expected
    expected_pair_count = sample_count * (sample_count - 1) // 2
    pair_count = pair_df['near_x_mm'].count()
    print(f"Got {pair_count} pairs, expected {expected_pair_count}")

    # find lag distance for each pair
    pair_df['euclidean_distance'] = stats.euclidean_distance_2d(
                                        pair_df['near_x_mm'],
                                        pair_df['far_x_mm'],
                                        pair_df['near_y_mm'],
                                        pair_df['far_y_mm'])

    # find individual semivariance values for the pairs
    pair_df['semivariance'] = stats.raw_semivariance(
                                        pair_df['near_permeability'],
                                        pair_df['far_permeability'])

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
        ax.scatter(pair_df['euclidean_distance'], pair_df['semivariance'], s=0.2)
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
        ax.hist(pair_df['euclidean_distance'])
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

    if options.bin_width:  # use a fixed fixed bin size (HW5 Q2.ii)

        # use DuckDB query to calculate per-bin average lag and semivariance
        # TODO: add some sort of validation for bin width?

        binner = f"""
        WITH bins AS (
            SELECT
                euclidean_distance AS lag_distance,
                semivariance,
                {options.bin_width} AS bin_width,
                CEIL(euclidean_distance / {options.bin_width}) AS bin
            FROM pair_df
            ORDER BY euclidean_distance ASC
        )

        SELECT
            AVG(lag_distance) AS avg_lag_distance,
            AVG(semivariance) AS avg_semivariance,
            COUNT() AS n
        FROM bins
        GROUP BY bin
        ORDER BY 1
        """

    elif options.bin_max:  # use a custom set of bins (HW5 Q2.iii)
        
        # sort to make mapping function nicer
        options.bin_max.sort()
        print(options.bin_max)

        # calculate max lag distance in the pair dataset
        max_lag = pair_df['euclidean_distance'].max()

        # mapping function for binning
        def binner(x):

            # want the first bin that is greater than or equal to the distance
            for m in options.bin_max:
                if m >= x:
                    return m
            
            # the user didn't specify a bin large enough for the biggest lag
            # so just bin all data above their largest bin together
            return max_lag

        # map each pair to a bin max
        pair_df['bin'] = pair_df['euclidean_distance'].map(binner)
        print(pair_df.head(20))
        
        # use DuckDB query to calculate per-bin average lag and semivariance
        binner = f"""
        SELECT
            AVG(euclidean_distance) AS avg_lag_distance,
            AVG(semivariance) AS avg_semivariance,
            COUNT() AS n
        FROM pair_df
        GROUP BY bin
        ORDER BY 1
        """
        
    else:  # each bin has a fixed number of points (HW5 Q2.i) (default behavior)
        
        # calculate using square-route method shown in class
        n_bins = math.ceil(pair_count ** (1/2))

        # use DuckDB query to calculate per-bin average lag and semivariance
        binner = f"""
            WITH bins AS (
                SELECT
                    euclidean_distance AS lag_distance,
                    semivariance AS semivariance,
                    FLOOR(ROW_NUMBER() OVER (ORDER BY euclidean_distance ASC) / {n_bins}) * {n_bins} AS bin
                FROM 
                    pair_df
            )

            SELECT
                AVG(lag_distance) AS avg_lag_distance,
                AVG(semivariance) AS avg_semivariance,
                COUNT() AS n
            FROM
                bins
            GROUP BY bin
            ORDER BY 1
            """

    # turn query results into dataframe
    bins_df = duckdb.sql(binner).df()

    # check work
    print(bins_df.head(22))
    
    # TODO: checkpoint the data and implement loading

    #######################################################################
    # Plot option 2: show raw pairs with binned average lags, semivariances
    #######################################################################

    if options.plot == 'raw_semivariogram':
        # Plot the raw semivariance.
        # The x-axis being lag distances,
        # The y-axis being semivariance.
        # Raw points should be plotted in gray
        # Averages per bins should be plotted in red.

        # start the plot
        fig, ax = plt.subplots()

        # plot raw semivariance points
        ax.scatter(pair_df['euclidean_distance'], pair_df['semivariance'], color="lightgray", s=0.2)

        title_str = 'Raw semivariogram'
        
        if options.bin_width: 
            title_str += f': {options.bin_width} mm fixed bins'
        elif options.bin_max:
            title_str += ': bin divisions at \n' + ', '.join([str(int(i)) for i in options.bin_max]) + ' mm'
        else:
            title_str += f': equal points per bin ({n_bins} bins)'
        
        ax.set_title(title_str)
        ax.set_xlabel('Lag Distance (h)')
        ax.set_ylabel('Semivariance')

        # plot average semivariance points
        ax.scatter(bins_df['avg_lag_distance'], bins_df['avg_semivariance'], color='red', marker='x')

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
        fig.savefig('images/berea_raw_semivariogram.png')

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
            # TODO: this would be so annoying to get to if previous steps take a while

        # fit and plot model
        elif options.model in ('spherical', 'exponential', 'gaussian', 'isotonic'):
            model = models.fit_model(options.model, bins_df, 'avg_lag_distance', 'avg_semivariance', options.range)

            # start the plot
            fig, ax = plt.subplots()
            ax.set_title(f'Semivariogram: {options.model} model')
            ax.set_xlabel('Lag Distance (h)')
            ax.set_ylabel('Semivariance')

            # plot the average semivariance points
            ax.scatter(bins_df['avg_lag_distance'], bins_df['avg_semivariance'], color='red', marker='x')

            # plot the modeled semivariance
            ax.plot(model['fit_x'], model['fit_y'], color='black')

            # plot the range
            ax.axvline(x=options.range,
                    # ymax=spherical_model['sill'], # this is not doing what I want
                    label=f'Range: {options.range}',
                    color='black',
                    linestyle='--')

            ax.axhline(y=model['sill'],
                label=f"Sill: {model['sill']}",
                color='black',
                linestyle=':')

            # add in the Ns
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            textstr = '\n'.join((
                f"sill = {model['sill']:.1f}",
                f"range = {options.range}"
                ))

            # place a text box in lower right in axes coords
            ax.text(0.75, 0.25, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.show()
            fig.savefig('images/berea_semivariogram.png')

        else:
            print("Model not implemented yet, sorry.")
            return 1

    # TODO: make plots into classes
    # TODO: plots overwrite each other, fix this. 
    # TODO: make models into classes

    return 0


if __name__ == "__main__":
    main()
