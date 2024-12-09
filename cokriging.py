#!/usr/bin/python3
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

# import scripts.models as models

def main():
    ###############################
    # Parse command line arguments
    ###############################

    parser = argparse.ArgumentParser(
                prog='Co-Kriging',
                description='Run Co-Kriging')

    parser.add_argument(
        'config_file',
        type=str,
        help="""
        Specify the name of the config file to use. Ex: if you have 'config/myfile.py',
        this argument should be `myfile`
        """)

    parser.add_argument("--test", action=argparse.BooleanOptionalAction,
        help="""
        Test with a small subset of data.
        """)

    try:
        options = parser.parse_args()
    except:
        parser.print_help()
        return 1

    ###############################################
    # Get the config file to use via command line
    ###############################################

    # This sets a bunch of the important settings
    # Like how to get the data and the model to use
    # config/textbook.py is an example of what
    # functions and variables are set.

    workflow_config = importlib.import_module(f'config.{options.config_file}')

    ##############################################
    # Get input data
    ##############################################

    # Get the primary data
    known_sample_primary = workflow_config.primary_data()
    print(f"Found {len(known_sample_primary)} primary data points.")

    # Get the secondary data
    known_sample_secondary = workflow_config.secondary_data()
    print(f"Found {len(known_sample_secondary)} secondary data points.")

    # Construct meshgrids of target x, y locations for which we
    # want to estimate values for. 
    # This format will be useful later when we plot the output.
    target_X, target_Y = np.meshgrid(workflow_config.TARGET_X_COORDS, workflow_config.TARGET_Y_COORDS)

    # Turn the meshgrids into a list of [x,y] coordinates
    # For use in calculation
    target_locations = np.vstack([target_X.ravel(), target_Y.ravel()]).T
    print(f"Found {len(target_locations)} target locations to estimate.")

    ################################################
    # Get models
    ################################################

    # all the heavy lifting is in the config file
    primary_model = workflow_config.PRIMARY_MODEL
    secondary_model = workflow_config.SECONDARY_MODEL
    cross_model = workflow_config.CROSS_MODEL

    #############################################
    # Calculate distances
    #############################################

    # PRIMARY - PRIMARY
    primary_coords = known_sample_primary[['x', 'y']].values
    n = len(primary_coords)
    H_primary = distance_matrix(primary_coords, primary_coords, p=2)
    if options.test:
        print("\tH PRIMARY")
        print(H_primary.shape)
        print(H_primary)

    # SECONDARY - SECONDARY
    secondary_coords = known_sample_secondary[['x', 'y']].values
    m = len(secondary_coords)
    H_secondary = distance_matrix(secondary_coords, secondary_coords, p=2)
    if options.test:
        print("\tH SECONDARY")
        print(H_secondary.shape)
        print(H_secondary)
    
    # CROSS (PRIMARY - SECONDARY)
    H_cross = distance_matrix(primary_coords, secondary_coords, p=2)
    if options.test:
        print("\tH CROSS")
        print(H_cross.shape)
        print(H_cross)

    # PRIMARY - TARGET
    H_0_primary = distance_matrix(primary_coords, target_locations, p=2)
    if options.test:
        print("\tH PRIMARY TARGET")
        print(H_0_primary.shape)
        print(H_0_primary)

    # SECONDARY - TARGET
    H_0_secondary = distance_matrix(secondary_coords, target_locations, p=2)
    if options.test:
        print("\tH SECONDARY TARGET")
        print(H_0_secondary.shape)
        print(H_0_secondary)

    ############################################
    # Calculate covariances / semivariances
    ############################################

    # PRIMARY - PRIMARY
    C_primary = primary_model.fit_h(H_primary) 
    if options.test:
        print("\tC PRIMARY")
        print(C_primary)

    # SECONDARY - SECONDARY
    C_secondary = secondary_model.fit_h(H_secondary)
    if options.test: 
        print("\tC SECONDARY")
        print(C_secondary)

    # CROSS (PRIMARY - SECONDARY)
    C_cross = cross_model.fit_h(H_cross)
    if options.test:
        print("\tC CROSS")
        print(C_cross)

    # PRIMARY - TARGET
    C_0_primary = primary_model.fit_h(H_0_primary)
    if options.test:
        print("\tC PRIMARY TARGET")
        print(C_0_primary)

    # SECONDARY_TARGET
    # NB: use the CROSS model not secondary 
    # bc we want to estimate the primary value
    C_0_secondary = cross_model.fit_h(H_0_secondary)
    if options.test:
        print("\tC SECONDARY TARGET")  
        print(C_0_secondary)

    ############################################
    # Set up C matrix
    ############################################

    # first "row": primary-primary matrix, cross matrix, lagranges
    q = np.hstack((C_primary, C_cross, np.ones((n,1)), np.zeros((n,1))))

    # second "row": cross matrix transpose, secondary matrix, lagranges
    b = np.hstack((C_cross.T, C_secondary, np.zeros((m,1)), np.ones((m,1))))

    # third "row": lagrange 1
    x = np.hstack((np.ones((1,n)), np.zeros((1,m)), np.zeros((1,2))))

    # fourth "row": lagrange 2
    r = np.hstack((np.zeros((1,n)), np.ones((1,m)), np.zeros((1,2))))

    C = np.vstack((q,b,x,r))

    if options.test:
        print("\tFULL C MATRIX")
        print(C)
    
    ############################################
    # Set up D matrix
    ############################################

    z = len(target_locations)
    D = np.vstack((C_0_primary, C_0_secondary, np.ones((1,z)), np.zeros((1,z))))

    if options.test:
        print("\tFULL D MATRIX")
        print(D)

    ############################################
    # Solve for weights matrix
    ############################################

    W = np.matmul(np.linalg.inv(C), D)

    if options.test:
        print("\tWEIGHTS")
        print(W)

    # astonishingly, this correct so far

    ############################################
    # Produce estimates
    ############################################

    # get known sample point values and stack to matrix
    primary_actual = known_sample_primary[[workflow_config.PRIMARY_VAR_NAME]].values
    secondary_actual = known_sample_secondary[[workflow_config.SECONDARY_VAR_NAME]].values
    V = np.vstack((primary_actual, secondary_actual))

    if options.test:
        print("\tACTUAL KNOWN")
        print(V)

    # produce estimate by multiplying with weights
    # (Lagrange parameters removed)
    V_est = np.matmul(np.rot90(V), W[:-2])
    if options.test:
        print("\tESTIMATES")
        print(V_est)

    ############################################
    # Produce error variance
    #############################################

    # combined_error = np.matmul(np.rot90(D[:-2]), W[:-2])
    # lagrange_1 = W[-2]
    # lagrange_2 = W[-1]
    # sill = 500000  # TODO  # variance / stddev^2 1/n sum((xi - m)^2)

    # print("\tERROR VARIANCE")
    # print(combined_error)
    # print(lagrange_1)
    # print(lagrange_2)
    # print(sill)

    # error_variance = sill - (lagrange_1 + lagrange_2) - combined_error
    # print(error_variance)

    #############################################
    # Plot data
    #############################################

    # reshape estimates V_est into meshgrid format

    n = target_X.shape[0]
    m = target_X.shape[1]

    if options.test:
        print(target_Y)
        print(target_Y.ravel())
        print(target_Y.ravel().reshape((n,m)))

    E = V_est[0].reshape((n,m))
    
    # make a colormap plot of the estimates
    fig, ax = plt.subplots()
    a = ax.pcolormesh(
            target_X,
            target_Y,
            E,
            vmin=E.min(),
            vmax=E.max())

    plt.colorbar(a)  # show the color bar to the right
    ax.scatter(known_sample_primary['x'], known_sample_primary['y'],color='red', marker='x', label="Primary Sample Point")  # plot known points on top
    ax.scatter(known_sample_secondary['x'], known_sample_secondary['y'],color='black', marker='+', label="Secondary Sample Point")
    plt.title(f"Estimated values ({workflow_config.UNITS}) for {workflow_config.PRIMARY_VAR_NAME}")
    plt.savefig(f'images/{workflow_config.IM_TAG}_cokriging.png')

    # make a colormap plot of the error variances
    # fig, ax = plt.subplots()
    # a = ax.pcolormesh(x_df, y_df, error_df, vmin=error_variances.min(), vmax=error_variances.max())
    # plt.colorbar(a)  # show color bar
    # ax.scatter(raw_df['x'], raw_df['y'],color='red', label="Sampled Point")  # plot known points on top

    # plt.title(f"Error variance for {workflow_config.PRIMARY_VAR_NAME}")
    # plt.savefig(f'images/{workflow_config.IM_TAG}_cokriged_error.png')

    workflow_config.co_kriging_custom_plots(known_sample_primary, known_sample_secondary, target_X, target_Y, E)


    return 0

if __name__ == "__main__":
    main()

