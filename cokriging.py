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
        Test Kriging with a small subset of data.
        """)

    parser.add_argument("--validation", action=argparse.BooleanOptionalAction,
        help="""
        Validate the results of Kriging with a holdout set of known sample points
        that you define in the config file.
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

    if options.validation:
        validation_data = workflow_config.validation_data()
        target_x = validation_data['x']
        target_y = validation_data['y']
        target_locations = validation_data[['x', 'y']]
        print(f"Found {len(target_locations)} validation locations to estimate.")
    else:
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

    ####### Correct negative weights. ##########

    # take weights except for Lagrange multipliers 
    Q = W[:-2] 
    
    # set negative values to zero
    Q[Q < 0] = 0

    # the weights no longer sum to 1 at each target location
    # so we now have to renormalize for that

    # sum each set of weights to see what they add up to
    # then put over 1 to scale each element of the weights matrix
    x = (1 / Q.sum(axis=0))

    # iterate through each row of the weights
    rows = len(Q)
    for i in range(rows):

        # set each row equal to the elementwise product of the row and x
        Q[i] = np.multiply(Q[i], x)

    # pull out the lagrange multipliers
    lagrange_1 = W[-2]
    lagrange_2 = W[-1]

    # restack the normalized weights with the lagrange multipliers
    W = np.vstack((Q, lagrange_1, lagrange_2))


    ############################################
    # Produce estimates
    ############################################

    # get known sample point values and stack to matrix
    # these are the untransformed points
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

    # via Cailin: std= mu1+mu2+(sum(wi*ui)+sum(wj*vj))

    # Var{R} = Cov{V(io)V(io)} + mu1 + mu2 - sum(aiCov{VoVj} - sum(bjCov{VoUj}))
    # Var{R} = (Cov of primary variable = sill) + mu1 + mu2 - (primary error) - (secondary/cross error)

    # Hadamard multiplication multiply each weight by target covariance
    # Then sum vertically
    # This excludes the Lagrange multipliers
    # This sums together primary error and secondary / cross error
    combined_error = np.multiply(W[:-2], D[:-2]).sum(axis=0)  # vector of length j for j target locations

    # pull the first lagrange multiplier
    lagrange_1 = W[-2]  # vector of length j for j target locations

    # pull the second lagrange multiplier
    lagrange_2 = W[-1]  # vector of length j for j target locations

    # make of vector representing Cov. of primary variable = sill = Cov{V(io)V(io)}
    sill = np.ones(len(target_locations)) * workflow_config.primary_sill

    # sum the weight and target components for each target location
    # then subtract from the sill and lagrange multipliers

    error_variances = sill + lagrange_1 + lagrange_2 - combined_error

    if options.test:
        print("\tERROR VARIANCES")
        print(error_variances)

    ## Not quite working
    ## Doing a simple inverse transform doesn't preserve unbiasedness

    # de-transform error variances
    # error_variances = np.exp(error_variances)

    # if options.test:
    #     print("\tERROR VARIANCES")
    #     print(error_variances)

    #############################################
    # Plot data
    #############################################

    if not options.validation:
        # reshape estimates V_est into meshgrid format

        n = target_X.shape[0]
        m = target_X.shape[1]

        if options.test:
            print(target_Y)
            print(target_Y.ravel())
            print(target_Y.ravel().reshape((n,m)))

        E = V_est[0].reshape((n,m))
        R = error_variances.reshape((n,m))
        
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
        fig, ax = plt.subplots()
        a = ax.pcolormesh(target_X,
                          target_Y,
                          R,
                          vmin=R.min(),
                          vmax=R.max())
        plt.colorbar(a)  # show color bar
        ax.scatter(known_sample_primary['x'], known_sample_primary['y'],color='red', marker='x', label="Primary Sample Point")  # plot known points on top
        ax.scatter(known_sample_secondary['x'], known_sample_secondary['y'],color='black', marker='+', label="Secondary Sample Point")

        plt.title(f"Error variance for {workflow_config.PRIMARY_VAR_NAME}")
        plt.savefig(f'images/{workflow_config.IM_TAG}_cokriged_error.png')

        workflow_config.co_kriging_custom_plots(known_sample_primary, known_sample_secondary, target_X, target_Y, E, R)

    else:  # we are doing validation
        validation_data['estimate'] = V_est[0]
        validation_data['difference'] = validation_data['estimate'] - validation_data[workflow_config.PRIMARY_VAR_NAME]
        print(validation_data)
        print(np.mean(validation_data['difference']))
        print(np.std(validation_data['difference']))
        print(max(validation_data['difference']))
        workflow_config.co_kriging_validation_plots(known_sample_primary, known_sample_secondary, validation_data)


    return 0

if __name__ == "__main__":
    main()

