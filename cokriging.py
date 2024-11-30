#!/usr/bin/python3
import argparse
import importlib
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
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
    print("\tH PRIMARY")
    print(H_primary.shape)
    print(H_primary)

    # SECONDARY - SECONDARY
    secondary_coords = known_sample_secondary[['x', 'y']].values
    m = len(secondary_coords)
    H_secondary = distance_matrix(secondary_coords, secondary_coords, p=2)
    print("\tH SECONDARY")
    print(H_secondary.shape)
    print(H_secondary)
    
    # CROSS (PRIMARY - SECONDARY)
    H_cross = distance_matrix(primary_coords, secondary_coords, p=2)
    print("\tH CROSS")
    print(H_cross.shape)
    print(H_cross)

    # PRIMARY - TARGET
    H_0_primary = distance_matrix(primary_coords, target_locations, p=2)
    print("\tH PRIMARY TARGET")
    print(H_0_primary.shape)
    print(H_0_primary)

    # SECONDARY - TARGET
    H_0_secondary = distance_matrix(secondary_coords, target_locations, p=2)
    print("\tH SECONDARY TARGET")
    print(H_0_secondary.shape)
    print(H_0_secondary)

    ############################################
    # Calculate covariances / semivariances
    ############################################

    # PRIMARY - PRIMARY
    C_primary = primary_model.fit_h(H_primary) 
    print("\tC PRIMARY")
    print(C_primary)

    # SECONDARY - SECONDARY
    C_secondary = secondary_model.fit_h(H_secondary) 
    print("\tC SECONDARY")
    print(C_secondary)

    # CROSS (PRIMARY - SECONDARY)
    C_cross = cross_model.fit_h(H_cross)
    print("\tC CROSS")
    print(C_cross)

    # PRIMARY - TARGET
    C_0_primary = primary_model.fit_h(H_0_primary)
    print("\tC PRIMARY TARGET")
    print(C_0_primary)

    # SECONDARY_TARGET
    # NB: use the CROSS model not secondary 
    # bc we want to estimate the primary value
    C_0_secondary = cross_model.fit_h(H_0_secondary)
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
    print("\tFULL C MATRIX")
    print(C)
    
    ############################################
    # Set up D matrix
    ############################################

    D = np.vstack((C_0_primary, C_0_secondary, 1, 0))
    print("\tFULL D MATRIX")
    print(D)

    ############################################
    # Solve for weights matrix
    ############################################

    W = np.matmul(np.linalg.inv(C), D)
    print("\tWEIGHTS")
    print(W)

    # astonishingly, this correct so far






    return 1

if __name__ == "__main__":
    main()

