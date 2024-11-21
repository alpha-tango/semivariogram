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

    # PRIMARY
    primary_coords = known_sample_primary[['x', 'y']].values
    H_primary = distance_matrix(primary_coords, primary_coords, p=2)
    print(H_primary.shape)

    # SECONDARY
    secondary_coords = known_sample_secondary[['x', 'y']].values
    H_secondary = distance_matrix(secondary_coords, secondary_coords, p=2)
    print(H_secondary.shape)
    
    # CROSS
    H_cross = distance_matrix(primary_coords, secondary_coords, p=2)
    print(H_cross.shape)

    ############################################
    # Calculate covariances / semivariances
    ############################################

    



    return 1

if __name__ == "__main__":
    main()

