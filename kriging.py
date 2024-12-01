#!/usr/bin/python3
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import seaborn as sns

import scripts.models as models


def main():
    ###################################
    # parse command-line arguments
    ###################################

    parser = argparse.ArgumentParser(
                    prog='Kriging',
                    description='Krig')

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

    workflow_config = importlib.import_module(f'config.{options.config_file}')

    #############################################################
    # Kriging
    #############################################################

    # get raw dataset of known sample points
    raw_df = workflow_config.raw_data()

    if options.test:
        print(raw_df)
        return 1

    # get the model
    # fit the model to the data
    model = workflow_config.MODEL

    # get coordinates of the sample points to calculate distances
    sample_coords = raw_df[['x', 'y']].values

    try:
        h_matrix = workflow_config.distance_matrix(sample_coords, sample_coords)
    except AttributeError:
        # create distance matrix using scipy function
        # it uses minkowski distance, p=2 is same as Euclidean distance
        h_matrix = distance_matrix(sample_coords, sample_coords, p=2)  # checked manually

    # calculate semivariance from model and distance matrix
    semivariance_knownpts_matrix = model.fit_h(h_matrix)  # checked manually

    # add the ones and zero as described in class (due to Lagrange Param)
    b = np.ones((len(sample_coords),1)) * -1
    q = np.ones((1, len(sample_coords) + 1))
    C_semivariance_matrix = np.hstack((semivariance_knownpts_matrix,b))
    C_semivariance_matrix = np.vstack((C_semivariance_matrix,q))
    C_semivariance_matrix[len(sample_coords), len(sample_coords)] = 0

    if options.test:
        print("C - semivariance matrix")
        print(C_semivariance_matrix)

    # construct a target grid
    # get min and max coords from sample data so we know where to center our grid
    min_x = raw_df['x'].min()
    min_y = raw_df['y'].min()
    max_x = raw_df['x'].max()
    max_y = raw_df['y'].max()

    # lets add a tolerance around the edges
    x_tol = (max_x - min_x) * .1
    y_tol = (max_y - min_y) * .1

    # now make a n_pts x n_pts grid around the sample points
    n_pts = 100
    xs = np.linspace(min_x - x_tol, max_x + x_tol, n_pts)
    ys = np.linspace(min_y - y_tol, max_y + y_tol, n_pts)
    target_coords = [[i,j] for i in xs for j in ys]

    # calculate the distances between the sample points and the target point
    if options.test:
        target_coords = workflow_config.TEST_COORDS
        print(target_coords)


    try:
        target_vector = workflow_config.distance_matrix(sample_coords, target_coords)
    except AttributeError:
        target_vector = distance_matrix(sample_coords, target_coords)

    if options.test:
        print("Vector Target Distance")
        print(target_vector)
    D_semivariance_target_matrix = model.fit_h(target_vector)

    if options.test:
        print("D matrix shape (pre adding Lagrange)", D_semivariance_target_matrix.shape)

    # Add one on to end of vector due to Lagrange Param
    q = np.ones((1, len(target_coords)))
    D_semivariance_target_matrix = np.vstack((D_semivariance_target_matrix,q))

    if options.test:
        print("D matrix shape (with Lagrange)", D_semivariance_target_matrix.shape)
        print(D_semivariance_target_matrix)

    # do the matrix operations to get the w matrix
    C_inv = np.linalg.inv(C_semivariance_matrix)
    W_weights_matrix = np.matmul(C_inv, D_semivariance_target_matrix)

    if options.test:
        print("W matrix shape", W_weights_matrix.shape)
        print(W_weights_matrix)
        print(W_weights_matrix[:-1].sum())  # check sum to 1 (except Lagrange param) -- YES!

    # get the values of the known samples points
    actual_values_vector = raw_df[['primary']].values

    # add in a zero to deal with the Lagrange param
    q = np.zeros((1,1))
    actual_values_vector = np.vstack((actual_values_vector, q))
    actual_values_col = np.rot90(actual_values_vector)

    if options.test:
        print("actual values shape", actual_values_col.shape)
        print(actual_values_col)

    # use matrix multiplication to calculate estimated value
    estimates = np.matmul(actual_values_col, W_weights_matrix)
    try: 
        estimates = workflow_config.transform(estimates)
        print("De-transformed estimates")
    except AttributeError:
        print("No transform found")
        pass

    if options.test:
        print("estimate matrix", estimates.shape)
        print(estimates)

    #######################################
    # calculate error variance
    #######################################

    # # DONNA'S METHOD ---------------------
    # # GIVES SAME RESULT
    # # Remove padding from the C matrix and W matrix
    # W_unpad = W_weights_matrix[:-1]
    # C_unpad = semivariance_knownpts_matrix  # we've already calculated it above
    # mu = W_weights_matrix[-1]
    # # Re-estimate the D matrix using D = WC - mu
    # D_update = np.matmul(C_unpad, W_unpad) - mu
    # errors = np.multiply(W_unpad, D_update)
    # error_variances = errors.sum(axis=0) + mu
    # # ------------------------------------

    # multiply using element-wise / Hadamard multiplication
    # for i known sample points and j targets
    # D matrix and W matrix are both (i + 1) rows by j columns
    # element-wise gives one matrix of (i + 1) rows by j columns
    # each element is weight i,j multipled by semivariance i,j
    
    errors = np.multiply(W_weights_matrix, D_semivariance_target_matrix)
    # now sum down the columns
    # so we are summing over weight/semivariance product over the n sample points
    # and get get a vector of length j, each element is error variance
    # at target point j
    error_variances = errors.sum(axis=0)

    if options.test:
        print("error variance matrix", error_variances.shape)
        print(error_variances)

    if options.test:
        # Don't plot in test mode, just end here
        return 0

    # turn all the data into a dataframe
    data = pd.DataFrame(target_coords, columns=['x', 'y'])
    data['estimate'] = estimates[0]
    data['error_variance'] = error_variances
    
    # create pivoted dataframe for plotting purposes
    estimate_df = data.pivot(index='y', columns='x', values='estimate')
    error_df = data.pivot(index='y', columns='x', values='error_variance')
    x_df = data.pivot(index='y', columns='x', values='x')
    y_df = data.pivot(index='y', columns='x', values='y')
    
    make a colormap plot of the estimates
    fig, ax = plt.subplots()
    a = ax.pcolormesh(x_df, y_df, estimate_df, vmin=estimates.min(), vmax=estimates.max())

    plt.colorbar(a)  # show the color bar to the right
    ax.scatter(raw_df['x'], raw_df['y'],color='red', label="Sampled Point")  # plot known points on top
    plt.title(f"Estimated values ({workflow_config.UNITS}) for {workflow_config.PRIMARY_VAR_NAME}, range = {workflow_config.RANGE} {workflow_config.H_UNITS}")
    plt.savefig(f'images/{workflow_config.IM_TAG}_kriged_values.png')

    # make a colormap plot of the error variances
    fig, ax = plt.subplots()
    a = ax.pcolormesh(x_df, y_df, error_df, vmin=error_variances.min(), vmax=error_variances.max())
    plt.colorbar(a)  # show color bar
    ax.scatter(raw_df['x'], raw_df['y'],color='red', label="Sampled Point")  # plot known points on top

    plt.title(f"Error variance for {workflow_config.PRIMARY_VAR_NAME}, range = {workflow_config.RANGE} {workflow_config.H_UNITS}")
    plt.savefig(f'images/{workflow_config.IM_TAG}_kriged_error.png')

    fig, ax = plt.subplots()

    return 0


if __name__ == "__main__":
    main()
