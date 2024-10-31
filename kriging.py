#!/usr/bin/python3
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

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

    parser.add_argument("--model",
        choices=['exponential', 'hw6'],
        help="""
        Select a model to fit the semivariogram on, when used in 'semivariogram' mode.
        Choose a single model or 'all'. Currently must be used with "--range" setting.
        """)

    parser.add_argument("--range", type=float,
        help="""
        If used in 'semivariogram' setting, this sets the range the model will
        be fitted on.
        """)
    # TODO: add option to run all models


    try:
        options = parser.parse_args()
    except:
        parser.print_help()
        return 1

    workflow_config = importlib.import_module(f'config.{options.config_file}')
    print(workflow_config)

    #############################################################
    # Kriging
    #############################################################

    # get raw dataset of known sample points
    raw_df = workflow_config.raw_data()

    # get the model
    # fit the model to the data
    model_class = models.get_model(options.model)
    model = model_class(data_df=raw_df, fit_range=options.range)

    # get coordinates of the sample points to calculate distances
    sample_coords = raw_df[['x', 'y']].values

    # create distance matrix using scipy function
    # it uses minkowski distance, p=2 is same as Euclidean distance
    pair_matrix = distance_matrix(sample_coords, sample_coords, p=2)  #

    # calculate semivariance from model and distance matrix
    semivariance_knownpts_matrix = model.fit_h(pair_matrix)

    # add the ones and zero as described in class (due to Lagrange Param)
    b = np.ones((len(sample_coords),1))
    q = np.ones((1, len(sample_coords) + 1))
    C_semivariance_matrix = np.hstack((semivariance_knownpts_matrix,b))
    C_semivariance_matrix = np.vstack((C_semivariance_matrix,q))
    C_semivariance_matrix[len(sample_coords), len(sample_coords)] = 0

    # construct a target grid
    # get min and max coords from sample data so we know where to center our grid
    min_x = raw_df['x'].min()
    min_y = raw_df['y'].min()
    max_x = raw_df['x'].max()
    max_y = raw_df['y'].max()

    # lets add a tolerance around the edges
    x_tol = (max_x - min_x) * .1
    y_tol = (max_y - min_y) * .1

    # now make a 20 x 20 grid around the sample points
    n_pts = 5
    xs = np.linspace(min_x - x_tol, max_x + x_tol, n_pts)
    ys = np.linspace(min_y - y_tol, max_y + y_tol, n_pts)
    target_coords = [[i,j] for i in xs for j in ys]

    # calculate the distances between the sample points and the target point
    target_coords = [[7, 14]]

    print("Vector Target Distance")
    target_vector = distance_matrix(sample_coords, target_coords)
    D_semivariance_target_matrix = model.fit_h(target_vector)
    print("target vector shape", target_vector.shape)
    print("D matrix shape (pre adding Lagrange)", D_semivariance_target_matrix.shape)

    # Add one on to end of vector due to Lagrange Param
    q = np.ones((1, len(target_coords)))
    D_semivariance_target_matrix = np.vstack((D_semivariance_target_matrix,q))
    print("D matrix shape (with Lagrange)", D_semivariance_target_matrix.shape)

    # do the matrix operations to get the w matrix
    C_inv = np.linalg.inv(C_semivariance_matrix)
    W_weights_matrix = np.matmul(C_inv, D_semivariance_target_matrix)
    print("W matrix shape", W_weights_matrix.shape)
    # print(W_weights_matrix[:-1].sum())  # check sum to 1 (except Lagrange param) -- YES!

    # get the values of the known samples points
    actual_values_vector = raw_df[['primary']].values

    # add in a zero to deal with the Lagrange param
    q = np.zeros((1,1))
    actual_values_vector = np.vstack((actual_values_vector, q))
    actual_values_col = np.rot90(actual_values_vector)
    print("actual values shape", actual_values_col.shape)

    # use matrix multiplication to calculate estimated value
    estimates = np.matmul(actual_values_col, W_weights_matrix)
    print(estimates)
    print("estimate matrix", estimates.shape)
    
    # # turn the coords and estimates into a dataframe
    # data = np.hstack((target_coords, np.rot90(estimates)))
    # data_df = pd.DataFrame(data, columns=['x','y', 'estimate'])


    # fig, ax = plt.subplots()
    # c = ax.pcolormesh(data_df['x'], data_df['y'], data_df['estimate'],
    #     cmap='RdBu',
    #     vmin=data_df['estimate'].min(),
    #     vmax=data_df['estimate'].max())
    # ax.set_title('pcolormesh')
    # # set the limits of the plot to the limits of the data
    # ax.axis([data_df['x'].min(), data_df['x'].max(), data_df['y'].min(), data_df['y'].max()])
    # fig.colorbar(c, ax=ax)

    # plt.show()

    return 0


if __name__ == "__main__":
    main()
