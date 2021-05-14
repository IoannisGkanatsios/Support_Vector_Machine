from pathlib import Path
import argparse
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
import rasterio
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


def scale_data(raster):
    """ It scales the input data between 0 and 1
    """
    # remove nans. PCA does not like nan values
    where_are_NaNs = np.isnan(raster)
    raster[where_are_NaNs] = 0

    new_shape = raster.shape[0], raster.shape[1]*raster.shape[2]
    glcm_2d = raster[:, :].reshape(new_shape)
    scaler = MinMaxScaler(feature_range=[0, 1])
    data_rescaled = scaler.fit_transform(glcm_2d.T)
    return data_rescaled


def explained_variance(scaled_data):
    """It estimates the number of PC with a variance above 95%
    """
    pca = decomposition.PCA(0.95)
    pca.fit_transform(scaled_data)
    # Estimate the eigvalues of the eigvectors. The eigenvectors
    # with the lowest eigenvalues bear the least information
    eig_val = pca.explained_variance_ratio_
    print('{}'.format(len(eig_val)), 'Principal Components have been chosen', '\n')
    return eig_val, pca


def PCA(eig_val, scaled_data, raster):
    """Calculate PCA
    """
    pca = decomposition.PCA(n_components=len(eig_val))
    dataset = pca.fit_transform(scaled_data)
    glcm_reshape = dataset.reshape(
        raster.shape[1], raster.shape[2], dataset.shape[1])
    # move the number of bands at the front of the 3d array (bands x rows x cols), so that we can write the file
    return glcm_reshape


def PCA_calculation(data):
    data_scaled = scale_data(data)
    variance, pca = explained_variance(data_scaled)
    pca = PCA(variance, data_scaled, data)
    return pca
