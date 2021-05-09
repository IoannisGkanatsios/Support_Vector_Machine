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

import matplotlib.pylab as plt


def load_raster(raster):
    """Reads a raster file. The format should be '.tif' 
    """
    with rasterio.open(raster) as src:
        glcm_profile = src.profile
        glcm = src.read()
        glcm_shape = glcm.shape

        if np.isinf(glcm).any() or np.isnan(glcm).any():
            glcm[np.isnan(glcm)] = 0
            glcm[np.isinf(glcm)] = 0

        print('Input data has', '{}'.format(glcm_shape[0]), 'features')
        print('The dimensions of the input data is',
              'width:{} Height:{}'.format(src.width, src.height), '\n')
    return glcm, glcm_profile


def write(raster_input, raster_profile, raster_output):
    """ writes a raster file
    """
    profile = raster_profile
    profile.update(
        dtype=raster_input.dtype,
        compress='lzw',
        count=raster_input.shape[0],
        nodata=0)

    with rasterio.open(raster_output, "w", **profile) as dst:
        print('Writing the file....')
        for i, band in enumerate(raster_input):
            dst.write_band(i + 1, band)


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


def plot_components(eig_val, scaled_data, pca):
    """Plot the Principal Components that contribute the most
    """
    fig, ax = plt.subplots(ncols=2, figsize=(12, 3))
    ax[0].plot(np.cumsum(eig_val))
    ax[0].set_xlabel('Number of Components')
    ax[0].set_ylabel('Variance (%)')
    ax[0].grid(True, lw=1, ls='--', c='.75')
    ax[0].set_title('Number of PC selection')

    # project the data in 2D
    pca.transform(scaled_data)
    explained_variance = np.round(
        pca.explained_variance_ratio_*100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(explained_variance)+1)]

    ax[1].bar(x=range(1, len(explained_variance)+1),
              height=explained_variance, tick_label=labels)
    ax[1].set_xlabel('Number of PC that contribute the most')
    ax[1].set_ylabel('Significance of the Principal Components')
    plt.show()
    return


def PCA(eig_val, scaled_data, raster):
    """Calculate PCA
    """
    pca = decomposition.PCA(n_components=len(eig_val))
    dataset = pca.fit_transform(scaled_data)
    glcm_reshape = dataset.reshape(
        raster.shape[1], raster.shape[2], dataset.shape[1])
    # move the number of bands at the front of the 3d array (bands x rows x cols), so that we can write the file
    glcm_reshape = np.rollaxis(glcm_reshape, 2, 0)
    return glcm_reshape


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o',
        '--outdir',
        required=False,
        help='Specify an output directory'
    )

    parser.add_argument(
        '-i',
        '--inputraw',
        type=str,
        required=False,
        help='Provide a path to the raw data'
    )

    args = parser.parse_args()

    if not args.outdir:
        parser.error('Please provide an output path (use the flag -o)')

    if args.inputraw is None:
        parser.error(
            'Please provide a path to input data (use the flag -i)')

    else:

        data, data_profile = load_raster(args.inputraw)
        data_scaled = scale_data(data)
        variance, pca = explained_variance(data_scaled)
        PCA_calculation = PCA(variance, data_scaled, data)
        write(PCA_calculation, data_profile, args.outdir)
        plot_components(variance, data_scaled, pca)
