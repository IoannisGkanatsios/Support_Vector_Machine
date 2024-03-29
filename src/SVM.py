"""
This script performs image classification based on the SVM classifier

"""

from pathlib import Path
import argparse

from multiprocessing import Pool, cpu_count
import time

import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

import pandas as pd
import geopandas as gpd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from PCA import PCA_calculation


def int_or_str(value):
    """
    parameters
    ----------
    value: [str, int]
        This funtion is used in the parameter 'type' of argparse module.
        This allows us to use either str or int as an argument
    """
    try:
        return int(value)
    except:
        return value


def load_raster(input_file):
    """Returns a raster array which consists of its bands and transformation matrix
    parameters
    ----------
    input_file: str
        path directory to the raster file
    sensor: str
        choose between landsat and senstinel2
    reproject: if true, reprojects the data into WGS 84
    """
    with rasterio.open(input_file) as src:
        raster_raw = src.read()
        transform = src.transform
        crs = src.crs
        shape = src.shape
        profile = src.profile
        raster = np.rollaxis(raster_raw, 0, 3)

        return raster, raster_raw,  profile


def write_raster(raster, crs, transform, output_file):

    profile = {
        'driver': 'GTiff',
        'compress': 'lzw',
        'width': raster.shape[0],
        'height': raster.shape[1],
        'crs': crs,
        'transform': transform,
        'dtype': raster.dtype,
        'count': 1,
        'tiled': False,
        'interleave': 'band',
        'nodata': 0
    }

    profile.update(
        dtype=raster.dtype,
        height=raster.shape[0],
        width=raster.shape[1],
        nodata=0,
        compress='lzw')

    with rasterio.open(output_file, 'w', **profile) as out:
        out.write_band(1, raster)


def rasterize(raster, vector, profile):
    """Use of vector data to train the SVM model.

    parameters
    ----------
    raster: ndarray
        raster (raw data) to be used for the classification (numpy array)
    vector: shapefile
        number of shapefiles which each one of them corresponds to a landcover class
    """
    proj = profile['crs']
    labeled_pixels = np.zeros((raster.shape[0], raster.shape[1]))
    for i, shp in enumerate(sorted(vector, reverse=True)):
        label = i + 1
        df = gpd.read_file(shp)
        df = df.to_crs(crs=proj)
        geom = df['geometry']
        vectors_rasterized = features.rasterize(geom,
                                                out_shape=raster.shape[0:2],
                                                transform=profile['transform'],
                                                all_touched=True,
                                                fill=0,
                                                default_value=label)
        labeled_pixels += vectors_rasterized

    return labeled_pixels


def split(raster_img, labeled_pixels, training_data, split=0.30):
    """Splits the observations into training and testing.

    parameters
    ----------
    raster_img: ndarray
        raster (raw data) to be used for the classification (numpy array)
    labeled_pixels: ndarray
        training data that has been rasterized
    split: Proportion of the data for testing. Default value is 30%.
    """
    for i, shp in enumerate(sorted(training_data, reverse=True)):
        count = i + 1
        land_classes = shp.stem
        print('Class {land_classes} contains {n} pixels'.format(
            land_classes=land_classes, n=(labeled_pixels == count).sum()))

    roi_int = labeled_pixels.astype(int)
    # X is the matrix containing our features
    X = raster_img[roi_int > 0]
    # y contains the values of our training data
    y = labeled_pixels[labeled_pixels > 0]

    # Split our dataset into training and testing. Test data will be used to
    # make predictions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, stratify=y)

    return X_train, X_test, y_train, y_test


def svm_params(X_train, y_train, tune=False, search_type=None):
    """Returns the best parameters (C and gamma) for the SVM model
    parameters
    ----------
    X_train: ndarray
        subset of the features used for training
    y_train: ndarray
        subset of the labelled data
    tune: if tune==True, the algorithm is trying to find the best parameters for the model
          if tune==False, the model uses some pre-defined parameters that work generally well
    search_type: str
        search_type options: random, which uses RandomizedSearchCV to tune the hyperparameters
                             grid, which uses GridSearchCV to tune the hyperparameters (computational expensive)
    """
    if tune:
        param_range_c = np.logspace(0, 2, 8)
        param_range_gamma = np.logspace(-6, -1, 8)

        param_grid = {'svm__C': param_range_c,
                      'svm__gamma': param_range_gamma}

        pip = Pipeline([('scale', preprocessing.StandardScaler()),
                        ('svm', SVC(kernel='rbf', class_weight='balanced'))])

        if search_type == 'grid':
            clf = GridSearchCV(estimator=pip,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=3,
                               n_jobs=-1)

            clf = clf.fit(X_train, y_train)

            # print accuracy of the model
            print('Best parameters:', clf.best_params_)
            print('Classification accuracy', clf.best_score_)

        elif search_type == 'random':
            clf = RandomizedSearchCV(estimator=pip,
                                     param_distributions=param_grid,
                                     scoring='accuracy',
                                     cv=3,
                                     n_iter=15,
                                     error_score='raise',  # it supresses the warning error
                                     n_jobs=-1)

            clf = clf.fit(X_train, y_train)

            # print accuracy of the model
            print('Best parameters:', clf.best_params_)
            print('Classification accuracy:', clf.best_score_)

    else:
        pip = Pipeline([('scale', preprocessing.StandardScaler()), ('svm', SVC(
            kernel='rbf', C=100, gamma=0.1, class_weight='balanced'))])
        clf = pip.fit(X_train, y_train)

    return clf


def predict(raster_img):
    """It makes predictions on an array of dimensions (cols,rows,features)
    parameters
    ----------
    raster_img: ndarray
        raster (raw data) to be used for the classification (numpy array)
    """
    y_predict = clf.predict(raster_img)
    return y_predict


def image_classification(raster_img, cpu_num=None, count=None):
    """It performs SVM classification using parallel processing
    parameters
    ----------
    raster: ndarray
        crs from the raster dict
    raster_img: ndarray
        ndarray from the raster dict to be used for the classification

    """
    # Reshape the data so that we make predictions for the whole raster
    new_shape = (raster_img.shape[0] *
                 raster_img.shape[1], raster_img.shape[2])

    img_as_array = raster_img[:, :].reshape(new_shape)
    image_array = np.copy(img_as_array)

    if cpu_num == 'all':
        # split good data into chunks for parallel processing
        cpu_n = cpu_count()

        split_img = np.array_split(image_array, cpu_n)

        # run parallel processing of all data with SVM
        pool = Pool(cpu_n)
        svmLablesPredict = pool.map(predict, split_img)

        # join results back from the queue and insert into full matrix
        svmLablesPredict = np.hstack(svmLablesPredict)
        svm_classified = svmLablesPredict.reshape(
            raster_img.shape[0], raster_img.shape[1])

    elif cpu_num == 'number':
        # split good data into chunks for parallel processing
        cpu_n = count

        split_img = np.array_split(image_array, cpu_n)

        # run parallel processing of all data with SVM
        pool = Pool(cpu_n)
        svmLablesPredict = pool.map(predict, split_img)

        # join results back from the queue and insert into full matrix
        svmLablesPredict = np.hstack(svmLablesPredict)
        svm_classified = svmLablesPredict.reshape(
            raster_img.shape[0], raster_img.shape[1])

    elif cpu_num == 'none':
        svmLablesPredict = clf.predict(image_array)
        svm_classified = svmLablesPredict.reshape(
            raster_img.shape[0], raster_img.shape[1])

    return svm_classified


def model_accuracy(raster, profile, svm_classified, shp):
    """It produces a classification report and confusion matrix of the classified raster

    parameters
    ----------
    raster: ndarray
        raster to be used for classification
    svm_classified: ndarray
        The classified raster
    shp: vector
        Location of shapefiles
    """
    labeled_pixels = rasterize(raster, shp, profile)
    target_names = [s.stem for s in shp]

    for_verification = np.nonzero(labeled_pixels)
    verification_labels = labeled_pixels[for_verification]
    predicted_labels = svm_classified[for_verification]

    print('Confusion matrix: \n %s' %
          confusion_matrix(verification_labels, predicted_labels))

    print('\n')

    print(
        'Classificaion report: \n %s' %
        classification_report(
            verification_labels,
            predicted_labels,
            target_names=target_names))

    return confusion_matrix, classification_report


if __name__ == "__main__":

    start = time.time()
    print('SVM classification starts....', '\n')

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

    parser.add_argument(
        '--train',
        type=str,
        required=False,
        help='Provide a path to the training data. A different shapefile for each landcover class \
            for instance, for the following 3 classes: urban, river and crops, we should have 3 \
            shapefiles: urban.shp, river.shp and crops.shp'
    )

    parser.add_argument(
        '--tune',
        action='store_true',
        help='Tune the SVM model so that it chooses the optimum hyperparameters (use the flag --tunetype) \
            to choose the method of tuning.'
    )

    parser.add_argument(
        '--tunetype',
        type=str,
        required=False,
        help='Select a method for tuning the SVM model. The two optios are grid or random \
            grid method exhoustively search all the values that have been defined and trains the model \
            for every possible combination. Random method uses a smaple of the values provided which \
            makes the optimization process much faster'
    )

    parser.add_argument(
        '--cpu',
        type=int_or_str,
        required=False,
        help='Select the number of CPUs to be used during processing. if --cpu all passed as an argument \
            then the computer uses all the CPU cores. If --cpu <int number> passed as an argument \
            then the computer uses the number of cores specifed by the user, else only one core is used'
    )

    parser.add_argument(
        '--pca',
        action='store_true',
        required=False,
        help='Performs dimensionality reduction based on the PCA algorithm'
    )

    args = parser.parse_args()

    if not args.outdir:
        parser.error('Please provide an output path (use the flag -o)')

    if args.inputraw is None:
        parser.error(
            'Please provide a path to input data (use the flag -i)')

    if args.train is None:
        parser.error(
            'Please provide a path to the training data (use the flag --train)')

    if not args.pca:
        raster_name = Path(args.inputraw).stem
        print(
            f"The sattelite scene: {raster_name} is being processed", '\n')

        raster, raster_raw, profile = load_raster(args.inputraw)
        training_data = [train for train in Path(args.train).glob('*.shp')]

        rasterized = rasterize(raster=raster,
                               vector=training_data,
                               profile=profile)

        X_train, X_test, y_train, y_test = split(
            raster, rasterized, training_data)

    elif args.pca:
        raster_name = Path(args.inputraw).stem
        print(
            f"The sattelite scene: {raster_name} is being processed", '\n')

        raster, raster_raw, profile = load_raster(args.inputraw)
        training_data = [train for train in Path(args.train).glob('*.shp')]

        raster = PCA_calculation(raster_raw)

        rasterized = rasterize(raster=raster,
                               vector=training_data,
                               profile=profile)

        X_train, X_test, y_train, y_test = split(
            raster, rasterized, training_data)

    if args.tune and not args.tunetype:
        parser.error(
            'Please specify the type of tunning. options: grid or random (use the flag --tunetype)')

    elif args.tune and args.tunetype == 'random':
        print('Model tunning based on random method. This process is very computationally expensive')
        clf = svm_params(X_train, y_train, tune=True, search_type='random')

    elif args.tune and args.tunetype == 'grid':
        print('Model tunning based on grid method. This process is very computationally expensive')
        clf = svm_params(X_train, y_train, tune=True, search_type='grid')

    elif not args.tune:
        clf = svm_params(X_train, y_train)

    if args.cpu == 'all':
        svm_classified = image_classification(
            raster, cpu_num='all')

    elif args.cpu:
        svm_classified = image_classification(
            raster, cpu_num='number', count=args.cpu)

    elif not args.cpu:
        svm_classified = image_classification(
            raster, cpu_num='none')

    model_accuracy(raster, profile, svm_classified, training_data)

    print('writting...', '\n')
    write_raster(
        svm_classified, profile['crs'], profile['transform'], args.outdir)

    end = time.time()
    print(
        f"the processing time is:, {round((end - start)/60, 2)} minutes", '\n')
    print('ALL DONE!')
