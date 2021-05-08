from pathlib import Path
import argparse

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd


def write_raster(raster, crs, dst_transform, raster_output):
    profile = {
        'driver': 'GTiff',
        'compress': 'lzw',
        'width': raster.shape[0],
        'height': raster.shape[1],
        'crs': crs,
        'transform': dst_transform,
        'dtype': raster.dtype,
        'count': 1,
        'tiled': False,
        'interleave': 'band'
    }

    profile.update(
        dtype=raster.dtype,
        height=raster.shape[0],
        width=raster.shape[1],
        compress='lzw',
        nodata=0)

    with rasterio.open(raster_output, 'w', **profile) as out:
        out.write_band(1, raster)


def read(raster_path):
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        crs = src.crs
        transform = src.transform
        profile = src.profile
    return band, crs, transform, profile


def raster_align(source,
                 dst,
                 src_transform,
                 src_crs,
                 dst_transform,
                 dst_crs):

    destination = np.ndarray((dst.shape))
    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=0,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0,
        resampling=Resampling.nearest)

    return destination


def difference(raster_1, raster_2):
    # re-classify the arrays.keep only pixels that correspond to buildings
    before = np.where(raster_1 != 1, 0, 1).astype(np.int16)
    after = np.where(raster_2 != 1, 0, 1).astype(np.int16)

    # take the difference between before and after images
    diff = after - before

    return diff


def loss_estimation(raster, pixel_size):
    # estimate infrastructure loss and gain
    loss = np.count_nonzero(raster == -1)
    total_loss = ((pixel_size * pixel_size) * loss) / 1000

    gain = np.count_nonzero(raster == 1)
    total_gain = ((pixel_size * pixel_size) * gain) / 1000

    no_change = np.count_nonzero(raster == 0)
    no_change = ((pixel_size * pixel_size) * no_change) / 1000

    print(f"Infrastructure loss: {total_loss}")
    print(f"Infrastructure gain: {total_gain}")
    print(f"Unchanged: {no_change}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o',
        '--output',
        required=False,
        help='Specify an output directory'
    )

    parser.add_argument(
        '--before',
        type=str,
        required=False,
        help='Provide a path to the raw data'
    )

    parser.add_argument(
        '--after',
        type=str,
        required=False,
        help='Provide a path to the raw data'
    )

    args = parser.parse_args()

    if not args.before or not args.after:
        parser.error(
            'Please provide two images acquired at different times (before and after)')

    elif (args.before and args.after) and not args.output:

        # load before and after images
        raster_1, crs1, transform1, profile1 = read(args.before)
        raster_2, crs2, transform2, profile2 = read(args.after)
        # align after (image) over before
        raster_2 = raster_align(raster_2,
                                raster_1,
                                transform2,
                                crs2,
                                transform1,
                                crs1
                                )
        pixel_size = transform1[0]
        img_difference = difference(raster_1, raster_2)
        loss = loss_estimation(img_difference, pixel_size)

    else:
        # load before and after images
        raster_1, crs1, transform1, profile1 = read(args.before)
        raster_2, crs2, transform2, profile2 = read(args.after)
        # align after (image) over before
        raster_2 = raster_align(raster_2,
                                raster_1,
                                transform2,
                                crs2,
                                transform1,
                                crs1
                                )
        pixel_size = transform1[0]
        img_difference = difference(raster_1, raster_2)
        loss = loss_estimation(img_difference, pixel_size)
        write_raster(img_difference, crs1, transform1, args.output)
