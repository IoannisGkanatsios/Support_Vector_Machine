# Introduction

This repository contains all the algorithms used to produce the results in the paper with the title: `Integration of Sentinel-1 and Sentinel-2 data for change detection: A case study in a war conflict area of Mosul city`


# Run the code

**Create virtual environment**

```
$ mkdir svm
$ cd svm
$ pipenv --python 3
$ pipenv install -r path/to/requirements.txt
```

**SVM classification**
```
usage: SVM.py [-h] 
              [-o OUTDIR] 
              [-i INPUTRAW] 
              [--train TRAIN] 
              [--tune] 

optional arguments:

  -h, --help            show this help message and exit  
  -o OUTDIR, --outdir OUTDIR
                        Specify an output directory                        
  -i INPUTRAW, --inputraw INPUTRAW  
                        Provide a path to the raw data
  --train TRAIN         Provide a path to the training data. A different shapefile for each landcover class 
                        for instance, for the following 3 classes: urban, river and crops, we should have 3 
                        shapefiles: urban.shp, river.shp and crops.shp  
  --tune                tune the model to choose the optimum hyperparameters 
  --tunetype TUNETYPE   select a method for tuning the SVM model. THe two optios are grid or random grid method exhoustively 
                        search all the values that have  been defined and trains the model for every possible combination. 
                        Random method uses a sample of the values provided which makes the optimization process much faster

```

# Change detection
```
usage: infrastructure_loss.py [-h] 
                              [--out_image OUT_IMAGE] 
                              [--before BEFORE] 
                              [--after AFTER]

optional arguments:
  -h, --help            show this help message and exit
  --out_image OUT_IMAGE
                        It outputs the change detection map. It shows the infrastructure loss and gain values of -1 shows 
                        the loss and values of 1 show the gain
  --before BEFORE       Provide a path to the data before the event. This is the reference image
  --after AFTER         Provide a path to the data after the event. This is the second image that is used to 
                        estimate the change in comparison with the first one

```


# Licence
                     
MIT