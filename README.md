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

**How to use the code**
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
  --train TRAIN         Provide a path to the training data  
  --tune                tune the model to choose the optimum hyperparameters 
  --tunetype TUNETYPE   select a method for tuning the SVM model. THe two optios are grid or random grid method exhoustively 
                        search all the values that have  been defined and trains the model for every possible combination. 
                        Random method uses a sample of the values provided which makes the optimization process much faster

```


# Licence
                     
MIT