# COMP4336_Proj

## Introduction 
The project involves detecting gestures from WiFi packets using the RSS values. This project tunes a LSTM-FCN model based of the following paper [LSTM Fully Convolutional Networks for Time Series Classification](https://ieeexplore.ieee.org/document/8141873).

The majority of the codebase was forked from [https://github.com/titu1994/LSTM-FCN.git](https://github.com/titu1994/LSTM-FCN.git)

The original codebase contains a LSTM-FCN model and an attention based LSTM-FCN model. For our needs we have only applied to the baseline LSTM-FCN model as it achieved reasonable results. 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13hrYU_LfKJHxtbwVhUFvzwzAIu4PHnBK?usp=sharing)


## Requirements 

* tensorflow-gpu>=1.2.0.
* tensorflow==2.2.
* keras==2.3.1.
* scipy.
* numpy.
* pandas.
* scikit-learn>=0.18.2.
* h5py.
* matplotlib.
* joblib>=0.12.


## Installation 
Download the repository and apply pip install -r requirements.txt to install the required libraries.

Keras with the Tensorflow backend has been used for the development of the models

## Data
The project folder contains two data folders. One folder contains CSV files labeled /CSV_files and the second folder contains the model dataset used for training/testing labeled /data


### CSV Files
The following CSV files contains beacon frame information exported from a wireshark pcapng file. These CSV files have been manually cleaned to contain only the frame times and the RSS values. Only the frames between 0 and 200 seconds are stored in these files. 

The data collection procedure was completed in 4 rounds. Each round was completed by performing and collecting the RSS data from the 3 gestures. 

The 3 gestures selected are vertical hand gesture, wave up and down gesture and push and pull gesture. 

The CSV files are labeled in the following format:

```

[Gesture Name]_[Round Number].csv

```

### Model Files 
Model Files are stored in the folder /data. These files contain RSS values from the CSV files and are preprocessed to the data format used to train and test the LSTM-FCN model.

To convert the CSV files to the model data format run the following code

```

python preprocessing.py

```

The default values in this file generates a training and test split of the 240 dataset. Training and Test datasets are stored in separate files and are used in conjunction with one anoher. 

The files in this folder are described below:

* RSS_TRAIN_60  : The training set for the 60 sample dataset 
* RSS_TEST_60   : The test set for the 60 sample dataset 
* RSS_TRAIN_120 : The training set for the 120 sample dataset 
* RSS_TEST_120  : The test set for the 120 sample dataset 
* RSS_TRAIN_180 : The training set for the 180 sample dataset 
* RSS_TEST_180  : The test set for the 180 sample dataset
* RSS_TRAIN_240 : The training set for the 240 sample dataset 
* RSS_TEST_240  : The test set for the 240 sample dataset

The folder also contains the 4 fold cross validation split used on the 240 gesture dataset 

* RSS_TRAIN1  : The training set for split 1
* RSS_TEST1   : The test set for split 1
* RSS_TRAIN2  : The training set for split 2
* RSS_TEST2   : The test set for split 2
* RSS_TRAIN3  : The training set for split 3
* RSS_TEST3   : The test set for split 3
* RSS_TRAIN4  : The training set for split 4
* RSS_TEST4   : The test set for split 4

## Training 

To train the following model run the following command:

```

python all_datasets_training.py

```
The code currently is set to training on the first split of the 240 gesture dataset. The code currently performs the training process and then the evaluation process afterwards.

A few parameters must be set in advance :

Datasets: Datasets must be listed as a pair (RSS, 0). The (name, id) pair has been preset in constants.py inside the utils directory. To adjust the the training and test dataset update the training and test file path in the file.

Models : The model_function can accept 3 parameters - maximum sequence length, number of classes. These values are been preset to 27 and 3 respectively for our project.

Cells : The configurations of cells required to be trained over. he values can be either [8,64,128]. For our results, we used an 8 cell configuration for training and test as the results gained were reasonable.



## Evaluation 

To evaluate the model comment out the following line in all_datasets_training.py.


```

train_model(model, did, dataset_name_, epochs=2000, batch_size=128,normalize_timeseries=normalize_dataset)

```

This will skip the training process and only evaluate the model. 

















