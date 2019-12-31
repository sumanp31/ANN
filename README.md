# ANN

## Preprocessing Data
1. Convert categorical values using one hot encoding.

## Model Description

1. First activation layer has 6 units, is uniformly initialized and has a RELU activation function.

2. Second activation layer has 8 units, is uniformly initialized and has a RELU activation function.

3. Third activation layer has 6 units, is uniformly initialized and has a RELU activation function.

4. Fourth activation layer has 6 units, is uniformly initialized and has a RELU activation function.


## Experiment

1. There are 10000 data point. And it is split into training and testing dataset in the ratio of 80-20.

2. Grid search is used for the following hyperparameters:
	a) batch_size : [10, 25, 64]
	b) epochs : [100, 500, 700]
	c) optimizer : ['adam', 'rmsprop']

3. 10 crossfold validation is used.


## Result

The final accuracy is at 85.2% and it is achieved for the following set of hyperparameters. 
batch_size : 32
epochs : 700
optimizer : rmsprop

![](https://github.com/sumanp31/ANN/blob/master/result_screenshot.png) 