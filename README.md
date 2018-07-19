## Overview
The objective of [Kaggle's Humpback Whale Challenge](https://www.kaggle.com/c/whale-categorization-playground) is to classify whales given an image of a whale fluke (tail). The challenge requires that for a given image, a classifier make top5 predictions over an entire set of N whale ids. 

Our best submission (team name Improbability Drive) scored 79th on the [public leaderboard](https://www.kaggle.com/c/whale-categorization-playground/leaderboard). Please see below for a summary of our training and testing methodology.

## Data
Kaggle provides both training and test sets for this challenge. The training and test sets consist of 9850 and 15,610 images respectively. Please [see here](data/README.md) for more details.

From the training set data, we create a database of 4250 entries, with each entry corresponding to a unique whale id and image. 

## Method

Classification predictions are made using the following pipeline:

Raw Image > Landmarks Prediction > Deskew > Bounding Box Prediction > Crop > Similarity Prediction 

First 4 landmarks (keypoints) are predicted on the whale fluke, which can be used to deskew the raw image. From the deskewed image, a bounding box is predicted around the borders of the fluke, from which the image is cropped. The final stage of the above pipeline involves predicting a similarity between the raw image and images in the whale id database.

## Training

## Test Set Evaluation
