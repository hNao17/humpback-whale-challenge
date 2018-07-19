## Overview
The training / test sets can be downloaded from the main Kaggle challenge page: https://www.kaggle.com/c/whale-categorization-playground/data

Image data is saved in JPEG format for both training and test sets. Additionally, the training set download has a CSV file that contains information relating whale ids to specific image files. 

## Training / Test Split
The training set has 9850 labeled images, while the test set has 15,610 unlabeled images. Some examples from training set are shown in the image below:

![alt_text](/images/training_set_examples.png)

Some observations:
- contains both RGB and grayscale images
- common image characteristics
    - the whale fluke is the dominant / largest object in most images
    - a majority of the whale fluke is positioned out of the water
- anomalies
    - the whale fluke is small / distant in the image background (i.e w_c90feaa in (3,3))
    - the whale fluke is almost completely submerged in water (i.e. w_896023f in (1,2))
    - text is shown at the bottom of the image (i.e. w_37523b2 in (3,2))
    - image is rotated by more than 90 degress (i.e w_5b99089 in (3,2), w_dbb786d in (3,5))

## Whale Id Distribution
There are 4251 ids in the training set, with 4250 corresponding to unique 9-character/digit whale ids and the remaining 1 id corresponding to the "new_whale" category. 

A coarse look at the distribution of images per whale id is as follows: 
- bin 1: N = 1 image: 2220
- bin 2: 1 < N < 10 images: 1965
- bin 3: N >= 10 images: 66

Over 50% of whale ids only have 1 image, whereas just 1.6% of ids have 10 or more images. For the third bin, a more detailed distribution is shown in the bar graph below: 

![alt_text](/images/id_distribution.png?v=2)

The most frequent id in the training set is the "new_whale", with a total of 810 images. The drop-off between new_whale and the second-most frequent id "w_1287fbc" is quite significant, with "w_1287fbc" only having 34 images.

## Image Size Distribution
2587 unique image resolutions / sizes appear in the training set. The top 10 most frequent sizes are shown below:

![alt_text](/images/img_size_distribution.png?v=2)

The most frequent size is (1050,600) with 1113 images. 

## RGB / Grayscale Distribution
Since all JPEG images are imported as 3-channel images, a pixel-wise variance metric can be used to determine if an image is truely RGB or grayscale. For grayscale, if the mean pixel variance is equal to 0, the image is considered to be grayscale.

From this, of the 9850 images on the training set, 4915 (~49.9%) images are grayscale and 4935 (50.1%) are RGB.

## Conclusions
The training set's image characteristics can help make several design decisions regarding training:

### Classifier Type
Because of the training set's unequal distribution of ids, using a vanilla CNN will most likely not work. Even with data augmentation, for ids with only 1 image, such a network would struggle to generalize to new examples for those particular ids. Therefore, it may be advantageous to look towards siamese network methods where the number of unique images per id is less important as opposed to forming similar and dissimilar id pairs. 

### Architecture 
Most traditional architectures (i.e VGG, ResNet) use fixed image sizes and RGB images. Since this datset is split roughly 50% / 50% in terms of grayscale / RGB, either all images should be converted to grayscale or vice-versa. With the former, this would require modifying a traditional architecture to work with 1-channel images.


