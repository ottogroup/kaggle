# Otto Group Product Classification Challenge

## About the competition

For the [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge), 
hosted by Kaggle, we have provided a dataset with 93 features for more 
than 200,000 products. The objective is to build a predictive model which 
is able to distinguish between our main product categories.

![competition graph](https://github.com/ottogroup/kaggle/blob/master/figure/Grafik2.jpg)

## Using the script

This repository contains a script for the benchmark submission of 
the competition. Use this script in the following way:

    python benchmark.py <path-to-train> <path-to-test> <name-of-submission>

Each argument is optional as the script will guess the right names if you don't
change them after downloading and put them in subfolder called _data_. It will
then create a submission called _my_submission.csv_ which should produce the
benchmark posted on the [leaderboard](https://www.kaggle.com/c/otto-group-product-classification-challenge/leaderboard).

## Requirements

To run the script, you will need to install the following packages:

* [Numpy](http://www.scipy.org/scipylib/download.html)
* [Pandas](http://pandas.pydata.org/getpandas.html)
* [Scikit-learn](http://scikit-learn.org/stable/install.html)

This script was tested using [Python 2.7.9](https://www.python.org/downloads/).

## Questions

If you have a question regarding this script or the competition in general,
head to the [forum](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums) 
and post them.




![ottogroup logo](http://ottogroup.com/wLayout/wGlobal/layout/images/logo-transparent.png)


