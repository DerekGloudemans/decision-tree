"""
Derek Gloudemans
Machine Learning Assignment 1
"""

#-----------------------------------------------------------------------------#
#--------------------------Import Packages------------------------------------#
import numpy as np
import copy
import csv
from DecisionTree import TreeNode

def load_cancer_CSV(file_name):
    """
    loads data from CSV (where first column in class label) into an X and Y 
    np array
    file_name - string - name of data csv.
    returns:
        X - m x n array of features for data examples
        Y - m x 1 array of labels for data examples
    """
    data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    Y = data[:,0]
    X = data[:,1:]
    return X,Y

X,Y = load_cancer_CSV("cancer_datasets_v2/training_1.csv")