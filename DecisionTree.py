"""
Add header text here
"""

#-----------------------------------------------------------------------------#
#--------------------------Import Packages------------------------------------#
import numpy as np
import sklearn
import matplotplib.pyplot as plt


class TreeNode():
    """
    A class for representing one node of a decision tree, and, recursively, 
    an entire decision tree. Contains functions for calculating impurity metrics,
    splitting nodes, pruning, and plotting. Throughout, data examples are assumed
    to be represented as numpy arrays:
        X - m x n array of examples (m examples, n features)
        Y - m x 1 array of labels (m examples, 1 label per example)
    """
    def __init__(X,Y,criterion = "gini"):
        """
        Initialize a TreeNode
        """
        # row indices in X and Y of all examples within the node 
        #(data is not stored explicity in the nodes)
        self.contents = []
        # root node has depth 0 by convention
        self.depth = 0
        # contains zero or two TreeNodes
        self.children = []
        #"gini" or "entropy" impurity
        self.split_criterion = criterion
        
        # store all data, not just the examples within a given node
        self.X = X
        self.Y = Y 