"""
Add header text here
"""

#-----------------------------------------------------------------------------#
#--------------------------Import Packages------------------------------------#
import numpy as np
import sklearn
import matplotlib.pyplot as plt


class TreeNode():
    """
    A class for representing one node of a decision tree, and, recursively, 
    an entire decision tree. Contains functions for calculating impurity metrics,
    splitting nodes, pruning, and plotting. Throughout, data examples are assumed
    to be represented as numpy arrays:
        X - m x n array of examples (m examples, n features)
        Y - m array of labels (m examples, 1 label per example)
    """
    
    def __init__(self,X,Y,criterion = 'gini'):
        """
        Initialize a TreeNode
        """
        # row indices in X and Y of all examples within the node 
        #(data is not stored explicity in the nodes)
        
        # store all data, not just the examples within a given node
        self.X = X
        self.Y = Y 
        
        self.contents = [i for i in range(len(Y))]
        # root node has depth 0 by convention
        self.depth = 0
        # contains zero or two TreeNodes
        self.children = []
        #"gini" or "entropy" impurity
        self.split_criterion = criterion
        
        # keep track of node's impurity (all grouped into 1 partition)
        if criterion == 'gini':
            self.gini_val,_ = self.gini(0,np.inf)
        else:
            self.entropy_val,_= self.entropy(0,np.inf)
        
 

    def gini(self,j,t):
        """
        Calculate the gini impurity for a subset of tree data partitioned on 
        a given feature j and split value t
        x - np array - all feature data at the node - x = self.X[self.contents,:]
        y - np array - all label data at the node   - y = self.Y[self.contents,:]
        j - integer - split feature
        t - float - split threshold (data is split <= t, > t)
        returns:
            partition - list of lists, indicies in left and right partition 
            impurity - gini impurity
        """
        x = self.X[self.contents,:]
        y = self.Y[self.contents]
        
        # get indices of left partition
        left = [i[0] for i in (np.argwhere(x[:,j] <= t))]
        right = [i[0] for i in (np.argwhere(x[:,j] > t))]
        
        if left:
            # get most common labels of each partition
            mcl_left = np.bincount(y[left]).argmax()
            # count correct labels
            left_correct = len(np.argwhere(y[left] == mcl_left))
            n_left = len(left)
            left_ratio = 1- left_correct/n_left
        else:
            left_ratio = 0
            n_left = 0
        if right:
            mcl_right = np.bincount(y[right]).argmax()
            right_correct = len(np.argwhere(y[right] == mcl_right))
            n_right = len(right)
            right_ratio = 1- right_correct/n_right
        else:
            right_ratio = 0
            n_right = 0      
            
        impurity = left_ratio * n_left/len(y) + right_ratio * n_right/len(y)
        
        return impurity, [left,right]
        
    def information_gain(self,j,t):
        """
        Calculate the information_gain for a subset of tree data partitioned on 
        a given feature j and split value t
        x - np array - all feature data at the node - x = self.X[self.contents,:]
        y - np array - all label data at the node   - y = self.Y[self.contents,:]
        j - integer - split feature
        t - float - split threshold (data is split <= t, > t)
        returns:
            partition - list of lists, indicies in left and right partition 
            information_gain - change in entropy impurity
        """
        

        child_entropy, [left,right] = self.entropy(j,t)
        
        information_gain = self.entropy_val - child_entropy
        return information_gain, [left,right]
        
    def entropy(self,j,t):
        """
        Used to get initial entropy value for tree
        Calculate the entropy for a subset of tree data partitioned on 
        a given feature j and split value t
        x - np array - all feature data at the node - x = self.X[self.contents,:]
        y - np array - all label data at the node   - y = self.Y[self.contents,:]
        j - integer - split feature
        t - float - split threshold (data is split <= t, > t)
        returns:
            partition - list of lists, indicies in left and right partition 
            entropy - entropy impurity
        """
        x = self.X[self.contents,:]
        y = self.Y[self.contents]
        
        # get indices of left partition
        left = [i[0] for i in (np.argwhere(x[:,j] <= t))]
        right = [i[0] for i in (np.argwhere(x[:,j] > t))]
        
        if left:
            n_left = len(left)
            p_left = np.bincount(y[left])  /n_left
            left_impurity = -p_left[0]*np.log(p_left[0]) + -p_left[1]*np.log(p_left[1])
        else:
            left_impurity = 0
            n_left = 0
        if right:
            n_right = len(right)
            p_right = np.bincount(y[right])  /n_right
            right_impurity = -p_right[0]*np.log(p_right[0]) + -p_right[1]*np.log(p_right[1])
        else:
            n_right = 0
            right_impurity = 0      
        
        # calculate impurity = -p * log(p) for each class, for each class     
        entropy = left_impurity * n_left/len(y) + right_impurity * n_right/len(y)
        return entropy, [left, right]        
    
    def compute_optimal_split():
        """
        Finds optimal split for all data at node
        
        """
        pass
        
    def fit():
        pass
    def predict():
        pass
    def plot():
        pass
    def prune():
        pass


X = np.random.rand(100,10)
Y = np.random.randint(0,2,100)    
tree = TreeNode(X,Y,criterion  = 'gini')
tree = TreeNode(X,Y,criterion  = 'information_gain')