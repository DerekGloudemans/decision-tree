"""
Derek Gloudemans
Machine Learning Assignment 1
"""

#-----------------------------------------------------------------------------#
#--------------------------Import Packages------------------------------------#
import numpy as np
import copy
import csv
from TreeNode import TreeNode


#-----------------------Function Definitions----------------------------------#

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
    Y = data[:,0].astype(int)
    X = data[:,1:]
    return X,Y


#----------------------------Main Code Body-----------------------------------#
    
X_train,Y_train = load_cancer_CSV("cancer_datasets_v2/training_1.csv")
X_val,Y_val = load_cancer_CSV("cancer_datasets_v2/validation_1.csv")
X_test,Y_test = load_cancer_CSV("cancer_datasets_v2/testing_1.csv")
print("Data loaded.")


# fit a tree to each dataset
tree = TreeNode(X_train,Y_train,criterion = 'entropy')
tree.fit(depth_limit = 25)
print("Decision tree classifier fit to data.")

# score tree
_,train_result = tree.predict_score(X_train,Y_train)
_,test_result = tree.predict_score(X_test,Y_test)
print("Results obtained for test dataset.")

# get node counts
non_leaf_count = tree.get_node_count(include_leaves = False)
leaf_count = tree.get_node_count(include_leaves = True) -non_leaf_count

# plot tree
#tree.plot(legend = ["Benign", "Malignant"])

# summarize results in table
print("------------------Results for Dataset 1 -----------------")
print("|  Metric       |  Training Dataset  |  Testing Dataset  |")
print("|  Total Nodes  |                   {}                   |".format(leaf_count+non_leaf_count))
print("|  Leaf Nodes   |                   {}                   |".format(non_leaf_count))
print("|  Accuracy     |  {:.03f}             |  {:.03f}            |".format(train_result['acc'],test_result['acc']))
print("---------------------------------------------------------")
# report training and testing accuracy, number of nodes, number of leaf nodes for each dataset

# prune trees iteratively - at each step, report classification accuracy and F1 score, on training, validation and testing sets
results_list = []
num_nodes = tree.get_node_count()
while num_nodes > 0:
    # get results for all data
    _,train_result = tree.predict_score(X_train,Y_train)
    _,validation_result = tree.predict_score(X_val,Y_val)
    _,test_result = tree.predict_score(X_test,Y_test)
    
    # store results
    results_list.append([num_nodes,train_result,validation_result,test_result])
    
    # prune a node from the tree
    tree,_ = tree.prune_single_node_greedy(X_val,Y_val)
    try:
        num_nodes = tree.get_node_count()
    except:
        break
    
# generate ROC curves and precision-recall curves for each pruned tree
