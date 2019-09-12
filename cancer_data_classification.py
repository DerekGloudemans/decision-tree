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
import matplotlib.pyplot as plt

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


def condense_results_list(results_list):
    """
    Condenses list of list of dictionaries into 3 numpy arrays of results
    where one row corresponds to:
        num_nodes, acc, f1, precision, recall, tpr, fpr
    for one tree in the pruning process
    """
    # condense data into an array for train, test and val
    # num_nodes, acc, f1, precision, recall, tpr, fpr
    train_res = np.zeros([len(results_list),7])
    val_res = np.zeros([len(results_list),7])
    test_res = np.zeros([len(results_list),7])
    
    for i,res in enumerate(results_list):
        train_res[i,0] = res[0]
        val_res[i,0] = res[0]
        test_res[i,0] = res[0]
        
        train_res[i,1] = res[1]['acc']
        val_res[i,1] = res[2]['acc']
        test_res[i,1] = res[3]['acc']
        
        train_res[i,2] = res[1]['f1']
        val_res[i,2] = res[2]['f1']
        test_res[i,2] = res[3]['f1']
        
        train_res[i,3] = res[3]['precision']
        val_res[i,3] = res[3]['precision']
        test_res[i,3] = res[3]['precision']
        
        train_res[i,4] = res[1]['recall']
        val_res[i,4] = res[2]['recall']
        test_res[i,4] = res[3]['recall']
        
        train_res[i,5] = res[1]['tpr']
        val_res[i,5] = res[2]['tpr']
        test_res[i,5] = res[3]['tpr']
        
        train_res[i,6] = res[3]['fpr']
        val_res[i,6] = res[3]['fpr']
        test_res[i,6] = res[3]['fpr']
        
    return train_res,val_res,test_res

def plot_prune_results(results_list, dataset_id = 1,impurity = "gini"):
    """
    Generates three plots for a tree pruning test:
        - Accuracy and F1 versus number of nodes (train, val, and test)
        - Precision recall curve (train, val, and test)
        - Receiver operating characteristic curve (train, val and test)
    """
    
    # condense results
    train_res,val_res,test_res = condense_results_list(results_list)
    
    # generate figure
    with plt.style.context('ggplot'):
        color1 = (129/255, 161/255, 214/255)
        color2 = (207/255, 137/255, 105/255)
        color3 = (173/255, 136/255, 179/255)
        fig, axarr = plt.subplots(1, 3, figsize = (20,5))
        fig.suptitle("Results for {} impurity decision tree on dataset {}".format(impurity,dataset_id),fontsize = 18)
        plt.subplots_adjust(left=None, bottom=None, right=None, top= None, wspace=0.25)
        
        # subplot 1 - accuracy versus number of nodes
        axarr[0].plot(train_res[:,0],train_res[:,1],'-',color = color1)
        axarr[0].plot(train_res[:,0],train_res[:,2],'--',color = color1)
        axarr[0].plot(val_res[:,0],val_res[:,1],'-',color = color2)
        axarr[0].plot(val_res[:,0],val_res[:,2],'--',color = color2)
        axarr[0].plot(test_res[:,0],test_res[:,1],'-',color = color3)
        axarr[0].plot(test_res[:,0],test_res[:,2],'--',color = color3)
        
        axarr[0].set_xlim([0, max(train_res[:,0])])
        axarr[0].set_ylim([0.5, 1])
        axarr[0].set_ylabel('Metric',fontsize = 14)
        axarr[0].set_xlabel('Number of nodes',fontsize = 14)
        axarr[0].set_title('Performance vs. number of nodes',fontsize = 16)
        legend = ["train acc.","train f1", "val. acc.", "val. f1.", "test acc.", "test f1"]
        axarr[0].legend(legend)

    
        # subplot 2 - precision versus recall
        axarr[1].plot(train_res[:,4],train_res[:,3],'-',color = color1)
        axarr[1].plot(val_res[:,4],val_res[:,3],'-',color = color2)
        axarr[1].plot(test_res[:,4],test_res[:,3],'-',color = color3)
        
        axarr[1].set_xlim([0, 1])
        axarr[1].set_ylim([0, 1])
        axarr[1].set_ylabel('Precision',fontsize = 14)
        axarr[1].set_xlabel('Recall',fontsize = 14)
        axarr[1].set_title('Precision vs. recall',fontsize = 16)
        legend = ["train PR.", "val PR", "test PR"]
        axarr[1].legend(legend)
        
        # subplot 3 - ROC curve
        axarr[2].plot(train_res[:,6],train_res[:,5],'-',color = color1)
        axarr[2].plot(val_res[:,6],val_res[:,5],'-',color = color2)
        axarr[2].plot(test_res[:,6],test_res[:,5],'-',color = color3)
        
        axarr[2].set_xlim([0, 1])
        axarr[2].set_ylim([0, 1])
        axarr[2].set_ylabel('True positive rate',fontsize = 14)
        axarr[2].set_xlabel('False positive rate',fontsize = 14)
        axarr[2].set_title('ROC curve',fontsize = 16)
        legend = ["train ROC.", "val ROC", "test ROC"]
        axarr[2].legend(legend)



#----------------------------Main Code Body-----------------------------------#
    
X_train,Y_train = load_cancer_CSV("cancer_datasets_v2/training_2.csv")
X_val,Y_val = load_cancer_CSV("cancer_datasets_v2/validation_2.csv")
X_test,Y_test = load_cancer_CSV("cancer_datasets_v2/testing_2.csv")
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

feature_labels = ["avg. radius","avg texture","avg permtr","avg area","avg smthness","avg cmpctness","avg concvty","avg num cncv prtns", "avg sym","avg frct dim",
 "radius stdev","texture stdev","permtr stdev","area stdev","smthness stdev","cmpctness stdev","concvty stdev","num cncv prtns stdev", "sym stdev","frct dim stdev",
 "max radius","max texture","max permtr","max area","max smthness","max cmpctness","max concvty","max num cncv prtns", "max sym","max frct dim"]
#plot tree
tree.plot(legend = ["Benign", "Malignant"],feature_labels = feature_labels)

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
    
    tree.plot(legend = ["Benign", "Malignant"])
    
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
plot_prune_results(results_list,dataset_id = 1, impurity = "gini")