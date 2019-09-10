"""
Add header text here
"""

#-----------------------------------------------------------------------------#
#--------------------------Import Packages------------------------------------#
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import copy

class TreeNode():
    """
    A class for representing one node of a binary decision tree, and, recursively, 
    an entire decision tree. Contains functions for calculating impurity metrics,
    splitting nodes, pruning, and plotting. Throughout, data examples are assumed
    to be represented as numpy arrays:
        X - m x n array of examples (m examples, n features)
        Y - m x 1 array of labels (m examples, 1 label per example)
    """
    
    def __init__(self,X,Y,depth = 0,criterion = 'gini'):
        """
        Initialize a TreeNode
        """
        
        # store data within (or within children of) a node
        self.X = X
        self.Y = Y 
        
        # keeps track of split at node
        self.split = (0,-np.inf)
        
        # root node has depth 0 by convention
        self.depth = depth
        
        # contains zero or two TreeNodes
        self.children = []
        
        #"gini" or "entropy" impurity
        if criterion == 'gini':
            self.criterion =  self.gini
            self.criterion_name = "gini"
        else:
            self.criterion = self.entropy
            self.criterion_name = "entropy"
        
        # keep track of node's impurity (all grouped into 1 partition)
        if criterion == 'gini':
            self.impurity,_ = self.gini(0,np.inf)
        else:
            self.impurity,_= self.entropy(0,np.inf)

    def __len__(self):
        return len(self.Y)
    
    def get_depth(self):
        if self.children:
            return max(self.children[0].get_depth(),
                       self.children[1].get_depth())
        else:
            return self.depth
    
    def gini(self,j,t):
        """
        Calculate the gini impurity for a subset of tree data partitioned on 
        a given feature j and split value t
        x - np array - all feature data at the node
        y - np array - all label data at the node
        j - integer - split feature
        t - float - split threshold (data is split <= t, > t)
        returns:
            partition - list of lists, indicies in left and right partition 
            impurity - gini impurity
        """
        x = self.X
        y = self.Y
        
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
        
#    def information_gain(self,j,t):
#        """
#        Calculate the information_gain for a subset of tree data partitioned on 
#        a given feature j and split value t
#        x - np array - all feature data at the node - x = self.X[self.contents,:]
#        y - np array - all label data at the node   - y = self.Y[self.contents,:]
#        j - integer - split feature
#        t - float - split threshold (data is split <= t, > t)
#        returns:
#            partition - list of lists, indicies in left and right partition 
#            information_gain - change in entropy impurity
#        """
#        child_entropy, [left,right] = self.entropy(j,t)
#        
#        information_gain = self.entropy_val - child_entropy
#        return information_gain, [left,right]
        
    def entropy(self,j,t):
        """
        Used to get initial entropy value for tree
        Calculate the entropy for a subset of tree data partitioned on 
        a given feature j and split value t
        x - np array - all feature data at the node 
        y - np array - all label data at the node
        j - integer - split feature
        t - float - split threshold (data is split <= t, > t)
        returns:
            partition - list of lists, indicies in left and right partition 
            entropy - entropy impurity
        """
        x = self.X
        y = self.Y
        
        # get indices of left partition
        left = [i[0] for i in (np.argwhere(x[:,j] <= t))]
        right = [i[0] for i in (np.argwhere(x[:,j] > t))]
        
        if left:
            n_left = len(left)
            p_left = np.bincount(y[left])  /n_left + 0.000001 # to prevent divide by 0 errors
            if len(p_left) == 1:
                left_impurity = 0
            else:
                left_impurity = -p_left[0]*np.log(p_left[0]) + -p_left[1]*np.log(p_left[1])
        else:
            left_impurity = 0
            n_left = 0
        if right:
            n_right = len(right)
            p_right = np.bincount(y[right])  /n_right + 0.000001 # to prevent divide by 0 errors
            if len(p_right) == 1:
                right_impurity = 0
            else:
                right_impurity = -p_right[0]*np.log(p_right[0]) + -p_right[1]*np.log(p_right[1])
        else:
            n_right = 0
            right_impurity = 0      
        
        # calculate impurity = -p * log(p) for each class, for each class     
        entropy = left_impurity * n_left/len(y) + right_impurity * n_right/len(y)
        return entropy, [left, right]        
    
    def compute_optimal_split(self):
        """
        Finds optimal split for all data at node
        returns:
            j_opt - integer -  optimal feature to split on
            t_popt - float - optimal threshold to split on
            val - float - impurity value of optimal split
            left - list - indices of left partition
            right - list - indices of right partition
        """
        x = self.X
        
        j_opt = 0
        t_opt = -np.inf
        best_val = np.inf 
        # consider each feature
        for j in range(len(x[0])): 
            # consider each value as the threshold value
            for i in range(len(x)):
                
                # get val
                val = self.criterion(j,x[i,j])[0]
                if val < best_val:
                        best_val = val
                        j_opt = j
                        t_opt = x[i,j]
        val,[left,right] = self.criterion(j_opt,t_opt)
        return j_opt,t_opt, best_val, [left,right]
        
    def fit(self,depth_limit = 1000):
        """
        Recursively creates a decision tree in a breadth-first search manner 
        by computing the optimal split of each node until
        nodes are pure, features are identical, or depth limit is reached
        depth_limit - int >= 0 - specifies maximum depth of tree
        """
        
        # compute optimal split on data
        if self.depth < depth_limit:
            j,t,val,[left,right] = self.compute_optimal_split()
            self.split = (j,t)
            
            # tree is still able to be further subdivided
            if len(left) > 0 and len(right) > 0:
                #create two new nodes on partitions of data and add to self.children
                left_node = TreeNode(self.X[left,:],self.Y[left],depth = self.depth + 1, criterion = self.criterion_name ) 
                right_node = TreeNode(self.X[right,:],self.Y[right],depth = self.depth + 1, criterion = self.criterion_name ) 
                
                left_node.fit(depth_limit)
                right_node.fit(depth_limit)
                
                self.children = [left_node, right_node]

                
    def predict(self,X_new):
        """
        outputs predicted classes for each example in X_new
        X_new - m x n array of examples (m examples, n features)
        returns:
            Y_pred - m x 1 array of predicted labels
        """
        Y_pred = np.zeros(len(X_new))
        
        for i in range(0,len(X_new)):
            Y_pred[i] = self._walk(X_new[i])
        return Y_pred

    
    def _walk(self,x):
        """
        gets predicted class for one example
        x - 1 x n array for one example
        returns:
            y_pred - int - predicted label
        """
        if self.children:
            # pass example to correct child node
            j = self.split[0]
            t = self.split[1]
            if x[j] <= t:
                return self.children[0]._walk(x)
            else:
                return self.children[1]._walk(x)
        else:
            # get most common label in node
            return np.bincount(self.Y).argmax()
            
            
            
    def predict_score(self,X_new,Y_new):
        """
        """
        Y_pred = self.predict(X_new)
        
        n = np.size(X_new)
        diff = Y_pred - Y_new
        fp = 0
        fn = 0
        tn = 0
        tp = 0
        acc = 1- (sum(np.abs(diff))/n)
        precision = 0
        recall = 0
        f1 = 0
        return acc
    
    def plot():
        pass
     
    def get_all_node_paths(self,current_path = []):
        """ 
        returns a list of lists, each list corresponding to the path through 
        children to reach a node that is not already a leaf node
        """
        all_node_paths = []
        
        if self.children:
            path_left = current_path.copy()
            path_left.append(0)
            all_node_paths_left = self.children[0].get_all_node_paths(path_left)
            
            path_right = current_path.copy()
            path_right.append(1)
            all_node_paths_right = self.children[1].get_all_node_paths(path_right)
            
            all_node_paths = [current_path] + all_node_paths_left + all_node_paths_right
            
        return all_node_paths
    
    def prune_single_node_greedy(self,X_val,Y_val):
        """
        Removes all of a node's children and descendant nodes, condensing all examples into that node
        Node is selected greedily according to classification accuracy on validation
        data provided.
        """
        
        # get all paths to nodes
        paths = self.get_all_node_paths()
        best = None
        best_acc = 0
        
        
        
        # each path corresponds to one node in the tree
        for path in paths:
            pruned = copy.deepcopy(self)
            current_node = pruned
            while path != []:
                current_node = current_node.children[path.pop(0)]
                
                
            # remove children
            current_node.children = []
            
            # make predictions and get error
            acc = pruned.predict_score(X_val,Y_val)
            if acc > best_acc:
                best_acc = acc
                best = copy.deepcopy(pruned)
        
        return best,best_acc
        


X = np.random.rand(100,10)
Y = np.random.randint(0,2,100)    
tree = TreeNode(X,Y,criterion  = 'gini')
tree = TreeNode(X,Y,criterion  = 'entropy')
tree.fit()
errors = tree.predict(X)
acc = tree.predict_score(X,Y)
paths = tree.get_all_node_paths()
pruned_tree, pruned_acc = tree.prune_single_node_greedy(X,Y)
