"""
Derek Gloudemans
Machine Learning Assignment 1
DecisionTree.py - implements TreeNode class
"""

#-----------------------------------------------------------------------------#
#--------------------------Import Packages------------------------------------#
import numpy as np
import copy
import cv2
import time

#---------------------TreeNode Class Definition-------------------------------#


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

    ##### functions for getting info about the tree
    def __len__(self):
        """
        returns - int-  length of data used to fit tree
        """
        return len(self.Y)
    
    def get_depth(self):
        """
        returns - int - maximum depth of TreeNode in Tree
        """
        if self.children:
            return max(self.children[0].get_depth(),
                       self.children[1].get_depth())
        else:
            return self.depth
        
    def get_node_count(self,include_leaves = True):
        """
        include_leaves - bool - specifies whether leaf nodes are included in count
        returns:
            int - number non-leaf nodes in tree, including root
            int - number of leaf nodes in tree    
        """
        if self.children == []:
            if include_leaves:
                return 1
            else:
                return 0
        else:
            return 1 + self.children[0].get_node_count(include_leaves = include_leaves) + self.children[1].get_node_count(include_leaves = include_leaves)
    
    ##### functions for getting impurity values
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
    
    ##### functions for splitting and fitting
    def compute_optimal_split(self,show = False):
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
                
                # get impurity val
                val,[l,r] = self.criterion(j,x[i,j])
                if show:
                    print("i:{} j:{} t:{}  val:{} l:{} r:{}".format(i,j,x[i,j],val,l,r))
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
        
        # compute optimal split on data if there is still class impurity
        if self.depth < depth_limit and self.impurity > 0.01:
            j,t,val,[left,right] = self.compute_optimal_split()
            self.split = (j,t)
            
            # tree is still able to be further subdivided
            if len(left) > 0 and len(right) > 0:
                #create two new nodes on partitions of data and add to self.children
                left_node = TreeNode(self.X[left,:],self.Y[left],depth = self.depth + 1, criterion = self.criterion_name ) 
                right_node = TreeNode(self.X[right,:],self.Y[right],depth = self.depth + 1, criterion = self.criterion_name ) 
                
                # recursively fit each child node
                left_node.fit(depth_limit)
                right_node.fit(depth_limit)
                
                self.children = [left_node, right_node]

    ##### functions for predicting and scoring            
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
            
    def predict_score(self,X,Y):
        """
        """
        eps = 0.00000001
        Y_pred = self.predict(X)
        diff = Y_pred - Y
        n = len(Y)
        fp =  len(np.where(diff == 1)[0])# Y_pred = 1, Y_ = 0
        fn = len(np.where(diff == -1)[0]) # Y_pred = 0, Y = 1
        tn = len(np.where(Y == 0)[0]) - fp 
        tp = len(np.where(Y == 1)[0]) - fn
        assert fp+fn+tn+tp == n, "Error!!"
        acc = 1- (sum(np.abs(diff))/n)
        precision = tp/ (tp + fp +eps) # correct predictions over all predictions
        recall = tp/(tp + fn +eps) # predicted 1s over correct 1s
        f1 = (2*precision*recall)/(precision + recall +eps)
        result_dict ={
                "fp":fp,
                "fn":fn,
                "tn":tn,
                "tp":tp,
                "acc":acc,
                "precision":precision,
                "recall":recall,
                "f1":f1,
                "tpr": tp/(tp + fn +eps),
                "fpr": fp/(fp + tn +eps),
                }
                
        return acc,result_dict
    
    ##### functions for pruning 
    def get_all_node_paths(self,current_path = [], include_leaves = False):
        """ 
        returns a list of lists, each list corresponding to the path through 
        children to reach a node that is not already a leaf node
        """
        all_node_paths = []
        
        # check for improper pruning of only one child node
        if len(self.children) == 1:
            raise Exception
        
        # if node has children, recursively call get_all_node _paths
        if self.children:
            path_left = current_path.copy()
            path_left.append(0)
            all_node_paths_left = self.children[0].get_all_node_paths(path_left,include_leaves = include_leaves)
            
            path_right = current_path.copy()
            path_right.append(1)
            all_node_paths_right = self.children[1].get_all_node_paths(path_right,include_leaves = include_leaves)
            
            all_node_paths = [current_path] + all_node_paths_left + all_node_paths_right
       
        # otherwise, return current path (only if include_leaves, since this is a leaf node)
        elif include_leaves:
            all_node_paths.append(current_path)
            
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
            acc,_ = pruned.predict_score(X_val,Y_val)
            if acc > best_acc:
                best_acc = acc
                best = copy.deepcopy(pruned)
        
        return best,best_acc
       
    ##### functions for displaying tree
    def plot(self,x_scale = 3000,y_scale = 900,legend = ["class 0","class 1"],feature_labels = None):
        """
        Plots each node as a rectangle containing text with splitting criteria,
        impurity, and number of examples with each label in the node. 
        Color varies according to the class proportion
        """
        depth = self.get_depth()
        paths = self.get_all_node_paths(include_leaves = True)
        
        # for storing relevant info on nodes
        node_dict_list = {}
        node_coords = np.zeros([len(paths),2])
        
        for p, path in enumerate(paths):
            
            # calculate x and y offset
            y_off = y_scale/(depth+1) *len(path)+ 50
#            if len(path) > 0 and path[-1] == 0:
#                y_off = y_off + 50
            x_off = x_scale/2
            for i in range(len(path)):
                offset = x_scale*(2**(-i-2))
                if path[i] == 0:
                    x_off = x_off - offset
                else:
                    x_off = x_off + offset
                
            #get node corresponding to path
            node = self
            # copy path to use as key to dictionary
            path_copy = copy.deepcopy(path)
            # get node corresponding to path
            while path_copy != []:
                node = node.children[path_copy.pop(0)]
              
            # save stats from node
            node_dict = {}
            (j,t) = node.split
            node_dict['j'] = j
            node_dict['t'] = t
            class0 = len(np.where(node.Y == 0)[0])
            class1 = len(np.where(node.Y == 1)[0])
            total = class0+class1
            node_dict["impurity"] = node.impurity
            node_dict['color'] = (int(class0/total*255),150,int(class1/total*255))
            node_dict['class0'] = class0
            node_dict['class1'] = class1
            node_dict['idx'] = p
            node_dict['impurity_type'] = node.criterion_name[:4]
            node_dict_list[str(path)] = node_dict
            
            # store node_offsets
            node_coords[p,0] = x_off
            node_coords[p,1] = y_off
            
        
        # dimensions for plotting        
        node_width = 150
        node_height = 50
        
        # get rid of node overlap
        node_coords = self._jiggle(node_coords,node_width)
        max_x = np.max(node_coords[:,0])
        max_y = np.max(node_coords[:,1])
        
        # create blank image
        im = 255 * np.ones(shape=[int(max_y+node_height), int(max_x + node_width), 3], dtype=np.uint8)
        
        #plot lines
        self._plot_lines(im,node_coords,node_dict_list,paths,node_height)
        
        # plot nodes
        for path in paths:
            nd = node_dict_list[str(path)]
            idx = nd['idx']
            x_off = node_coords[idx,0]
            y_off = node_coords[idx,1]
            
            self._place_node(im,nd,x_off,y_off,node_width,node_height,feature_labels)
        
        # plot legend
        self._plot_legend(im,legend)
        
        # show and save tree
        cv2.imshow("Tree", im)
        cv2.imwrite("temp.jpg",im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _place_node(self,im,node_dict, x_offset, y_offset,width,height,feature_labels = None):
        """
        Places a rectangle with info corresponding to a single node
        """
        # get split criterion
        color = node_dict['color']
    
        # place border
        bw = 1
        border1 = (int(-bw + x_offset-width/2), int(-bw+y_offset-height/2))
        border2 = (int(bw + x_offset+width/2), int(bw+y_offset+height/2))
        im = cv2.rectangle(im,border1,border2, (50,50,50), bw*2)
        
        # place rectangle
        corner1 = (int(x_offset-width/2), int(y_offset-height/2))
        corner2 = (int(x_offset+width/2), int(y_offset+height/2))
        im = cv2.rectangle(im,corner1,corner2, color, -1)
        
        # plot text
        font_scale = 1.0
        font = cv2.FONT_HERSHEY_PLAIN
        to1 = (int(x_offset-width/2),int(y_offset-height/2+15))
        to2 = (int(x_offset-width/2),int(y_offset-height/2+30))
        to3 = (int(x_offset-width/2),int(y_offset-height/2+45))
        
        if feature_labels:
            text1 = "{}<={:.3f}".format(feature_labels[node_dict['j']],node_dict['t'])
        else:
            text1 = "x_{}<={:.3f}".format(node_dict['j'],node_dict['t'])
        if node_dict['t'] == -np.inf:
            text1 = ""
        text2 = "{}: {:.3f}".format(node_dict['impurity_type'],node_dict['impurity'])
        text3 = "cls: [{},{}]".format(node_dict['class0'],node_dict['class1'])
        
        cv2.putText(im,text1,to1,font,fontScale=font_scale/2, color=(0,0,0), thickness=1)
        cv2.putText(im,text2,to2,font,fontScale=font_scale, color=(0,0,0), thickness=1)
        cv2.putText(im,text3,to3,font,fontScale=font_scale, color=(0,0,0), thickness=1)
        
    def _plot_lines(self,im,node_coords, node_dict_list, paths,height = 0):
        """
        Plots line from each node to its parent node, if any
        im - cv2 image of tree
        node_coords - np array with xy offsets for each node
        node_dict_list - dict of dicts, as defined in TreeNode.plot()
        paths - list of lists - each list is the binary path to one node
        node_height - int
        node_width -int
        """
        for path in paths:
            if len(path) > 0: #not the root node
                node_dict = node_dict_list[str(path)]
                idx = node_dict['idx']
                point1 = (int(node_coords[idx,0]),
                          int(node_coords[idx,1]-height/2))
                
                parent_node_dict = node_dict_list[str(path[:-1])]
                par_idx = parent_node_dict['idx']
                point2 = (int(node_coords[par_idx,0]),
                          int(node_coords[par_idx,1]+height/2))
                
                cv2.line(im,point1,point2,(50,50,50),2)

    def _plot_legend(self,im,legend,buffer = 10):
        """
        Plots legend on image
        """  
        font_scale = 1.0
        font = cv2.FONT_HERSHEY_PLAIN
        # label 0
        im = cv2.rectangle(im,(buffer,buffer),(buffer+150,buffer + 30),(255,150,0),-1)
        im = cv2.rectangle(im,(buffer-1,buffer-1),(buffer+151,buffer + 30),(50,50,50,),2)
        cv2.putText(im,legend[0],(buffer+2,buffer+15),font,fontScale=font_scale, color=(0,0,0), thickness=1)
        # label 1
        im = cv2.rectangle(im,(buffer,buffer+30),(buffer+150,buffer + 60),(0,150,255),-1)
        im = cv2.rectangle(im,(buffer-1,buffer+30),(buffer+151,buffer + 61),(50,50,50,),2)
        cv2.putText(im,legend[1],(buffer+2,buffer+45),font,fontScale=font_scale, color=(0,0,0), thickness=1)
        
    def _jiggle(self,node_coords,node_width, buffer = 10):
        """
        Moves nodes slightly horizontally so they don't overlap
        node_coords - np array with xy offsets for each node
        node_width - int
        buffer - int - specifies spacing between nodes
        returns:
            node_coords - np array with non-overlapping offsets for each node
        """  
        # find all depths (y values) 
        depths = np.unique(node_coords[:,1])
        # for each depth
        for depth in depths:
            # find all nodes at that depth
            idxs = np.where(node_coords[:,1] == depth)[0]
            
    
        
            # find center coordinate
            center = np.average(node_coords[idxs,0])
        
            # generate an array with new x_vals that don't intersect
            new_x = np.asarray(range(0,(node_width+buffer)*len(idxs),node_width+buffer))
            
            #shift to be centered
            center_new = np.average(new_x)
            new_x = new_x + center - center_new
            
            # assign to node_coords values at that depth
            node_coords[idxs,0] = new_x
        
        # shift so min value is on frame
        min_x = np.min(node_coords[:,0])
        node_coords[:,0] = node_coords[:,0] - min_x + node_width/2 + buffer
        return node_coords


#------------------------------Tester Code------------------------------------#
if __name__ == "__main__":
    X = np.random.rand(100,10)
    X_val = np.random.rand(100,10)
    Y = np.random.randint(0,2,100)    
    Y_val = np.random.randint(0,2,100)
    tree = TreeNode(X,Y,criterion  = 'gini')
#    tree = TreeNode(X,Y,criterion  = 'entropy')
    tree.fit()
    errors = tree.predict(X)
    acc,_ = tree.predict_score(X,Y)
    paths = tree.get_all_node_paths()
    tree.plot()
    while False:
        tree.plot(x_scale = 1000, y_scale = 1000,legend = ["Non-Cancerous","Cancerous"])
        tree, pruned_acc = tree.prune_single_node_greedy(X_val,Y_val)
        print(pruned_acc)
        time.sleep(0.1)
        break
    
    
