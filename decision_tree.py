import numpy as np
from sklearn.datasets import load_iris
data=load_iris()
x=data.data
y=data.target
x=(x-x.mean(axis=0))/x.std(axis=0)
'''def gini_impurity(y):
    values,count=np.unique(y,return_counts=True)
    p = count / np.sum(count)
    gini = 1 - np.sum(p**2)
    return gini,p
    # return the gini impurity
print(gini_impurity(np.array([1, 1, 1, 0, 0])))'''
def best_split(x, y):
    # X is a 2D numpy array (features), y is labels
    # loop over each feature
    # loop over each unique value in that feature as threshold
    # split y into left (<=threshold) and right (>threshold)
    # calculate weighted gini of that split
    # track the best feature and threshold (lowest weighted gini)
    # return best_feature_index, best_threshold
    #weighted_gini = (len(left)/len(y)) * gini(left) + (len(right)/len(y)) * gini(right)
    a=0
    b=float("inf")
    c=0
    for i in range(x.shape[1]):
        f=np.unique(x[:,i])
        for j in f:
            left=x[:,i]<=j
            right=~left
            y_left=y[left]
            y_right=y[right]
            val1,cot1=np.unique(y_left,return_counts=True)
            val2,cot2=np.unique(y_right,return_counts=True)
            p_left=cot1/np.sum(cot1)
            p_right=cot2/np.sum(cot2)
            gini_left=1-np.sum(p_left**2)
            gini_right=1-np.sum(p_right**2)
            weighted_gini = (len(y_left)/len(y))*gini_left+(len(y_right)/len(y))*gini_right
            gini=1-np.sum(gini_left**2)-np.sum(gini_right**2)
            if weighted_gini<b:
                b=weighted_gini
                a=i
                c=j
    return a,c
def build_tree(X, y, depth=0, max_depth=5):
    # 1. stopping conditions:
    #    - all samples same class (pure node)
    #    - reached max_depth
    # 2. find best split
    # 3. split X and y into left and right
    # 4. recurse on left and right
    # 5. return a node (use a dict: {"feature": ..., "threshold": ..., "left": ..., "right": ...})
    # stopping condition 1: pure node
    if len(np.unique(y)) == 1:
        return {"leaf": True, "prediction": y[0]}
    
    # stopping condition 2: max depth
    if depth == max_depth:
        values, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "prediction":values[np.argmax(counts)] }  # most common class
    
    # find best split
    feat, thresh = best_split(X, y)
    
    # split data
    left_mask = X[:, feat] <= thresh
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[~left_mask], y[~left_mask]
    
    # recurse
    return {
        "feature": feat,
        "threshold": thresh,
        "left": build_tree(X_left,y_left,depth+1,max_depth),
        "right": build_tree(X_right,y_right,depth+1,max_depth)
    }
def predict_one(tree, X):
    # for each sample in X, traverse the tree
    # at each node: if sample's feature value <= threshold, go left, else go right
    # at a leaf: return the prediction
    if "leaf" in tree:
        return tree["prediction"]
    if X[tree["feature"]]<=tree["threshold"]:
        return predict_one(tree["left"],X)
    else:
        return predict_one(tree["right"],X)
def predict(tree, X):
    return np.array([predict_one(tree,x) for x in X])  
#print(best_split(x,y))
#print(build_tree(x,y))
print(np.mean(predict(build_tree(x,y),x)==y))