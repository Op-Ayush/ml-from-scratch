import numpy as np
from sklearn.datasets import load_iris
from scipy.stats import mode
data=load_iris()
x=data.data
y=data.target
x=(x-x.mean(axis=0))/x.std(axis=0)
def best_split(x, y):
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
            if len(y_left) == 0 or len(y_right) == 0:
                continue
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
    if len(y) == 0:
        return {"leaf": True, "prediction": 0}
    
    if len(np.unique(y)) == 1:
        return {"leaf": True, "prediction": y[0]}
    
    if depth == max_depth:
        values, counts = np.unique(y, return_counts=True)
        return {"leaf": True, "prediction":values[np.argmax(counts)] }  # most common class
    
    feat, thresh = best_split(X, y)

    left_mask = X[:, feat] <= thresh
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[~left_mask], y[~left_mask]
    
    return {
        "feature": feat,
        "threshold": thresh,
        "left": build_tree(X_left,y_left,depth+1,max_depth),
        "right": build_tree(X_right,y_right,depth+1,max_depth)
    }
def predict_one(tree, X):
    if "leaf" in tree:
        return tree["prediction"]
    if X[tree["feature"]]<=tree["threshold"]:
        return predict_one(tree["left"],X)
    else:
        return predict_one(tree["right"],X)
def predict(tree, X):
    return np.array([predict_one(tree,x) for x in X])  
def random_forest(X, y, n_trees=10, max_depth=5, max_features=2):
    trees = []
    for _ in range(n_trees):
        # 1. bootstrap sample (random rows with replacement)
        # 2. random feature subset (random columns without replacement)
        # 3. build a tree on that sample
        # 4. store the tree
        indices = np.random.choice(len(X), size=len(X), replace=True)
        features = np.random.choice(X.shape[1], size=max_features, replace=False)
        x_sample=X[indices][:,features]
        y_sample=y[indices]
        A=build_tree(x_sample,y_sample)
        trees.append((A,features))
    return trees
def rf_predict(trees, X):
    # for each tree, get predictions using only its feature subset
    # majority vote across all trees for each sample
    all_preds = np.array([predict(tree, X[:, features]) for tree, features in trees])
    final=mode(all_preds,axis=0).mode
    return final
#print(best_split(x,y))
#print(build_tree(x,y))
#print(np.mean(predict(build_tree(x,y),x)==y))
#print(random_forest(x,y))
trees=random_forest(x,y)
pred=rf_predict(trees,x)
print(np.mean(pred==y))