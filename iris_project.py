from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
data = load_iris()
y=data.target
x=data.data
x=(x-x.mean(axis=0))/x.std(axis=0)
def sigmoid(z):
    return 1/(1+np.exp(z))
w=np.zeros(x.shape[1])
b=0
l=0.01
loss=[]
for i in range(1000):
    yi=x@w +b
    e=np.mean((y-yi)**2)
    loss.append(e)
    dm=-2*(x.T @ (y-yi))/len(y)
    db=np.mean((y-yi))*-2
    w=w-l*dm
    b=b-l*db
    ss_rec = np.sum((y - yi)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_rec/ss_tot)
def knn_predict(X_train, y_train, x_test, k=3):
    distances = np.sqrt(np.sum((X_train - x_test)**2, axis=1))
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]
    return np.bincount(k_labels.astype(int)).argmax()
predictions_knn = np.array([knn_predict(x, y, x[i], k=3) for i in range(len(x))])
accuracy_knn = np.mean(predictions_knn == y)
print(f"KNN Accuracy: {accuracy_knn:.4f}")
print(f"w:{w} and b:{b}")
print(f"Linear Regression R²: {r2:.4f}")
plt.plot(loss)
plt.title("Linear Regression Loss Curve - Iris")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.show()
