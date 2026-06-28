from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
data = load_breast_cancer()
x=data.data
y=data.target
x=(x-x.mean(axis=0))/x.std(axis=0)
def sigmoid(z):
    return 1/(1+np.exp(-z))
w=np.zeros(x.shape[1])
b=0
l=0.1
loss=[]
for i in range(1000):
    z=x@w+b
    e=-np.mean(y*np.log(sigmoid(z)+(1-y)*np.log(1-sigmoid(z))))
    loss.append(e)
    dw=x.T@(sigmoid(z)-y)/len(y)
    db=np.mean(sigmoid(z)-y)
    w=w-l*dw
    b=b-l*db
prediction=(sigmoid(x@w+b)>=0.5).astype(int)
accuracy=np.mean(prediction==y)
print(f"accuracy:{accuracy:.4f}")
plt.plot(e)
plt.show()