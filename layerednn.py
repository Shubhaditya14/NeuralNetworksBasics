import numpy as np

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

X = np.array([ [0,0,1], [1,1,1], [1,0,1], [0,1,1]])
y = np.array([ [0,1,1,0]]).T

np.random.seed(1)

W0 = 2*np.random.random((3,1)) - 1

for i in range(6000):
    l0 = X
    l1 = sigmoid(np.dot(l0,W0))
    l1_error = y - l1  
    l1_delta = l1_error * sigmoid(l1,True)
    W0 += np.dot(l0.T,l1_delta)

print("Output after training")
print(l1)