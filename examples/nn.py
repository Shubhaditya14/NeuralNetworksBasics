import numpy as np

X = np.array([ [0,0,1], [1,1,1], [1,0,1], [0,1,1]])
y = np.array([ [0,1,1,0]]).T
W0 = 2*np.random.random((3,4)) - 1
W1 = 2*np.random.random((4,1)) - 1
for j in range(6000):
    l1 = 1/(1+np.exp(-(np.dot(X,W0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,W1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(W1.T) * (l1 * (1-l1))
    W1 += l1.T.dot(l2_delta)
    W0 += X.T.dot(l1_delta)

