import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from scipy.io import loadmat
from sklearn.utils import shuffle

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N,10))
    for i in range(N):
       ind[i,y[i]] = 1
    return ind

def error_rate(p,t):
    return np.mean(p != t)

def flatten(X):
    N = X.shape[-1]
    flat = np.zeros((N,3072))
    for i in range(N):
        flat[i] = X[:,:,:,i].reshape(3072)
    return flat

def main():
    train = loadmat('../data/housenumbers/train_32x32.mat')
    test = loadmat('../data/housenumbers/test_32x32.mat')

    Xtrain = flatten(train['X'].astype(np.float32) / 255)
    Ytrain = train['y'].flatten() - 1
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest = flatten(test['X'].astype(np.float32) / 255)
    Ytest = test['y'].flatten() - 1
    Ytest_ind = y2indicator(Ytest)

    max_iter = 20
    print_period =10
    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N / batch_sz

    M1 = 1000
    M2 = 500
    K = 10
    W1_init = np.random.randn(D, M1) / np.sqrt(D + M1)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2 + K)
    b3_init = np.zeros(K)

    # define variables and expressions
    X = tf.placeholder(tf.float32, shape=(None,D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')

    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Yish = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Yish, T))

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    # we'll use this to calculate the error rate
    predict_op = tf.arg_max(Yish, 1)

    t0 = datetime.now()

    LL = []
    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ybatch})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(LL)
    plt.show()

if __name__ == '__main__':
    main()






























