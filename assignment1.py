# Main file for assignment 1

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder 

def montage(W):
	""" Display the image for each label in W """
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[5*i+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

def computeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = computeCost(X, Y, W, b, lamda);

	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = computeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = computeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def computeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = computeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = computeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = computeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = computeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def loadBatch(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def generateData(batch):
    #d*n
    X = np.array(batch.get(b'data')).transpose()
    #k*n
    Y = np.zeros((k,np.size(X,1)))
    #n*1
    y = np.array(batch.get(b'labels'))
    #representing with one-hot
    for index, value in enumerate(y):
        Y[value][index] = 1
    Y = np.array(Y)
    return X, Y, y

def preProcessData(X):
    #ON BONUS, replace train_X with X
    mean_X = np.mean(train_X, axis = 1).reshape(-1,1)
    std_X = np.std(train_X, axis =1).reshape(-1,1)

    X = X - mean_X
    X = X/std_X
    return X

def initParameters():
    mu = 0
    sigma = 0.01
    #k*d
    W = np.random.normal(mu,sigma,(k,d))
    #k*1
    b = np.random.normal(mu,sigma,(k,1))
    return W,b

def evaluateClassifier(X,W,b):
    #calculates s = WX + b and returns probabibiltes of every datapoint belonging to every class
    # k X d * d X n = k X n
    WX = np.matmul(W,X)
    s = WX + b
    p = softmax(s)
    return p

def computeCost(X,Y,W,b,la):
	'''Computes loss for model and returns mean loss + reg.term = cost'''
    p = evaluateClassifier(X, W, b)
    l_cross = -np.log(np.dot(Y.T, p))
    l_cross = np.diag(l_cross).reshape(-1,1)
    l_cross = np.sum(l_cross) / X.shape[1]
	l2 = self.la * (np.sum(W['W1'],2) + np.sum(W['W2'],2) )

    return l_cross_mean + l2



		p, _ = self.evaluateClassifier(X, W, b)
        l_cross = -np.log(np.matmul(Y.transpose(), p))
        l_cross = np.diag(l_cross).reshape(-1,1)
        l_cross = np.sum(l_cross) / X.shape[1]

        l2 = self.lamb * (np.sum(np.power(W['W1'], 2)) + np.sum(np.power(W['W2'], 2)))
        cost = l_cross + l2


def computeAccuracy(p, y):
    score=0
    columns = p.shape[1]

    for col in range(columns):
        max_prob_i = np.argmax(p[:,col])
        if max_prob_i == y[col]:
            score+=1
    return score/columns

def computeGradients(X,Y,p,W,la):
    n = X.shape[1]
    g = -(Y-p)
    grad_W = ( (np.dot(g,np.transpose(X))) / n ) + 2*la*W
    grad_b = (np.dot(g,np.ones(n))) / n
    return grad_W, grad_b

def miniBatchGD(X, Y, GD_params, W, b, la):

    batch_size = GD_params[0]
    eta = GD_params[1]
    n_epochs = GD_params[2]
    n_batch = X.shape[1] // batch_size

    train_cost = np.zeros((n_epochs,1))
    val_cost = np.zeros((n_epochs,1))

    # no of epocs = itterations through ALL datapoints
    for i in range(n_epochs):
        print(i)

        #randomize X before each new epoch
        randomizer = np.arange(X.shape[1])
        np.random.shuffle(randomizer)
        X = X.T[randomizer].T
        Y = Y.T[randomizer].T

        #num of batches in every epoch
        for j in range(n_batch):
            X_batch = X[: , batch_size*j:batch_size*(j+1)-1]
            Y_batch = Y[: , batch_size*j:batch_size*(j+1)-1]
            p = evaluateClassifier(X_batch, W, b)
            grad_W, grad_b = computeGradients(X_batch, Y_batch, p, W, la)

            W = W - eta*grad_W
            b = b - eta*grad_b.reshape(-1,1)

        train_cost[i] = computeCost(X, Y, W, b, la)
        val_cost[i] = computeCost(val_X, val_Y, W, b, la)

    #Caluclate p for final W on test batch. And then Accuracy
    test_p = evaluateClassifier(test_X, W, b)
    test_acc = computeAccuracy(test_p, test_y)
    print(test_acc)

    plotCostEpoch(train_cost, val_cost, test_acc)

    return W, b

def plotCostEpoch(train_cost, valid_cost, test_acc):
    x = np.linspace(1, train_cost.shape[0], train_cost.shape[0])
    plt.plot(x, train_cost, 'b', label='Training loss')
    plt.plot(x, valid_cost, 'r', label='Validation loss')
    plt.legend()
    plt.xlabel('Epoch', fontsize = 12)
    plt.ylabel('Loss', fontsize = 12)

    plt.title('Test Accuracy: ' + str('{0:.5g}'.format(test_acc*100)) + '%', fontsize = 12 )

    name = str(la) + "-" + str(GD_params[1]) + ".png"
    #plt.savefig('ResultPics/' + name)
    print("Test Acc: ", test_acc)
    #plt.show()

def testGradient(X, Y):
	#choose 20 datapoints for testing
	X_batch = X[: , 0:20]
	Y_batch = Y[: , 0:20]
	p = evaluateClassifier(X_batch, W_init, b_init)
	eps = 1e-16

	#compute gradients, analytical and numerical(fast/slow)
	grad_W_ana, grad_b_ana = computeGradients(X_batch, Y_batch, p, W_init, la)
	grad_W_num, grad_b_num = computeGradsNumSlow(X_batch, Y_batch, p, W_init, b_init, la, 1e-6)
	grad_b_ana = grad_b_ana.reshape(-1,1)

	#absolute difference between ana and num
	abs_error_W = np.abs(grad_W_ana - grad_W_num)
	abs_error_b = np.abs(grad_b_ana - grad_b_num)

	#calculations for relative erro
	(i,j) = np.shape(grad_W_ana)
	rel_error_W = np.zeros((i,j))
	rel_error_b = np.zeros(i)
	for i in range(i):
		sum = np.abs(grad_b_ana[i]) + np.abs(grad_b_num[i])
		rel_error_b[i] = abs_error_b[i] / max(eps,sum)

		for j in range(j):
			sum = np.abs(grad_W_ana[i,j]) + np.abs(grad_W_num[i,j])
			rel_error_W[i,j] = abs_error_W[i,j] / max(eps,sum)


	#Prints of max-error, absolute and relatie
	print(np.amax(rel_error_W))
	print(np.amax(rel_error_b))
	print(np.amax(abs_error_W))
	print(np.amax(abs_error_b))


#--------------MAIN-------------
np.random.seed(0)

#no of classes
k = 10
#lambda for regularization
la = 1

A = loadBatch("Datasets/cifar-10-batches-py/data_batch_1")
B = loadBatch("Datasets/cifar-10-batches-py/data_batch_2")

TEST = loadBatch("Datasets/cifar-10-batches-py/test_batch")

train_X, train_Y, train_y = generateData(A)
val_X, val_Y, val_y = generateData(B)
test_X, test_Y, test_y = generateData(TEST)

#d = np.size(train_X,0)

val_X = preProcessData(val_X)
test_X = preProcessData(test_X)
train_X = preProcessData(train_X)

W_init, b_init = initParameters()

# batch_size , eta, n_epochs
GD_params = [100, 0.001, 40]

W_Star, b_Star = miniBatchGD(train_X, train_Y, GD_params, W_init, b_init, la)

testGradient(train_X, train_Y)
montage(W_Star)
