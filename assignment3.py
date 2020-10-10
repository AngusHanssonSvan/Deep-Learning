# Main file for assignment 3

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

'''FUNCTIONS FOR PRE-GENERATING DATA & NETWORK LAYERS'''
def oneBatch():
    train = loadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    val = loadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    test = loadBatch("Datasets/cifar-10-batches-py/test_batch")

    data = {
    'X_train':train[0], 'Y_train':train[1], 'y_train':train[2],
    'X_val':val[0], 'Y_val':val[1], 'y_val':val[2],
    'X_test':test[0], 'Y_test':test[1], 'y_test':test[2],
    }
    return data

def allBatches(final_test):
    A = loadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    B = loadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    C = loadBatch("Datasets/cifar-10-batches-py/data_batch_3")
    D = loadBatch("Datasets/cifar-10-batches-py/data_batch_4")
    E = loadBatch("Datasets/cifar-10-batches-py/data_batch_5")
    test = loadBatch("Datasets/cifar-10-batches-py/test_batch")

    X,Y,y = matrixFusion(A,B,C,D,E)

    if(final_test == False):
        #n=45000 for training set, n=5000 for validation, n=10000 for test
        data ={
        'X_train':X[: , :45000], 'Y_train':Y[: , :45000], 'y_train':y[:45000, :],
        'X_val':X[: , 45000:], 'Y_val':Y[: , 45000:], 'y_val':y[45000: , :],
        'X_test':test[0], 'Y_test':test[1], 'y_test':test[2]
        }

    if(final_test):
        #FINAL TEST: n=49000 for training set, n=1000 for validation, n=10000 for test
        data ={
        'X_train':X[: , :49000], 'Y_train':Y[: , :49000], 'y_train':y[:49000, :],
        'X_val':X[: , 49000:], 'Y_val':Y[: , 49000:], 'y_val':y[49000: , :],
        'X_test':test[0], 'Y_test':test[1], 'y_test':test[2]
        }
    return data

def loadBatch(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X,Y,y = dataFromBatch(dict)
    data = [X,Y,y]
    return data

def dataFromBatch(batch):
    X = np.array(batch.get(b'data')).transpose()
    Y = batch.get(b'labels')
    y = np.array(Y).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    onehotencoder = OneHotEncoder(categories='auto')
    Y = onehotencoder.fit_transform(Y).toarray()
    Y = Y.T
    return X, Y, y

def matrixFusion(A,B,C,D,E):
    '''Function for concatenating all batches for ex. 4'''
    #all datapoints
    superX = np.concatenate((A[0], B[0], C[0], D[0], E[0]), axis=1)
    superY = np.concatenate((A[1], B[1], C[1], D[1], E[1]), axis=1)
    supery = np.concatenate((A[2], B[2], C[2], D[2], E[2]), axis=0)

    return superX, superY, supery

def createLayers(no_layers):

    if no_layers == 9:
        layers = [(50,3072), (30,50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)]
    else:
        layers = []
        for l in range(no_layers):
            if l == 0:
                layers.append((50,3072))
            elif l == (no_layers-1):
                layers.append((10,50))
            else:
                layers.append((50,50))

    return layers
'''END of data generating functions'''

class kLayerNetwork():
    '''Class for k-LayerNetwork'''
    def __init__(self, data, layers, GD_params,
                 la, BN):
        '''Constructor.
           INPUT -  data: training, validation and test data
                    layers: dimensions for parameters in each layer
                    GD_params: parameters for gradient descent.

           0.000762368337623976 (original lambda)
           0.0028044274267758785 (pats)
           X: d*n, Y: k*n, y: n
           W1: m*d, W2: k*m
           b1: m*1, b2: k*1
        '''

        for key, value in data.items():
            setattr(self, key, value)

        for key, value in GD_params.items():
            setattr(self, key, value)


        self.d = self.X_train.shape[0]
        self.la = la
        self.layers = layers
        self.no_layers = len(self.layers) - 1
        self.k = len(self.layers)
        self.BN = BN
        self.sigma_fixed = 0

        # Storage for parameters
        self.W = []
        self.b = []

        if self.BN:
            self.gamma = []
            self.beta = []
            self.mean_avg = []
            self.var_avg = []
            self.parameters = {"W": self.W, "b": self.b,
                           "gamma": self.gamma,"beta": self.beta}

        else:
            self.parameters = {"W": self.W, "b": self.b}

        self.initParameters()
        self.standardize()

    def initParameters(self):
        '''Initializing parameters W and b for each layer
            If batchNormalize gamma and beta is initialized
        '''

        for i, value in enumerate(self.layers):
            #self.sigma_fixed = 1e-1
            #self.sigma_fixed = 1e-3
            self.sigma_fixed = 1e-4
            # self.sigma_fixed = np.sqrt(value[1])
            self.W.append(np.random.normal(0,self.sigma_fixed, value))
        #b's, gamna's, beta's
        for i, value in enumerate(self.layers):
            self.b.append(np.random.normal(0,0,(value[0],1)))
            if self.BN:
                self.gamma.append(np.ones((value[0],1)))
                self.beta.append(np.zeros((value[0],1)))
                self.mean_avg.append(np.zeros((value[0],1)))
                self.var_avg.append(np.zeros((value[0],1)))
        if self.BN:
            self.gamma.pop()
            self.beta.pop()
            self.mean_avg.pop()
            self.var_avg.pop()

    def standardize(self):
        '''Standardize datapoint(X) w.r.t training-batch'''
        mean_X = np.mean(self.X_train, axis = 1).reshape(-1,1)
        std_X = np.std(self.X_train, axis =1).reshape(-1,1)

        self.X_train = self.X_train-mean_X
        self.X_val = self.X_val-mean_X
        self.X_test = self.X_test-mean_X

        self.X_train = self.X_train/std_X
        self.X_val = self.X_val/std_X
        self.X_test = self.X_test/std_X

    def miniBatchGD(self, plot = True):
        '''Minibath Gradient Descent for altering parameters
           Cycling learning rates is implemented. One cycle is 2*n_s (up to eta_max and down to eta_min)
        '''

        eta_min = 1e-5
        eta_max = 1e-1
        n_s = int(2 * 45000 / self.batch_size)

        cycles = 2
        t=0

        #number of mini-batches
        n_batch = self.X_train.shape[1] // self.batch_size

        #Total Number of batches
        t_max = 2*n_s*cycles  #l in assignment

        train_cost = []
        val_cost = []
        train_loss = []
        val_loss = []
        train_acc = np.zeros((t_max//450+1,1))
        val_acc = []

        #For each batch á 100 datapoints 0 - 9000
        for j in range(t_max):
            if t<=n_s:
                self.eta = eta_min + (t/n_s)*(eta_max-eta_min)
            elif t >= n_s:
                self.eta = eta_max - ((t-n_s)/n_s)*(eta_max-eta_min)

            #Shuffle after each epoc!
            if j%450 == 0:
                randomizer = np.arange(self.X_train.shape[1])
                np.random.shuffle(randomizer)
                self.X_train = self.X_train.T[randomizer].T
                self.Y_train = self.Y_train.T[randomizer].T

            X_batch = self.X_train[: , self.batch_size*(j%n_batch):self.batch_size*((j%n_batch)+1)]
            Y_batch = self.Y_train[: , self.batch_size*(j%n_batch):self.batch_size*((j%n_batch)+1)]

            grad = self.computeGradients(X_batch, Y_batch)

            #Updating parameters
            for l in range(self.k):
                self.W[l] = self.W[l] - self.eta * grad['W'][l]
                self.b[l] = self.b[l] - self.eta * grad['b'][l].reshape(-1,1)
                if self.BN and l < (self.k-1):
                    self.gamma[l] = (self.gamma[l] - self.eta * grad['gamma'][l]).reshape(-1,1)
                    self.beta[l] = (self.beta[l] - self.eta * grad['beta'][l]).reshape(-1,1)
            t = (t+1) % (2*n_s)

            #Every epoch loss is computed.
            if j%450 == 0 and plot==True:
                print(j)
                tr_cost, tr_loss = self.computeCost(self.X_train, self.Y_train)

                #train_cost.append(tr_cost)
                train_loss.append(tr_loss)

                v_cost, v_loss = self.computeCost(self.X_val, self.Y_val)
                #val_cost.append(v_cost)
                val_loss.append(v_loss)
                print(tr_loss)
                print(v_loss)
                print("\n")
                #test_p1,_ = self.evaluateNetwork(self.X_train)
                #train_acc[j//450] = self.computeAccuracy(test_p1, self.y_train)

                #for lambdaSearch
                # test_p2,_,_,_,_,_ = self.evaluateNetwork(self.X_val,training_mode=False)
                # val_acc.append(self.computeAccuracy(test_p2, self.y_val))


        #Caluclate p for final W on test batch. And then computed accuracy for  test batch
        if self.BN:
            test_p,_,_,_,_,_ = self.evaluateNetwork(self.X_test, training_mode = False)
        else:
            test_p,_ = self.evaluateNetwork(self.X_test)
        test_acc = self.computeAccuracy(test_p, self.y_test)
    
        self.plotResults(t_max=t_max, train_loss=train_loss, val_loss=val_loss, test_acc=test_acc)

    def computeGradients(self, X , Y):
        '''Analytically comuptes gradients for network, w.r.t different parameters
           Forward(evaluateNetwork()) & Backwards Propagation

           X = Current minibatch
           Y = Corresponnding labels
        '''
        n = X.shape[1]
        ones = np.ones((n,1))

        grad = {'W':[], 'b':[], 'gamma':[], 'beta':[]}
        for key in self.parameters:
            for value in self.parameters[key]:
                grad[key].append(np.zeros_like(value))

        if self.BN:

            #fwd
            p, H, S, S_hat, means, vars = self.evaluateNetwork(X)

            #bwd
            g = -(Y-p)

            grad['W'][-1] = (1/n) * np.matmul(g ,H[-1].T) + 2 * self.la * self.W[-1]
            grad['b'][-1] = ((1/n) * np.matmul(g,ones)).reshape(-1,1)

            g = np.matmul(self.W[-1].T, g)
            H[-1][H[-1] <= 0] = 0
            g = np.multiply(g, H[-1]>0)


            for l in range(self.k-2 , -1,-1):
                grad['gamma'][l] = ( (1/n) * np.matmul(np.multiply(g,S_hat[l]), ones)).reshape(-1,1)
                grad['beta'][l] = ((1/n) * np.matmul(g,ones).reshape(-1,1))

                g = np.multiply(g, np.matmul(self.gamma[l],ones.T))
                g = self.batchNormBackPass(g, S[l],means[l],vars[l])

                grad['W'][l] = (1/n) * np.matmul(g, H[l].T) + 2 * self.la * self.W[l]
                grad['b'][l] = ((1/n) * np.matmul(g, ones)).reshape(-1,1)

                if l>0:
                    g = np.matmul(self.W[l].T, g)
                    H[l][H[l] <= 0] = 0
                    g = np.multiply(g, H[l]>0)
            return grad
        else:

            #fwd
            p, H = self.evaluateNetwork(X)
            #bwd
            g = -(Y-p)

            for l in range(self.k-1, 0,-1):
                grad['W'][l] = (1/n) * np.matmul(g, H[l-1].T) + 2 * self.la * self.W[l]
                grad['b'][l] = ((1/n) * np.matmul(g, ones)).reshape(-1,1)
                g = np.matmul(self.W[l].T, g)
                H[l-1][H[l-1] <= 0] = 0
                g = np.multiply(g, H[l-1]>0)
            grad['W'][0] = (1/n) * np.matmul(g, X.T) + 2 * self.la * self.W[0]
            grad['b'][0] = ((1/n) * np.matmul(g, ones)).reshape(-1,1)
        return grad

    def evaluateNetwork(self, X, training_mode = True, W=None, b=None):
        '''Forward propagation for network
           Computes probabibiltes for every class for every datapoint
           AND intermediary activations for each hidden layer.
           p: k*n
        '''
        n = X.shape[1]
        if W is None and b is None:
            W=self.W
            b=self.b

        if self.BN:
            H = []
            S = []
            S_hat = []
            means = []
            vars = []
            h = np.copy(X)
            H.append(h)

            #For each layer
            for l in range(self.k):
                s = np.matmul(W[l], h) + b[l]

                #If middle layer
                if l < (self.k-1):

                    #If in training mode, computing mean/var & exp. avg
                    if (training_mode) == True:
                        s_mean = np.mean(s, axis=1).reshape(-1,1)
                        s_var = (np.var(s, axis=1) * ((n-1)/n)).reshape(-1,1)

                        #computing exponentional moving average used when testing
                        self.mean_avg[l] = 0.9 * self.mean_avg[l] + (1-0.9)*s_mean
                        self.var_avg[l] = 0.9 * self.var_avg[l] + (1-0.9)*s_var

                    else:
                    #In test mode
                        s_mean = self.mean_avg[l]
                        s_var = self.var_avg[l]

                    #Normalizing score with above computed or pre-Computed mean/var
                    s_hat = self.batchNormalize(s,s_mean,s_var)

                    #Scale & shift
                    s_scale = np.multiply(self.gamma[l], s_hat) + self.beta[l]

                    #activation for next layer
                    h = np.maximum(0, s_scale)

                    #Storing intermediary results
                    H.append(h)
                    S.append(s)
                    S_hat.append(s_hat)
                    means.append(s_mean)
                    vars.append(s_var)

                    #ReLu for next layers score computation.


                else:
                    #Softmax activated
                    p = self.softmax(s)

            return p, H, S, S_hat, means, vars

        else:
            H = []
            h = np.copy(X)
            for l in range(self.k):
                s = np.matmul(W[l],h) + b[l]
                #If middle layer
                if l < self.k-1:
                    h = np.maximum(0,s)
                    H.append(h)
                else:
                    p = self.softmax(s)

            return p, H #where H is intermediary activations h(l)

    def batchNormalize(self, s, s_mean, s_var):
        eps = np.finfo(np.float64).eps
        return np.power(np.diag(s_var) + eps , -0.5) * (s - s_mean)

    def batchNormBackPass(self, g, S, mean, var):
        n = g.shape[1]
        ones = np.ones((n,1))
        eps = np.finfo(np.float64).eps

        sigma1 = (np.power(var + eps, -0.5).T).reshape(-1,1)
        sigma2 = (np.power(var + eps, -1.5).T).reshape(-1,1)

        G1 = np.multiply(g, np.matmul(sigma1, ones.T))
        G2 = np.multiply(g, np.matmul(sigma2, ones.T))

        D = S - np.matmul(mean,ones.T)
        c = np.matmul(np.multiply(G2,D),ones)
        g = G1 - (1/n) * np.matmul(G1,ones) * ones.T - (1/n) * np.multiply(D, np.matmul(c,ones.T)) # MAYBE ALTER
        return g

    def computeCost(self, X, Y, W=None, b=None):
        '''Computes cost and loss with loss function and regularization term l2'''
        if W is None and b is None:
            W=self.W
            b=self.b

        if self.BN:
            p,_,_,_,_,_ = self.evaluateNetwork(X, False)
        else:
            p,_ = self.evaluateNetwork(X)

        loss = 0
        for i in range(Y.shape[1]):
           loss += -np.log(np.matmul(Y[:, i], p[:, i]))
        loss = loss / X.shape[1]

        l2 = 0
        for l in range(self.k):
            l2+= np.sum(np.power(self.W[l],2))
        l2 = l2*self.la

        cost = loss + l2
        return cost, loss

    def computeAccuracy(self, p, y):
        '''Computes accuracy for evaluated network with probabilites p and true label y'''
        score=0
        columns = p.shape[1]

        for col in range(columns):
            max_prob_i = np.argmax(p[:,col])
            if max_prob_i == y[col]:
                score+=1
        return score/columns

    def plotResults(self, t_max = None,
                    train_cost = [], val_cost = [],
                    train_loss =[], val_loss = [],
                    train_acc = [], val_acc = [],
                    test_acc = None,
                    ifcost = False, ifloss = True, ifacc = False):
        '''Plot method'''
        if(ifcost):
            x = np.linspace(0, train_cost.shape[0], train_cost.shape[0])
            plt.plot(x, train_cost, 'b', label='Training cost')
            plt.plot(x, val_cost, 'r', label='Validation cost')
            plt.legend()
            plt.xlabel('t', fontsize = 12)
            plt.ylabel('Cost', fontsize = 12)
            #plt.title('Test Accuracy: ' + str('{0:.5g}'.format(test_acc*100)) + '%', fontsize = 12 )
            name = "costFINAL"  + ".png"
            plt.savefig('ResultPics/' + name)
            #print("Test Acc: ", test_acc)
            plt.show()

        if(ifloss):
            x = np.linspace(0, t_max, len(train_loss))
            plt.plot(x, train_loss, 'b', label='Training loss')
            plt.plot(x, val_loss, 'r', label='Validation loss')
            plt.legend()
            plt.xlabel('t', fontsize = 12)
            plt.ylabel('Loss', fontsize = 12)
            name = "k" + str(self.k) + "BN" + str(self.BN) + "sigma" + str(self.sigma_fixed) +".png"
            #name = "sanity"
            plt.savefig('ResultPics/' + name)
            #print("Test Acc: ", test_acc)
            plt.show()

        if(ifacc):
            x = np.linspace(0, train_acc.shape[0]*100, train_acc.shape[0])
            plt.plot(x, train_acc, 'b', label='Training accuracy')
            plt.plot(x, val_acc, 'r', label='Validation accuracy')
            plt.legend()
            plt.xlabel('t', fontsize = 12)
            plt.ylabel('Accuracy', fontsize = 12)
            plt.title('Test Accuracy: ' + str('{0:.5g}'.format(test_acc*100)) + '%', fontsize = 12 )
            name = "accFINAL"  + ".png"
            #plt.savefig('ResultPics/' + name)
            print("Test Acc: ", test_acc)
            plt.show()

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def computeGradsNumSlow(self, X, Y, W, b, h):
        '''Numerically computed gradients for validating analytical gradient'''
        grad_W = {}
        grad_b = {}

        for j in range(1, len(b)+1):
            grad_b['b'+str(j)] = np.zeros(b[('b'+str(j))].shape)

            for i in range(len(b['b'+str(j)])):
                b['b'+str(j)][i] -= h
                c1,_ = self.computeCost(X, Y, W, b)

                b['b'+str(j)][i] += 2*h
                c2,_ = self.computeCost(X, Y, W, b)

                grad_b['b'+str(j)][i] = (c2-c1) / (2*h)
                #resetting b to original pos.
                b['b'+str(j)][i] -= h

        for j in range(1, len(W)+1):
            grad_W['W'+str(j)] = np.zeros(W[('W'+str(j))].shape)
            for i in np.ndindex(W['W'+str(j)].shape):

                W['W'+str(j)][i] -= h
                c1,_ = self.computeCost(X, Y, W, b)

                W['W'+str(j)][i] += 2*h
                c2,_ = self.computeCost(X, Y, W, b)

                grad_W['W'+str(j)][i] = (c2-c1) / (2*h)
                #resetting W to original pos.
                W['W'+str(j)][i] -= h

        return grad_W, grad_b

    def testGradient(self):
        '''Tests analytically computed gradient by calculating relative error to numerical gradient'''

        #modification of attributes before testing gradients num/ana
        self.X_train = self.X_train[:20, 1:2]
        self.Y_train = self.Y_train[:, 1:2]
        self.y_train = self.y_train[1:2, :]
        self.W['W1'] = self.W['W1'][:, :20]

        eps =1e-16

        grad_ana = self.computeGradients(self.X_train, self.Y_train)
        grad_W_num, grad_b_num = self.computeGradsNumSlow(self.X_train, self.Y_train, self.W, self.b, 1e-5)
        print('W1')
        # print(grad_ana['W1'])
        max_errors = {}
        ix = 1

        #for every layer, including b and W, computing relative error between numerical and analytical
        for key,value in grad_W_num.items():
            (i,j) = np.shape(value)
            b_errors = []
            W_errors = []
            #absolute error for CURRENT layer's b and W
            abs_b = np.abs(grad_ana['b'+str(ix)] - grad_b_num['b'+str(ix)])
            abs_W = np.abs(grad_ana['W'+str(ix)] - grad_W_num['W'+str(ix)])
            for i in range(i):
                sum = np.abs(grad_ana['b'+str(ix)][i]) + np.abs(grad_b_num['b'+str(ix)][i])
                b_errors.append(abs_b[i] / max(eps,sum))
                for j in range(j):
                    sum = np.abs(grad_ana['W'+str(ix)][i,j]) + np.abs(grad_W_num['W'+str(ix)][i,j])
                    W_errors.append(abs_W[i,j] / max(eps,sum))
            max_errors['b'+str(ix)] = np.amax(b_errors)
            max_errors['W'+str(ix)] = np.amax(W_errors)
            ix+=1

        # for key ,value in max_errors.items():
        #     print(key , ": " , value)

    def lambdaSearch(self):
        '''Method for finding best lambda'''
        l_min = -4
        l_max = -3
        itterations = 10

        best_accuracies = np.zeros(itterations)
        lambdas = np.zeros(itterations)

        #Random lambda for every itteration. Best accuracy
        for i in range(itterations):
            print("Itteration: ", i)
            l = l_min + (l_max - l_min)*np.random.uniform(0,1)
            self.la = np.power(10,l)

            #returns all calculated accuracies and chooses the highest(often the latest, but amax for safety)
            val_acc = self.miniBatchGD()
            best_acc = np.amax(val_acc)
            print(best_acc)
            print(self.la)

            lambdas[i] = self.la
            best_accuracies[i] = best_acc

        #sort from HI to LOW and save to text file
        sortedIndices = np.flip(np.argsort(best_accuracies))

        lambdas = lambdas[sortedIndices]
        best_accuracies = best_accuracies[sortedIndices]

        save = np.array([lambdas, best_accuracies])
        save = save.T
        np.savetxt('A3 - AFTERTEST.txt', save , delimiter = '   -   ')

    def sanityTest(self):
        '''Method for ensure overfitting on training data, thus correct weights'''
        self.la = 0
        train_loss = []
        val_loss = []
        max_epoch = 200
        batch_size = 10
        eta = 0.01

        self.X_train = self.X_train[:, 0:100]
        self.Y_train = self.Y_train[:, 0:100]
        n_batch = self.X_train.shape[1] // batch_size

        #For each batch á 100 datapoints 0 - 9000
        for epoch in range(max_epoch):
            randomizer = np.arange(self.X_train.shape[1])
            np.random.shuffle(randomizer)
            self.X_train = self.X_train.T[randomizer].T
            self.Y_train = self.Y_train.T[randomizer].T
            print("Epoch: " , epoch)
            for j in range(n_batch):
                X_batch = self.X_train[: , batch_size*j:batch_size*(j+1)-1]
                Y_batch = self.Y_train[: , batch_size*j:batch_size*(j+1)-1]
                grad = self.computeGradients(X_batch, Y_batch)

                #Updating parameters
                for l in range(self.k):
                    self.W[l] = self.W[l] - eta * grad['W'][l]
                    self.b[l] = self.b[l] - eta * grad['b'][l].reshape(-1,1)
                    if self.BN and l < (self.k-1):
                        self.gamma[l] = (self.gamma[l] - eta * grad['gamma'][l]).reshape(-1,1)
                        self.beta[l] = (self.beta[l] - eta * grad['beta'][l]).reshape(-1,1)

            #Every epoch loss is computed.
            tr_cost, tr_loss = self.computeCost(self.X_train, self.Y_train)
            v_cost, v_loss = self.computeCost(self.X_val, self.Y_val)
            train_loss.append(tr_loss)
            val_loss.append(v_loss)
            print(tr_loss)

        #test_p,_,_,_,_,_ = self.evaluateNetwork(self.X_train, training_mode = False)
        #test_acc = self.computeAccuracy(test_p, self.y_train
        self.plotResults(t_max=max_epoch, train_loss=train_loss, val_loss=val_loss, test_acc=test_acc)


if __name__ == '__main__':
    #No BN:
        #52.85% with l=3 and c=2
        #49,28% with l=9 anc c=2
    #WITH BN:
        #53.06 with l=3 and c=2
        #50,72 with l=9 and c=2
    np.random.seed(0)

    GD_params = {
    'batch_size':100, 'eta':0.005, 'n_epochs':200
    }
    layers = createLayers(3)
    data = allBatches(False)
    la = 0.005
    TLN = kLayerNetwork(data, layers, GD_params,la, False)
    TLN.miniBatchGD()
    #TLN.lambdaSearch()

    print("DONE")
