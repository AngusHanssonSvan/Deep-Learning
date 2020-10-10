# Main file for assignment 2

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

'''Functions for data gathering'''
def loadBatch(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X,Y,y = generateData(dict)
    data = [X,Y,y]
    return data

def generateData(batch):
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
'''End of data generating functions'''

class TwoLayerNetwork():
    '''Class for TwoLayerNetwork. Used for exercise 1-3 & 4. Initparameters sent from MAIN'''
    def __init__(self, data, GD_params,
                W1=None, W2=None,
                b1=None, b2=None,
                W=None, b=None,
                k=None, m=50, la = 0.0028044274267758785,
                test = None
                ):
        '''Constructor. Initializing Data, W's and b's
           Data contains train, val, test stored in dict
           0.000762368337623976 (original lambda)
           X: d*n, Y: k*n, y: n
           W1: m*d, W2: k*m
           b1: m*1, b2: k*1
        '''

        for key, value in data.items():
            setattr(self, key, value)

        for key, value in GD_params.items():
            setattr(self, key, value)

        self.k = self.Y_train.shape[0]
        self.d = self.X_train.shape[0]
        self.m = m
        self.la = la

        self.W1 = np.random.normal(0,1/np.sqrt(self.d),(self.m,self.d))
        self.W2 = np.random.normal(0,1/np.sqrt(self.m),(self.k,self.m))
        self.b1 = np.random.normal(0,0,(self.m,1))
        self.b2 = np.random.normal(0,0,(self.k,1))

        self.W = {'W1':self.W1, "W2":self.W2}
        self.b = {'b1':self.b1, "b2":self.b2}
        self.standardize()


        self.test = self.W['W1']
        print(self.W['W1'])
        self.test = self.test*100
        print(self.W['W1'])



        #self.computeGradients(self.X_train, self.Y_train)
        #self.computeGradients(self.X_train, self.Y_train)
    def standardize(self):
        '''Standardize datapoints w.r.t training-batch'''
        mean_X = np.mean(self.X_train, axis = 1).reshape(-1,1)
        std_X = np.std(self.X_train, axis =1).reshape(-1,1)

        self.X_train = self.X_train-mean_X
        self.X_val = self.X_val-mean_X
        self.X_test = self.X_test-mean_X

        self.X_train = self.X_train/std_X
        self.X_val = self.X_val/std_X
        self.X_test = self.X_test/std_X

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def evaluateNetwork(self, X, W=None, b=None):
        '''Forward propagation for network
           Computes probabibiltes for every class for every datapoints
           p: k*n
        '''
        if W is None and b is None:
            W=self.W
            b=self.b

        s1 = np.matmul(W['W1'], X) + b['b1']
        h = np.maximum(0,s1)
        s2 = np.matmul(W['W2'], h) + b['b2']
        p = self.softmax(s2)
        return p, h

    def computeAccuracy(self, p, y):
        '''Computes accuracy for evaluated network with probabilites p and true label y'''
        score=0
        columns = p.shape[1]

        for col in range(columns):
            max_prob_i = np.argmax(p[:,col])
            if max_prob_i == y[col]:
                score+=1
        return score/columns

    def computeCost(self, X, Y, W=None, b=None):
        '''Computes cost and loss with loss function and regularization term l2'''
        if W is None and b is None:
            W=self.W
            b=self.b
        p,_ = self.evaluateNetwork(X)

        loss = 0
        for i in range(Y.shape[1]):
           loss += -np.log(np.matmul(Y[:, i], p[:, i]))
        loss = loss / X.shape[1]


        l2 = self.la * (np.sum(np.power(W['W1'],2)) + np.sum(np.power(W['W2'],2)))
        cost = loss + l2
        return cost, loss

    def computeGradients(self, X, Y):
        '''Analytically comuptes gradients for network, w.r.t different parameters
           Forward(evaluateNetwork()) & Backwards Propagation
        '''
        n = X.shape[1]

        p_batch, h_batch = self.evaluateNetwork(X)

        g_batch = -(Y-p_batch)

        grad_W2 = (1/n) * np.matmul(g_batch,h_batch.T) + 2 * self.la * self.W['W2']
        grad_b2 = ((1/n) * np.matmul(g_batch,np.ones(n))).reshape(-1,1)

        g_batch = np.matmul(self.W['W2'].T,g_batch)
        h_batch[h_batch <= 0] = 0
        g_batch = np.multiply(g_batch, h_batch>0)

        grad_W1 = (1/n) * np.matmul(g_batch,X.T) + 2 * self.la * self.W['W1']
        grad_b1 = ((1/n) * np.matmul(g_batch,np.ones(n))).reshape(-1,1)
        return {'W1':grad_W1, 'W2':grad_W2}, {'b1':grad_b1,'b2':grad_b2}

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

    def miniBatchGD(self):
        '''Minibath Gradient Descent for altering parameters
           Cycling learning rates is implemented. One cycle is 2*n_s (up to eta_max and down to eta_min)
        '''
        #given from assignment sheet
        eta_min = 1e-5
        eta_max = 1e-1
        n_s = int(2 * np.floor(self.X_train.shape[1]/self.batch_size))
        cycles = 3
        t=0

        #number of mini-batches
        n_batch = self.X_train.shape[1] // self.batch_size

        t_max = 2*n_s*cycles  #l in assignment

        train_cost = np.zeros((t_max//100+1,1))
        val_cost = np.zeros((t_max//100+1,1))
        train_loss = np.zeros((t_max//100+1,1))
        val_loss = np.zeros((t_max//100+1,1))
        train_acc = np.zeros((t_max//100+1,1))
        val_acc = np.zeros((t_max//100+1,1))

        for j in range(t_max):

            if t<=n_s:
                self.eta = eta_min + (t/n_s)*(eta_max-eta_min)
            elif t >= n_s:
                self.eta = eta_max - ((t-n_s)/n_s)*(eta_max-eta_min)


            X_batch = self.X_train[: , self.batch_size*(j%n_batch):self.batch_size*((j%n_batch)+1)]
            Y_batch = self.Y_train[: , self.batch_size*(j%n_batch):self.batch_size*((j%n_batch)+1)]
            grad_W, grad_b = self.computeGradients(X_batch, Y_batch)
            #print(grad_W['W1'])
            self.W['W1'] = self.W['W1'] - self.eta*grad_W['W1']
            self.b['b1'] = (self.b['b1'] - self.eta*grad_b['b1']).reshape(-1,1)
            self.W['W2'] = self.W['W2'] - self.eta*grad_W['W2']
            self.b['b2'] = (self.b['b2'] - self.eta*grad_b['b2']).reshape(-1,1)

            t = (t+1) % (2*n_s)

            #every 100th t, we compute loss
            # if j%100 == 0:
            #     #print(self.W['W2'])
            #     #calulates cost, loss, acc every 100th step. For plots.
            #
            #     train_cost[j//100], train_loss[j//100] = self.computeCost(self.X_train, self.Y_train)
            #     val_cost[j//100], val_loss[j//100] = self.computeCost(self.X_val, self.Y_val)
            #
            #     test_p1,_ = self.evaluateNetwork(self.X_train)
            #     train_acc[j//100] = self.computeAccuracy(test_p1, self.y_train)
            #
            #     test_p2,_ = self.evaluateNetwork(self.X_val)
            #     val_acc[j//100] = self.computeAccuracy(test_p2, self.y_val)

        #Caluclate p for final W on test batch. And then computed accuracy for  test batch
        test_p,_ = self.evaluateNetwork(self.X_test)
        #print(self.W)
        test_acc = self.computeAccuracy(test_p, self.y_test)
        print(test_acc)
        #self.plotResults(train_cost, val_cost, train_loss, val_loss, train_acc, val_acc, test_acc)

    def plotResults(self,
                    train_cost, val_cost,
                    train_loss, val_loss,
                    train_acc, val_acc,
                    test_acc,
                    ifcost = False, ifloss = False, ifacc = True):
        '''Plot method'''
        if(ifcost):
            x = np.linspace(0, train_cost.shape[0]*100, train_cost.shape[0])
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
            x = np.linspace(0, train_loss.shape[0]*100, train_loss.shape[0])
            plt.plot(x, train_loss, 'b', label='Training loss')
            plt.plot(x, val_loss, 'r', label='Validation loss')
            plt.legend()
            plt.xlabel('t', fontsize = 12)
            plt.ylabel('Loss', fontsize = 12)
            #plt.title('Test Accuracy: ' + str('{0:.5g}'.format(test_acc*100)) + '%', fontsize = 12 )
            name = "lossFINAL"  + ".png"
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

    def testGradient(self):
        #modification of attributes before testing gradients num/ana
        self.X_train = self.X_train[:20, 1:2]
        self.Y_train = self.Y_train[:, 1:2]
        self.y_train = self.y_train[1:2, :]
        self.W['W1'] = self.W['W1'][:, :20]

        eps =1e-16


        grad_W_num, grad_b_num = self.computeGradsNumSlow(self.X_train, self.Y_train, self.W, self.b, 1e-5)
        grad_W_ana, grad_b_ana = self.computeGradients(self.X_train, self.Y_train)

        W1_ana = grad_W_ana['W1']
        W2_ana = grad_W_ana['W2']
        b1_ana = grad_b_ana['b1']
        b2_ana = grad_b_ana['b2']

        W1_num = grad_W_num['W1']
        W2_num = grad_W_num['W2']
        b1_num = grad_b_num['b1']
        b2_num = grad_b_num['b2']

        abs_e_W1 = np.abs(W1_ana - W1_num)
        abs_e_W2 = np.abs(W2_ana - W2_num)
        abs_e_b1 = np.abs(b1_ana - b1_num)
        abs_e_b2 = np.abs(b2_ana - b2_num)

        (i,j) = np.shape(W1_ana)
        (k,l) = np.shape(W2_ana)

        rel_e_W1 = np.zeros((i,j))
        rel_e_W2 = np.zeros((k,l))
        rel_e_b1 = np.zeros(i)
        rel_e_b2 = np.zeros(k)

        #relative erros for b1,W1
        for i in range(i):
            sum = np.abs(b1_ana[i]) + np.abs(b1_num[i])
            rel_e_b1[i] = abs_e_b1[i] / max(eps,sum)
            for j in range(j):
                sum = np.abs(W1_ana[i,j]) + np.abs(W1_num[i,j])
                rel_e_W1[i,j] = abs_e_W1[i,j] / max(eps,sum)

        #relative erros for b2,W2
        for k in range(k):
            sum = np.abs(b2_ana[k]) + np.abs(b2_num[k])
            rel_e_b2[k] = abs_e_b2[k] / max(eps,sum)
            for l in range(l):
                sum = np.abs(W2_ana[k,l]) + np.abs(W2_num[k,l])
                rel_e_W2[k,l] = abs_e_W2[k,l] / max(eps,sum)

        #prints values for relative errors.
        print(np.amax(rel_e_W1))
        print(np.amax(rel_e_W2))
        print(np.amax(rel_e_b1))
        print(np.amax(rel_e_b2))

    def lambdaSearch(self):
        '''Method for finding best lambda'''
        l_min = -6
        l_max = -3
        itterations = 50

        best_accuracies = np.zeros(itterations)
        lambdas = np.zeros(itterations)

        #Random lambda for every itteration. Best accuracy
        for i in range(itterations):
            l = l_min + (l_max - l_min)*np.random.uniform(0,1)
            self.la = np.power(10,l)

            #returns all calculated accuracies and chooses the highest(often the latest, but amax for safety)
            val_acc = self.miniBatchGD()
            best_acc = np.amax(val_acc)

            lambdas[i] = self.la
            best_accuracies[i] = best_acc

            # print("ITTERATION: ", i)
            # print("LAMBDA: " , lambdas)
            # print("ACC: " , best_accuracies)

        #sort from HI to LOW and save to text file
        sortedIndices = np.flip(np.argsort(best_accuracies))

        lambdas = lambdas[sortedIndices]
        best_accuracies = best_accuracies[sortedIndices]

        save = np.array([lambdas, best_accuracies])
        save = save.T
        np.savetxt('AFTERTEST.txt', save , delimiter = '   -   ')

if __name__ == '__main__':
    np.random.seed(0)
    #---------------------EXERCISE 1-3----------------------------
    train = loadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    val = loadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    test = loadBatch("Datasets/cifar-10-batches-py/test_batch")

    data1 = {
    'X_train':train[0], 'Y_train':train[1], 'y_train':train[2],
    'X_val':val[0], 'Y_val':val[1], 'y_val':val[2],
    'X_test':test[0], 'Y_test':test[1], 'y_test':test[2],
    }

    GD_params1 = {
    'batch_size':100, 'eta':0.001, 'n_epochs':3
    }
    #TLN1.miniBatchGD()
    #TLN1.testGradient()
    #---------------------END OF EXERCISE 1-3----------------------------


    #---------------------EXERCISE 4-------------------------------------
    A = loadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    B = loadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    C = loadBatch("Datasets/cifar-10-batches-py/data_batch_3")
    D = loadBatch("Datasets/cifar-10-batches-py/data_batch_4")
    E = loadBatch("Datasets/cifar-10-batches-py/data_batch_5")

    X,Y,y = matrixFusion(A,B,C,D,E)

    #n=45000 for training set, n=5000 for validation, n=10000 for test
    data2 ={
    'X_train':X[: , :45000], 'Y_train':Y[: , :45000], 'y_train':y[:45000, :],
    'X_val':X[: , 45000:], 'Y_val':Y[: , 45000:], 'y_val':y[45000: , :],
    'X_test':test[0], 'Y_test':test[1], 'y_test':test[2]
    }

    #FINAL TEST: n=49000 for training set, n=1000 for validation, n=10000 for test
    data3 ={
    'X_train':X[: , :49000], 'Y_train':Y[: , :49000], 'y_train':y[:49000, :],
    'X_val':X[: , 49000:], 'Y_val':Y[: , 49000:], 'y_val':y[49000: , :],
    'X_test':test[0], 'Y_test':test[1], 'y_test':test[2]
    }

    GD_params2 = {
    'batch_size':100, 'eta':0.005, 'n_epochs':200
    }
    TLN = TwoLayerNetwork(data1,GD_params2)

    #TLN.miniBatchGD()
