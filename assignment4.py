#Main File for Assignment 4

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def prepareData():
    """Reads txt-file and preprocess data for class RNN"""
    book_fname = "Datasets/goblet_book.txt"
    f = open(book_fname, "r")
    book_data = f.read()
    book_chars = list(set(book_data))
    book_chars.sort()

    char_to_ind = {}
    ind_to_char = {}
    K = len(book_chars)

    for i, char in enumerate(book_chars):
        char_to_ind[char] = i
        ind_to_char[i] = char

    data = {'book_data':book_data, 'book_chars':book_chars,
            'char_to_ind':char_to_ind, 'ind_to_char':ind_to_char,
            'K':K
            }
    return data

class RNN():
    """Class for RNN, storing weight parameters and training model"""
    def __init__(self, test_gradient, data, m, eta=0.1, seq_length=25):

        for key, value in data.items():
            setattr(self, key, value)
        self.m = m
        self.eta = eta
        self.seq_length = seq_length
        self.test_gradient = test_gradient
        self.b, self.c, self.U, self.W, self.V = self.initParameters()
        self.params = {'b': self.b, 'c':self.c, 'U':self.U, 'W':self.W, 'V':self.V}

    def initParameters(self):
        """Initializing parameters for training RNN. Dimensions given"""

        sig = 0.01
        b = np.zeros((self.m,1))
        c = np.zeros((self.K,1))
        U = np.random.randn(self.m,self.K) * sig
        W = np.random.randn(self.m,self.m) * sig
        V = np.random.randn(self.K,self.m) * sig
        return b,c,U,W,V

    def synthesizeText(self, h, x0_i, n):
       '''Synthezie a sequence of characters
       h0 = hidden state at t=0,
       x0 = first dummy input,
       n = length of generated sequence'''
       x_next = np.zeros((self.K,1))
       x_next[x0_i] = 1
       txt = ''

       for t in range(n):
           _,h,_,p = self.fwdProp(h, x_next)
           x_i = np.random.choice(range(self.K), p=p.flat)
           x_next = np.zeros((self.K,1))
           x_next[x_i] = 1
           txt += self.ind_to_char[x_i]
       return txt

    def fwdProp(self, h, x):
        '''Forward propagates for current t
        Input h is h_t-1'''
        #print(h)
        a = np.matmul(self.W,h) + np.matmul(self.U,x) + self.b
        h = np.tanh(a)
        o = np.matmul(self.V, h) + self.c
        p = self.softmax(o)

        return a,h,o,p

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    def computeGradients(self, start_idx, h_start):
        '''Compute gradients for a sequence of 25 characters'''
        x_chars = self.book_data[start_idx : start_idx + self.seq_length]
        y_chars = self.book_data[start_idx+1 : start_idx + self.seq_length+1]

        #Matrices for storing intermediary parameters, every column is t of the sequence.
        x_mem = np.zeros((self.K,self.seq_length))
        y_mem = np.zeros((self.K,self.seq_length))
        a_mem = np.zeros((self.m,self.seq_length))
        h_mem = np.zeros((self.m,self.seq_length))
        o_mem = np.zeros((self.K,self.seq_length))
        p_mem = np.zeros((self.K,self.seq_length))

        loss = 0

        for i, char in enumerate(x_chars):

            #Store each one-hot of x & y as a column in x_mem & y_mem
            onehot_x = np.zeros((self.K,1))
            onehot_y = np.zeros((self.K,1))
            ix = self.char_to_ind[char]
            iy = self.char_to_ind[y_chars[i]]
            onehot_x[ix] = 1
            onehot_y[iy] = 1
            x_mem[:,i] = np.copy(onehot_x.reshape(self.K))
            y_mem[:,i] = np.copy(onehot_y.reshape(self.K))

            #Fwd propagating with one char
            if i == 0:
                a, h, o, p = self.fwdProp(h_start, onehot_x)
            else:
                a, h, o, p = self.fwdProp(h, onehot_x)

            a_mem[:,i], h_mem[:,i], o_mem[:,i], p_mem[:,i] = \
            a.reshape(self.m,), h.reshape(self.m,), o.reshape(self.K,), p.reshape(self.K,)

            truth_index = self.char_to_ind[y_chars[i]]

            #Cross Entropy Loss
            loss+= - (np.log(p[truth_index])[0])

        #Initializing storing for gradients
        W_grad = np.zeros_like(self.W)
        U_grad = np.zeros_like(self.U)
        V_grad = np.zeros_like(self.V)
        b_grad = np.zeros_like(self.b)
        c_grad = np.zeros_like(self.c)
        grads = {'W':W_grad,'U':U_grad,'V':V_grad,'b':b_grad,'c':c_grad}

        #Backward propagating w.r.t time t
        for t in reversed(range(self.seq_length)):

            #dL / do
            p_copy = np.copy(p_mem[:,t])
            dL_do = -(y_mem[:,t] - p_copy)
            dL_do = dL_do.reshape(-1,1).T

            #dL / dV
            grads['V'] += np.matmul(dL_do.T, h_mem[:, t].reshape(-1,1).T)

            #dl / dc
            grads['c'] += dL_do.T

            #dL / dh
            if t == self.seq_length-1:
                dL_dh = np.matmul(dL_do, self.V)
            else:
                dL_dh = np.matmul(dL_do, self.V) + np.matmul(dL_da, self.W)

            #dL / da
            at = a_mem[: , t]
            factor1 = dL_dh
            sqterm = np.square(np.tanh(at))
            factor2 = np.diag(1 - sqterm)
            dL_da = np.matmul(factor1,factor2)

            #dl / db
            grads['b'] += dL_da.T

            #dL / dU
            grads['U'] += np.matmul(dL_da.T, x_mem[:, t].reshape(-1,1).T)

            #dL / dW
            if t == 0:
                grads['W'] += np.matmul(dL_da.T, h_start.reshape(-1,1).T)
            else:
                grads['W'] += np.matmul(dL_da.T, h_mem[:, t-1].reshape(-1,1).T)


        #last computed h for current sequence, used to start off next sequence
        h = h_mem[:, self.seq_length-1].reshape(-1, 1)

        if not self.test_gradient:
            for key in grads.keys():
                grads[key] = np.clip(grads[key], -5, 5)

        return grads, loss, h

    def computeGradientsNUM(self, starth):
        '''Numerically computation of gradients. Used for testing gradients'''
        test_grad = True
        x_chars = self.book_data[0 : self.seq_length]
        y_chars = self.book_data[1 : self.seq_length+1]
        self.m = 5
        h = 1e-4

        num_grads = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U),
                     "V": np.zeros_like(self.V), "b": np.zeros_like(self.b),
                     "c": np.zeros_like(self.c)}

        for key, value in self.params.items():
            for i in range(value.shape[0]):
                for j in range(value.shape[1]):
                    self.params[key][i,j] -=h
                    _,loss1,_ = self.computeGradients(0, starth)
                    self.params[key][i,j] +=2*h
                    _,loss2,_ = self.computeGradients(0, starth)
                    num_grads[key][i,j] = (loss2-loss1) / (2*h)

        grad,_,_ = self.computeGradients(0, starth)
        print(self.test_gradient)
        if self.test_gradient:
            self.printRelGrad(grad,num_grads)

    def printRelGrad(self, grad, num_grads):
        '''Used for testing gradients'''
        eps = 1e-16
        for key,value in self.params.items():
            key = str(key)
            hej = np.absolute(grad[key] - num_grads[key]) / np.maximum(eps, (np.absolute(grad[key])+np.absolute(num_grads[key])))
            print(key , "\n" , hej , "\n")

if __name__ == '__main__':

    test_gradient = False
    np.set_printoptions(threshold=np.inf)
    data = prepareData()
    start_idx = 0
    iteration = 0 #25 chars per iteration
    epoch = 0
    epoch_max = 10

    #If Testing of Gradients
    if test_gradient:
        print("TESTING GRADIENTS FROM MAIN")
        m = 5
        RNN = RNN(test_gradient,data,m)
        h_prev = np.zeros((RNN.m, 1))
        RNN.computeGradientsNUM(h_prev)

    #If Training Model
    else:
        print("THIS IS NOT A DRILL")
        m = 100
        eps = 1e-16
        RNN = RNN(test_gradient,data,m)
        h_prev = np.zeros((RNN.m, 1))
        loss_memory = []

        #For saving synthezised text over time
        f=open("synth_over_time.txt", "a+")
        first_synth = RNN.synthesizeText(h_prev,0,200)
        f.write(first_synth)

        #memory for AdaGrad
        m = {
        'W' : np.zeros_like(RNN.W),
        'U' : np.zeros_like(RNN.U),
        'V' : np.zeros_like(RNN.V),
        'b' : np.zeros_like(RNN.b),
        'c' : np.zeros_like(RNN.c),
        }

        #ENTERING TRAINING LOOP
        while epoch<epoch_max:
            grads, loss, h_prev = RNN.computeGradients(start_idx, h_prev)

            #SMOOTH LOSS
            if iteration == 0 and epoch == 0:
                smooth_loss = loss
                lowest_loss = loss
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001*loss

            if smooth_loss<lowest_loss:
                lowest_loss = smooth_loss
                lowest_h = h_prev
                char = RNN.book_data[start_idx]
                lowest_char_idx = RNN.char_to_ind[char]

            if iteration % 100 == 0:
                print("Smoot Loss for itteration: ", iteration , "is ", smooth_loss, "\n")

            #PRINT SYNTETHIZED TEXT
            if iteration % 500 == 0:
                dummy_char = RNN.book_data[start_idx]
                dummy_ind = RNN.char_to_ind[dummy_char]
                synthesized_text = RNN.synthesizeText(h_prev, dummy_ind , 200)
                print("\nIteration: ", iteration)
                print("Smooth Loss: ", smooth_loss, "\n")
                print(synthesized_text)

            #For smooth loss plot
            if iteration % 1000 == 0:
                loss_memory.append(smooth_loss)

            if iteration % 10000 == 0 and iteration < 100000 and iteration != 0:
                dummy_char = RNN.book_data[start_idx]
                dummy_ind = RNN.char_to_ind[dummy_char]
                synthesized_text = RNN.synthesizeText(h_prev, dummy_ind , 200)
                f.write("\n\nIteration %d generated: \n %s" %(iteration,synthesized_text) )

            #AdaGrad UPDATE
            for key, value in RNN.params.items():
                m[key]+= np.power(grads[key],2)
                RNN.params[key]-= (RNN.eta*grads[key]) / (np.sqrt(m[key]+eps))

            #UPDATE TRACKERS
            iteration+=1
            start_idx = start_idx + RNN.seq_length

            #END OF BOOK, NEW EPOCH
            if start_idx >= len(RNN.book_data) - RNN.seq_length-1:
                start_idx = 0
                epoch+=1
                h_prev = np.zeros((RNN.m, 1))
                print("\nNEW EPOCH: ", epoch)

        #Print Smooth Loss Evolution after two epochs
        # x = np.linspace(0, len(loss_memory)*1000, len(loss_memory))
        # plt.plot(x, loss_memory, 'b', label='Smooth Loss')
        # plt.xlabel('t', fontsize = 12)
        # plt.ylabel('Smooth Loss', fontsize = 12)
        # plt.show()

        #Final synthezised text
        print("\n")
        print("Lowest loss is: " , lowest_loss)
        f1=open("finaltext.txt", "a+")
        final_synth = RNN.synthesizeText(lowest_h,lowest_char_idx,1000)
        f1.write(final_synth)
