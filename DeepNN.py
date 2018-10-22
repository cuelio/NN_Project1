import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import math

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


########################################################################################################################
# Start coding here.
########################################################################################################################







class n_layer_NN(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, num_hidden_layers, input_layer, hidden_layer, output_layer, actFun_type='tanh', reg_lambda=0.01, seed=10):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''

        self.num_hidden_layers = num_hidden_layers
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        self.losses = []

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.in_W = np.random.randn(self.input_layer, self.hidden_layer) / np.sqrt(self.input_layer)
        # self.in_W = np.ones((self.input_layer, self.hidden_layer))
        # self.b1 = np.zeros((1, self.hidden_layer))
        self.out_W = np.random.randn(self.hidden_layer, self.output_layer) / np.sqrt(self.hidden_layer)
        self.out_B = np.random.randn(1, self.output_layer) / np.sqrt(self.output_layer)

        ### N-Layers ###
        self.hidden_W = np.random.randn(self.num_hidden_layers-1, self.hidden_layer, self.hidden_layer) / np.sqrt(self.num_hidden_layers)
        self.hidden_Act = np.random.randn(self.num_hidden_layers, 200, self.hidden_layer)
        self.hidden_B = np.ones((self.num_hidden_layers, self.hidden_layer))
        self.hidden_A = np.random.randn(self.num_hidden_layers, 200, self.hidden_layer)


    def actFun(self, a, non_Linearity):
        '''
        actFun computes the activation functions
        :param a = net input
        :param non_Linearity = Tanh, Sigmoid, or ReLU
        :return: net activation
        '''
        if(non_Linearity == "tanh"):
            return np.tanh(a)

        elif(non_Linearity == "relu"):
            return max(0,a)

        elif(non_Linearity == "sigmoid"):
            e_x = math.e**a
            return (e_x/(e_x+1))
        else:
            print("INVALID FUNCTION NAME")


    def diff_actFun(self, a, non_Linearity):
        '''
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param a= net input
        :param non_Linearity = Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''
        # YOU IMPLEMENT YOUR diff_actFun HERE
        if(non_Linearity == "tanh"):
            return (1 - np.tanh(a)*np.tanh(a))

        elif(non_Linearity == "relu"):
            if(a > 0):
                return 1
            else:
                return 0

        elif(non_Linearity == "sigmoid"):
            act = self.actFun(a, "sigmoid")
            return act*(1-act)

        else:
            print("INVALID FUNCTION NAME")


    def softmax(self, z):
        exp_sum = np.sum(np.exp(z), 1)
        t = np.tile(exp_sum, [2, 1]).T
        out = np.exp(z) * (1 / t)
        return out

    def diff_softmax(self, z, i, j):
        if(i == j):
            return self.probs[i]*(1-self.probs[i])
        else:
            return -self.probs[j]*self.probs[i]


    def ForwardPass(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR ForwardPass HERE
        self.a1 = np.dot(X, self.in_W) + self.hidden_B[0]
        self.hidden_A[0] = self.a1
        self.hidden_Act[0] = actFun(self.a1)

        for l in range(0, self.num_hidden_layers-1):
            self.hidden_A[l+1] = np.dot(self.hidden_Act[l], self.hidden_W[l]) + self.hidden_B[l]
            self.hidden_Act[l + 1] = actFun(self.hidden_A[l+1])

        self.a2 = np.dot(self.hidden_Act[self.num_hidden_layers - 1], self.out_W) + self.out_B

        self.probs = self.softmax(self.a2)

        return None

    def calculate_loss(self, X, t):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.ForwardPass(X, lambda x: self.actFun(x, self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        data_loss = 0
        for n in range(0, len(t)):
            for i in range(0, len(t[0])):
                data_loss += t[n][i]*np.log(self.probs[n][i])

        data_loss = -(1/len(t))*data_loss
        return data_loss


    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.ForwardPass(X, lambda x: self.actFun(x, self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backwardPass(self, X, t):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''
        # IMPLEMENT YOUR BACKPROP HERE
        # print("PROBS: " + str(self.probs))
        delta_scores = (self.probs - t)

        dH = np.zeros((self.num_hidden_layers - 1, self.hidden_layer, self.hidden_layer))
        dHb = np.zeros((self.num_hidden_layers, self.hidden_layer))
        dH_temp = np.zeros((len(t), self.hidden_layer))

        # update output layer bias/weights

        d_out_W = np.dot(self.hidden_Act[self.num_hidden_layers-1].T, delta_scores)

        d_out_B = np.sum(delta_scores, axis=0)

        if(self.num_hidden_layers == 1):
            dH_temp = np.dot(delta_scores, self.out_W.T) * self.diff_actFun(self.a1, self.actFun_type)
            dHb[0] = np.sum(dH_temp, axis=0)
        else:
            for i in range(0, self.num_hidden_layers-1):
                # start from rightmost layer and move to the left
                index = self.num_hidden_layers - i - 1
                if(index == self.num_hidden_layers - 1):
                    dH_temp = np.dot(delta_scores, self.out_W.T) * self.diff_actFun(self.hidden_A[index], self.actFun_type)
                    dH[index-1] = np.dot(dH_temp.T, self.hidden_Act[index-1])
                    dHb[index] = np.sum(dH_temp, axis=0)

                else:
                    dH_temp = np.dot(dH_temp, self.hidden_W[index]) * self.diff_actFun(self.hidden_A[index], self.actFun_type)
                    dH[index-1] = np.dot(dH_temp.T, self.hidden_Act[index-1])
                    dHb[index] = np.sum(dH_temp, axis=0)

            dHb[0] = np.sum(dH_temp, axis=0)

        if(self.num_hidden_layers > 1):
            dH_temp = np.dot(dH_temp, self.hidden_W[0]) * self.diff_actFun(self.a1, self.actFun_type)

        d_in_W = np.dot(X.T, dH_temp)

        return d_in_W, d_out_W, d_out_B, dH, dHb

    def fit_model(self, X, t, epsilon, num_passes=14000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent implementation
        for i in range(0, num_passes+1):

            # Forward propagation
            self.ForwardPass(X, lambda x: self.actFun(x, self.actFun_type))
            # Backpropagation
            d_in_W, d_out_W, db2, dH, dHb = self.backwardPass(X, t)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            d_out_W += self.reg_lambda * self.out_W
            d_in_W += self.reg_lambda * self.in_W

            if(len(dH) > 0):
                dH += self.reg_lambda * self.hidden_W
                self.hidden_W += -epsilon * dH

            self.hidden_B += -epsilon * dHb

            # Gradient descent parameter update
            self.in_W += -epsilon * d_in_W
            self.out_W += -epsilon * d_out_W
            self.out_B += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, t)))
                result = self.calculate_loss(X, t)
                self.losses.append(result)
        mean = np.mean(self.losses)

        for i in range(0, 200):
            print(str(self.probs[i]) + ", " + str(t[i]))
        return mean, self.losses

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)


def main():
    # # generate and visualize Make-Moons dataset
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.25)
    t = np.zeros((len(y), 2))
    for i, val in np.ndenumerate(y):
        if val == 0:
            t[i, 0] = 0
            t[i, 1] = 1
        else:
            t[i, 0] = 1
            t[i, 1] = 0
    plt.scatter(X[:, 0], X[:, 1], s=45, c=y, cmap=plt.cm.plasma)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.show()

    units = 3
    layers = 1
    # act = "sigmoid"
    act = "tanh"
    # act = "relu"
    model = n_layer_NN(num_hidden_layers = layers, input_layer=2, hidden_layer=units, output_layer=2, actFun_type=act)

    mean, loss = model.fit_model(X, t, 0.01)
    # means.append(mean)
    # model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
    main()
