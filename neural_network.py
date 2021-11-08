import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.style.use('seaborn')

class Dense_layer:
    def __init__(self, input_shape, dims, activation):
        self.input_shape = input_shape
        self.dims = dims
        self.output_layer = None
        self.d_output = None
        self.activation = activation
        self.weight = np.random.randn(dims, input_shape) * 0.01
        self.bias = np.zeros((dims,1))
    
    def softmax(self, z):
        eps = 1e-8
        z = np.exp(z)
        for i in range(len(z)):
            z[i] = z[i]/(np.sum(z[i])+eps)
        return z
                               
    def relu(self, z):
        return np.maximum(z, 0)
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def d_relu(self, z):
        return np.where(z>0, 1, 0)
    
    def d_sigmoid(self, z):
        return z*(1-z)
        
    def forward_pass(self, input_layer):
        z = np.dot(input_layer, self.weight.transpose()) + self.bias.transpose()
        self.input_z = input_layer
        if self.activation == None:
            self.d_z = np.ones(z.shape)
            return z
        
        elif self.activation == 'relu':
            self.d_z = self.d_relu(z)
            return self.relu(z)
        
        elif self.activation == 'sigmoid':
            self.d_z = self.d_sigmoid(z)
            return self.sigmoid(z)
                               
        elif self.activation == 'softmax':
            self.d_z = np.ones(z.shape)
            return self.softmax(z)

    def backward_pass(self, d_loss = None):
        if self.output_layer == None:
            self.d_output = self.d_z*d_loss
            self.d_w = np.dot(self.input_z.transpose(), self.d_output).transpose()
            self.d_b = np.sum(self.d_output, axis = 0).reshape(-1,1)
            
        else:
            self.d_output = self.d_z*np.dot(
                self.output_layer.d_output, self.output_layer.weight
            )
            self.d_w = np.dot(self.input_z.transpose(), self.d_output).transpose()
            self.d_b = np.sum(self.d_output, axis = 0).reshape(-1,1)
            
    def reset(self):
        self.weight = np.random.randn(self.dims, self.input_shape) * 0.01
        self.bias = np.zeros((self.dims,1))
        self.d_w = np.zeros_like(self.d_w)
        self.d_b = np.zeros_like(self.d_b)
           
class Model:
    def __init__(self, input_shape, loss, learning_rate):
        self.layers = []
        self.input_shape = input_shape
        self.loss = loss
        self.learning_rate = learning_rate
        
    def add(self, dims, activation = None):
        if len(self.layers) == 0:
            layer = Dense_layer(self.input_shape, dims, activation)
            self.layers.append(layer)
        else:
            layer = Dense_layer(self.layers[-1].dims, dims, activation)
            self.layers[-1].output_layer = layer
            self.layers.append(layer)
            
    def predict(self, x):
        for layer in self.layers:
            output = layer.forward_pass(x)
            x = output
        return output
    
    def optimize(self, x, y):
        y_pred = self.predict(x)
        indice = np.arange(len(self.layers))
        for i in np.flip(indice):
            self.layers[i].backward_pass(d_loss = self.d_loss(y, y_pred))
        for i in indice:
            self.layers[i].weight = self.layers[i].weight - self.learning_rate * self.layers[i].d_w
            self.layers[i].bias = self.layers[i].bias - self.learning_rate * self.layers[i].d_b
    
    def compute_loss(self, y_pred, y_true):
        if self.loss == 'categorical_crossentropy':      
            log_prob = np.ma.log(y_pred)
            return - np.sum(y_true * log_prob.filled(0))
            
        elif self.loss == 'mean_squared_error':
            return np.sum((y_true - y_pred)**2)
        
    def evaluate_performance(self, y_pred, y_true):
        if self.loss == 'categorical_crossentropy':
            #accuracy
            y_pred = np.argmax(y_pred, axis = 1)
            return 1 - np.mean(np.abs(y_pred - y_true))
            
        elif self.loss == 'mean_squared_error':
            #RMS
            return np.mean((y_true - y_pred)**2)**0.5
        
    def d_loss(self, y_true, y_pred):
        if self.loss == 'categorical_crossentropy':
            if self.layers[-1].activation == 'softmax':
                for i in range(len(y_pred)):
                    y_pred[i,int(y_true[i])] -= 1
                return y_pred
            else:
                y_true_one_hot = np.zeros_like(y_pred)
                for i in range(y_pred.shape[0]):
                    if y_true[i] == 1:
                        y_true_one_hot[i,0] = 1
                    else:
                        y_true_one_hot[i,1] = 1
                eps = 1e-8
                d_log_prob = 1/(y_pred+eps)
                return -(y_true_one_hot * d_log_prob)
            
        elif self.loss == 'mean_squared_error':
            return -2*np.sum(y_true - y_pred)*np.ones(y_pred.shape)
        
    def shuffle_set(self, x):
        indice = np.arange(len(x))
        np.random.shuffle(indice)
        x = x[indice]
        return x
    
    def get_batch(self, x, y, batch_size, total_batch, batch_now):
        if batch_now == total_batch - 1:
            x_batch = x[batch_now * batch_size:]
            y_batch = y[batch_now * batch_size:]
        else:
            x_batch = x[batch_now * batch_size:(batch_now+1) * batch_size]
            y_batch = y[batch_now * batch_size:(batch_now+1) * batch_size]
            
        return x_batch, y_batch
            
    def fit(self, x, y, epochs, batch_size, is_val = False, val_x = None, val_y = None, is_plot = True, verbose = True):
        loss_trace = np.zeros((2,epochs))
        total_batch = int(np.ceil(len(x)/batch_size))
        for epoch in range(epochs):
            x = self.shuffle_set(x)
            y = self.shuffle_set(y)
            for i in range(total_batch):
                x_batch, y_batch = self.get_batch(x, y, batch_size, total_batch, i)
                self.optimize(x_batch, y_batch)
            training_loss = self.compute_loss(self.predict(x), y)
            perfo = self.evaluate_performance(self.predict(x), y)
            message = 'epoch:{:2d}|{}, loss:{}, performance:{}'.format(
                epoch+1, epochs, np.around(training_loss,3), np.around(perfo,3)
            )
            if is_val == True:
                val_perfo = self.evaluate_performance(self.predict(val_x), val_y)
                val_message = ', val_performance:{}'.format(np.around(val_perfo,3))
                loss_trace[1,epoch] = val_perfo
                message += val_message
                
            loss_trace[0,epoch] = perfo
            if verbose == True:
                print(message)
            
        if is_plot == True:
            self.plot_loss(
                loss_trace[0], 'Training and validation RMS', 'RMS', is_val = is_val, val_loss = loss_trace[1]
            )
            
    def plot_loss(self, train_loss, title, performance, is_val = False, val_loss = None):
        epoch=np.arange(1,len(train_loss)+1)
        plt.plot(epoch, train_loss, '-', label='Training'+ str(performance))
        if is_val == True:
            plt.plot(epoch, val_loss, '-', label='Validation'+ str(performance))
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
    def reset(self):
        for layer in self.layers:
            layer.reset()