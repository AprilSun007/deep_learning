import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchviz import make_dot

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import pdb

import logging
logger = logging.getLogger('Neural Network')
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

def neuralnet_cv(x, y, params, neuralnet, n_folds = 5, n_jobs = None):
    
    if neuralnet == 'FNN':  
        #pdb.set_trace()
        gs = GridSearchCV(FNNFitPredict(), param_grid=params, cv = n_folds, n_jobs = n_jobs)
    elif neuralnet == 'LSTM':
        gs = GridSearchCV(LSTMFitPredict(), param_grid=params, cv = n_folds, n_jobs = n_jobs)
    
    cv_result = gs.fit(x, y)
    
    return cv_result

class FNN(nn.Module):
    def __init__(self, input_size = 6, output_size = 1, hidden_sizes = [4,2], hidden_layer_fn = 'ReLU', output_layer_fn = 'Linear'):
        
        super().__init__()
        
        n = len(hidden_sizes)
        self.hidden = []
        for i in range(n):
            if i == 0:
                self.hidden.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.hidden.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                
        # Output layer, 10 units - one for each digit
        self.output = getattr(nn, output_layer_fn)(hidden_sizes[n-1], output_size)
        
        # Define sigmoid activation and softmax output 
        self.activate_fn = getattr(nn, hidden_layer_fn)()
        
    def forward(self, x):
        
        n = len(self.hidden)
        # passing through each hidden layer
        for i in range(n):
            x = self.hidden[i](x)
            x = self.activate_fn(x)
       
        x = self.output(x) 
        
        return x
    
class LSTM(nn.Module):
    
    def __init__(self, input_size=1, output_size=1, hidden_layer_size=100, n_layers = 1, n_directions = 1, batch_size = 1):
        
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_layer_size, n_layers)
        self.output = nn.Linear(hidden_layer_size, output_size)

        # hidden_cell = (h_0, c_0)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. 
        # If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        # c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        self.hidden_cell = (torch.zeros(n_layers * n_directions, batch_size, hidden_layer_size),
                            torch.zeros(n_layers * n_directions, batch_size, hidden_layer_size))

    def forward(self, input_seq):
        #print(input_seq.shape)
        # lstm(input_sequence, hidden_cell) where hidden_cell = (hidden_status, cell_status)
        # shape of input_sequence : seq_len, batch, input_size
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(input_seq.shape[1] ,input_seq.shape[0], -1), self.hidden_cell)
        # shape of lstm_out: seq_len, batch, num_directions * hidden_size
        predictions = self.output(lstm_out.view(input_seq.shape[1] ,input_seq.shape[0], -1))
        
        return predictions[-1]
    
class NNFitPredict(BaseEstimator, RegressorMixin):
    
    def __init__(self, loss_name = 'MSELoss'):
        self.loss = getattr(nn,loss_name)()
        
        
    def fit(self, x, y):
        
        train_loader, test_loader = self.data_process(x, y, 0)
        train_losses, _ = self.train(train_loader)
        
        return train_losses
    
    def predict(self, x):    
        x_tensor = torch.from_numpy(x).float()
        y_pred = self.model(x_tensor)
        y_pred = y_pred.detach().numpy()
        return y_pred
    
    def score(self, x, y):
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float().view(-1,1)
        y_pred = self.model(x_tensor)
        return self.loss(y_tensor, y_pred).item()
        
    def data_process(self, x, y, test_size = 0.33):
        
        n_sample = x.shape[0]
        n_test_size = int(n_sample*test_size)
        train_test_size = [n_sample - n_test_size, n_test_size]
        
        # This is wrong
        if self.normalize:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            x = scaler.fit_transform(x .reshape(n_sample, -1))
        
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.from_numpy(y).float()

        dataset = TensorDataset(x_tensor, y_tensor)

        train_dataset, val_dataset = random_split(dataset, train_test_size)
        
        if self.train_batch_size is None:
            train_loader = DataLoader(dataset=train_dataset, batch_size = train_dataset.__len__())
        else:
            train_loader = DataLoader(dataset=train_dataset, batch_size = self.train_batch_size)
        
        if self.train_batch_size is None:
            test_loader = DataLoader(dataset=val_dataset, batch_size = val_dataset.__len__())
        else:
            test_loader = DataLoader(dataset=val_dataset, batch_size = self.test_batch_size)
                    
        return [train_loader, test_loader]
    
    def train_step(self):
        pass    
        
    def train(self, train_loader, val_loader = None):
        
        train_losses = []
        for epoch in range(self.n_epochs):
            for x_batch, y_batch in train_loader:
                #print(x_batch.shape)
                loss = self.train_step(x_batch, y_batch)
                train_losses.append(loss)
                
            if epoch % 10 == 1:
                logger.info(f"epoch {epoch}: train loss -- {train_losses[-1]} ")

            with torch.no_grad():
                val_losses = []
                if val_loader is not None:
                    for x_val, y_val in val_loader:
                        self.model.eval()
                        yhat = self.model(x_val)
                        val_loss = self.loss(y_val.view(-1,1), yhat)
                        val_losses.append(val_loss.item())
                
                    if epoch % 10 == 1:
                        logger.info(f"epoch {epoch}: val loss -- {val_losses[-1]}") 
    
        return [train_losses, val_losses]
    
    def visualize(self, x):
        
        x_tensor = torch.from_numpy(x).float()
        y_pred = self.model(x_tensor)
        make_dot(y_pred)
            
class FNNFitPredict(NNFitPredict):
    
    def __init__(self, input_size = 6, output_size = 1, hidden_sizes = [4,2],  
                 optimizer_name = 'SGD', lr = 0.01, n_epochs = 100, 
                 train_batch_size = None, test_batch_size = None, normalize = False, **kwargs):
        
        super().__init__(**kwargs)    
        
        self.model = FNN(input_size = input_size, output_size = output_size, hidden_sizes = hidden_sizes)
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr = lr)        
        self.n_epochs = n_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.normalize = normalize
        
               
    def train_step(self,x, y):
        
        self.model.train()
        y_ = self.model(x)    

        loss = self.loss(y.view(-1,1), y_)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
        
class LSTMFitPredict(NNFitPredict):
    
    def __init__(self, input_size = 1, output_size = 1, hidden_size = 100, n_layers = 1, n_directions = 1, batch_size = 1,
                 optimizer_name = 'SGD', lr = 0.01, n_epochs = 100, 
                 train_batch_size = None, test_batch_size = None, normalize = False, **kwargs):
        
        super().__init__(**kwargs)
        self.model = LSTM(input_size, output_size, hidden_layer_size = hidden_size, n_layers = n_layers, n_directions = n_directions, batch_size = batch_size)
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = n_directions
        self.batch_size = batch_size
        
        self.lr = lr
        print(optimizer_name)
        print(type(optimizer_name))
        self.optimizer_name = optimizer_name
        self.optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr = lr)        
        self.n_epochs = n_epochs
        self.train_batch_size = batch_size
        self.test_batch_size = batch_size
        self.normalize = normalize
        
    def train_step(self, x, y):
        
        self.model.hidden_cell = (torch.zeros(self.n_layers * self.n_directions, self.train_batch_size, self.hidden_size),
                        torch.zeros(self.n_layers * self.n_directions, self.train_batch_size, self.hidden_size))

        #seq = TensorDataset(x, y)
        #print(x.shape)
        y_ = self.model(x)

        loss = self.loss(y, y_)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
        
    def create_data_sequences(self, ts, train_window = 20):       
        x = []
        y = []
        L = len(ts)
        for i in range(L-train_window):
            x.append(ts[i:i+train_window])
            y.append(ts[(i+train_window):(i+train_window+1)])
        return [np.array(x), np.array(y)]
    

    
