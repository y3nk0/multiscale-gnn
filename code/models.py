import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_scatter import scatter_mean

import networkx as nx
import numpy as np
import scipy.sparse as sp
import warnings

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('fbprophet')
datas = collect_data_files('fbprophet')

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)


def arima(ahead,start_exp,n_samples,labels):
    var = []
    for idx in range(ahead):
        var.append([])

    error = np.zeros(ahead)
    count = 0
    f = open("../results/preds_arima.txt","w")
    for test_sample in range(start_exp,n_samples-ahead):#
        print(test_sample)
        count+=1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            if(sum(ds.iloc[:,1])==0):
                yhat = [0]*(ahead)
            else:
                # import ipdb; ipdb.set_trace()
                try:
                    fit2 = ARIMA(ds.iloc[:,1].values, (2, 0, 2)).fit(disp=0)
                except:
                    fit2 = ARIMA(ds.iloc[:,1].values).fit()
                yhat = abs(fit2.predict(start = test_sample , end = (test_sample+ahead-1) ))
                # yhat = abs(fit2.predict(start = test_sample, end = (test_sample+ahead-2) ))
            f.write(str(yhat)+"\n")
            y_me = labels.iloc[j,test_sample:test_sample+ahead]
            f.write(str(y_me)+"\n\n")
            e = abs(yhat - y_me.values)
            err += e
            error += e

        for idx in range(ahead):
            var[idx].append(err[idx])

    f.close()
    return error, var



def prophet(ahead, start_exp, n_samples, labels):
    var = []
    for idx in range(ahead):
        var.append([])

    error= np.zeros(ahead)
    count = 0
    f = open("../results/preds_phophet.txt","w")
    for test_sample in range(start_exp,n_samples-ahead):#
        print(test_sample)
        count+=1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample].reset_index()
            ds.columns = ["ds","y"]
            #with suppress_stdout_stderr():
            m = Prophet(interval_width=0.95)
            m.fit(ds)
            future = m.predict(m.make_future_dataframe(periods=ahead))
            yhat = future["yhat"].tail(ahead)
            f.write(str(yhat)+"\n")
            y_me = labels.iloc[j,test_sample:test_sample+ahead]
            f.write(str(y_me)+"\n\n")
            e = abs(yhat-y_me.values).values
            err += e
            error += e
        for idx in range(ahead):
            var[idx].append(err[idx])
    f.close()
    return error, var




class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, classification=False):
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.classification = classification
        #self.batch_size = batch_size
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)

        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)

        #self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    def forward(self, adj, x, y, z, w ):
        lst = list()
       # print("--------------------")
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       # print(x.shape)
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
       # print(skip.shape)
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes

        x = self.relu(self.conv1(x, adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        x = torch.cat(lst, dim=1)

        #--------------------------------------
        #print(x.shape)
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        #print(x.shape)
        #print(x.shape)
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes

        #print(x.shape)
        #print("------")
        x, (hn1, cn1) = self.rnn1(x)


        out2, (hn2,  cn2) = self.rnn2(x)

        #print(self.rnn2._all_weights)
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
        #print(skip.shape)
        #print(x.shape)
        #skip = skip.view(skip.size(0),-1)
        skip = skip.reshape(skip.size(0),-1)
        #print(x.shape)
        #print(skip.shape)

        x = torch.cat([x,skip], dim=1)
        #--------------------------------------
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        if self.classification:
            x = F.log_softmax(self.fc2(x))
        else:
            x = self.relu(self.fc2(x)).squeeze()
            x = x.view(-1)
       # print("--------------------")


        return x




class MPNN(nn.Module):
    def __init__(self, nfeat, nfeat_iris, nhid, nout, dropout, n_nodes, n_nodes_iris, iris=False):
        super(MPNN, self).__init__()
        self.n_nodes_iris = n_nodes_iris
        self.n_nodes = n_nodes

        #self.batch_size = batch_size
        self.nhid = nhid
        self.iris = iris

        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        if self.iris:
            self.fc1 = nn.Linear(nfeat+4*nhid, nhid)
        else:
            self.fc1 = nn.Linear(nfeat+2*nhid, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        #self.bn3 = nn.BatchNorm1d(nhid)
        #self.bn4 = nn.BatchNorm1d(nhid)

        if self.iris:
            self.conv3 = GCNConv(nfeat_iris, nhid)
            self.conv4 = GCNConv(nhid, nhid)
            self.bn3 = nn.BatchNorm1d(nhid)
            self.bn4 = nn.BatchNorm1d(nhid)

            self.fc3 = nn.Linear(n_nodes_iris, nhid)
            self.fc4 = nn.Linear(nfeat, nhid)
            self.fc5 = nn.Linear(nhid, nout)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.logSoftmax = nn.LogSoftmax()

        # if self.classification:
        # self.layer_out = nn.Linear(nout, num_class)

        #nn.init.zeros_(self.conv1.weight)
        #nn.init.zeros_(self.conv2.weight)
        #nn.init.zeros_(self.fc1.weight)
        #nn.init.zeros_(self.fc2.weight)


    def forward(self, adj, x_init, iris_adj, iris_x, iris_ind):

        lst = list()
        #print(x.shape)
        #print(adj.shape)

        #lst.append(ident)

        #x = x[:,mob_feats]
        #x = xt.index_select(1, mob_feats)

        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()

        lst.append(x_init)

        if self.iris:
            # lst.append(iris_x)
            iris_weight = iris_adj.coalesce().values()
            iris_adj = iris_adj.coalesce().indices()

            x = self.relu(self.conv3(iris_x, iris_adj, edge_weight=iris_weight))

            x = self.bn3(x)
            x_d = self.dropout(x)

            #print(x)
            #print(x.shape)

            # x = self.fc3(x.T)
            # x = torch.cat(lst, dim=1)
            # import ipdb; ipdb.set_trace()

            iris_ind = torch.stack(iris_ind)
            #iris_ind = torch.reshape(iris_ind, (-1,))
            # print(iris_ind.size())
            iris_ind = iris_ind.view(-1)
            #x = gmp(x_d, iris_ind)
            x = gap(x_d, iris_ind)

            lst.append(x)
            # lst.append(x2)

            x = self.relu(self.conv4(x_d, iris_adj, edge_weight=iris_weight))
            # # # #print(x.shape)
            x = self.bn4(x)
            x = self.dropout(x)
            # lst.append(x)


            # print(iris_ind.size())
            #x = gmp(x, iris_ind)
            x = gap(x, iris_ind)
            # x = scatter_mean(x, iris_ind)
            # print(x.size())
            lst.append(x)
            # lst.append(x2)

        x = self.relu(self.conv1(x_init,adj,edge_weight=weight))
        #print(x)
        #print(x.size())
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        #print(x.shape)
        x = self.bn2(x)
        x = self.dropout(x)

        lst.append(x)
        # import ipdb; ipdb.set_trace()
        x = torch.cat(lst, dim=1)

        x = self.relu(self.fc1(x))
        #x = self.bn3(x)
        x = self.dropout(x)

        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
        #x = self.bn4(x)

        # if self.classification:
        #     # x = self.dropout(x)
        #     x = self.logSoftmax(x)

        # x = x.view(-1)

        return x




class LSTM(nn.Module):
    def __init__(self, nfeat, nhid, n_nodes, window, dropout,batch_size, recur, classification=False):
        super().__init__()
        self.nhid = nhid
        self.n_nodes = n_nodes
        self.nout = n_nodes
        self.window = window
        self.nb_layers= 2
        self.classification = classification
        self.nfeat = nfeat
        self.recur = recur
        self.batch_size = batch_size
        self.lstm = nn.LSTM(nfeat, self.nhid, num_layers=self.nb_layers)

        self.linear = nn.Linear(nhid, self.nout)
        self.cell = ( nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))

        #self.hidden_cell = (torch.zeros(2,self.batch_size,self.nhid).to(device),torch.zeros(2,self.batch_size,self.nhid).to(device))
        #nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))


    def forward(self, adj, features, y, w, z):
        #adj is 0 here
        #print(features.shape)
        features = features.view(self.window,-1, self.n_nodes)#.view(-1, self.window, self.n_nodes, self.nfeat)
        #print(features.shape)
        #print("----")


        #------------------
        if(self.recur):
            #print(features.shape)
            #self.hidden_cell =
            try:
                lstm_out, (hc,self.cell) = self.lstm(features,(torch.zeros(self.nb_layers,self.batch_size,self.nhid).cuda(),self.cell))
                # = (hc,cn)
            except:
                #hc = self.hidden_cell[0][:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
                hc = torch.zeros(self.nb_layers,features.shape[1],self.nhid).cuda()
                cn = self.cell[:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
                lstm_out, (hc,cn) = self.lstm(features,(hc,cn))
        else:
        #------------------
            lstm_out, (hc,cn) = self.lstm(features)#, self.hidden_cell)#self.hidden_cell

        if self.classification:
            predictions = F.log_softmax(lstm_out)
            predictions = predictions[-1]
        else:
            predictions = self.linear(lstm_out)#.view(self.window,-1,self.n_nodes)#.view(self.batch_size,self.nhid))#)
            predictions = predictions[-1].view(-1)
        #print(predictions.shape)
        return predictions
