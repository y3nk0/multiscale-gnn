#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import networkx as nx
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import EarlyStopping

from math import ceil
from datetime import timedelta, date

import itertools
import pandas as pd

from utils import generate_new_features, generate_new_batches, generate_new_batches_plus_iris, AverageMeter, generate_batches_lstm, read_meta_datasets
from models import MPNN_LSTM, LSTM, MPNN, prophet, arima

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)


def train(epoch, adj_sample, features_sample, y_sample):
    optimizer.zero_grad()
    output = model(adj_sample, features_sample, [], [], [])
    loss_train = F.mse_loss(output, y_sample)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train

def train_iris(epoch, adj_sample, features_sample, y_sample, adj_iris_sample, features_iris_sample, irises_ind):
    optimizer.zero_grad()
    output = model(adj_sample, features_sample, adj_iris_sample, features_iris_sample, irises_ind)
    loss_train = F.mse_loss(output, y_sample)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def test(adj_sample, features_sample, y_sample):
    output = model(adj_sample, features_sample, [], [], [])
    loss_test = F.mse_loss(output, y_sample)
    return output, loss_test

def test_iris(adj_sample, features_sample, y_sample, adj_iris_sample, features_iris_sample, irises_ind):
    output = model(adj_sample, features_sample, adj_iris_sample, features_iris_sample, irises_ind)
    loss_test = F.mse_loss(output, y_sample)
    return output, loss_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur', default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=3,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')

    parser.add_argument('--infra', type=bool, default=True,
                        help='Use of infradepartemental data.')
    parser.add_argument('--vac', type=bool, default=True,
                        help='Vaccination data.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

    infra = args.infra
    vac = args.vac
    sdate = date(2020, 12, 27)
    edate = date(2021, 6, 27)
    country = 'FR'
    idx = 0

    pos_cases = False
    holiday = False

    # meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets(args.window, sdate, edate, infra, vac, pos_cases)
    if infra:
        meta_labs, meta_labs_iris, meta_graphs, meta_graphs_iris, meta_features, meta_features_iris, meta_y, meta_y_iris, irises_in, deps, nodes_iris = read_meta_datasets(args.window, sdate, edate, infra, vac, pos_cases, holiday)
    else:
        meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets(args.window, sdate, edate, infra, vac, pos_cases, holiday)

    labels = meta_labs[idx]
    gs_adj = meta_graphs[idx]
    features = meta_features[idx]
    y = meta_y[idx]
    n_samples = len(gs_adj)
    nfeat = meta_features[0][0].shape[1]
    n_nodes = gs_adj[0].shape[0]
    print(n_nodes)

    if infra:
        labels_iris = meta_labs_iris[idx]
        gs_adj_iris = meta_graphs_iris[idx]
        features_iris = meta_features_iris[idx]
        y_iris = meta_y_iris[idx]
        n_samples_iris = len(gs_adj_iris)
        nfeat_iris = meta_features_iris[0][0].shape[1]
        n_nodes_iris = gs_adj_iris[0].shape[0]
        print(n_nodes_iris)

    # if not os.path.exists('../results'):
        # os.makedirs('../results')

    fw = open("../results_iter/hosp_NEW_cum_pop_dep_results_infra"+str(infra)+"_days"+str(args.ahead)+"_vac"+str(vac)+"_hid"+str(args.hidden)+"_window"+str(args.window)+"_batch"+str(args.batch_size)+"_"+sdate.strftime("%Y-%m-%d")+"_"+edate.strftime("%Y-%m-%d")+".csv","w")
    fw.write("cum_pop_dep_results_infra"+str(infra)+"_days"+str(args.ahead)+"_vac"+str(vac)+"_hid"+str(args.hidden)+"_window"+str(args.window)+"_batch"+str(args.batch_size)+"_"+sdate.strftime("%Y-%m-%d")+"_"+edate.strftime("%Y-%m-%d")+"\n\n")

    # fw = open("../results/epci_gmp_gap_plus_cum_pop_dep_results_days"+str(args.ahead)+"_vac"+str(vac)+"_"+sdate.strftime("%Y-%m-%d")+"_"+edate.strftime("%Y-%m-%d")+".csv","w")
    # fw.write("epci_gmp_gap_plus_cum_pop_dep_results_days"+str(args.ahead)+"_vac"+str(vac)+"_"+sdate.strftime("%Y-%m-%d")+"_"+edate.strftime("%Y-%m-%d")+"\n\n")

    ferr = open("../results/hosp_new_errors_new_mpnn_infra_plus_vac.txt","w")
    #ferr = open("../results/errors_new_hosp_infra_plus_vac_plus_pos.txt","a")
    #ferr = open("../results/errors_new_hosp_infra_plus_vac_plus_pos.txt","a")
    # fw.write(str(args)+"\n\n")
    ferr.write("cum_pop_dep_results_infra"+str(infra)+"_days"+str(args.ahead)+"_vac"+str(vac)+"_hid"+str(args.hidden)+"_window"+str(args.window)+"_batch"+str(args.batch_size)+"_"+sdate.strftime("%Y-%m-%d")+"_"+edate.strftime("%Y-%m-%d")+"\n\n")
    fw.write(str(args)+"\n\n")

    # done "PROPHET"
    #for args.model in ["PROPHET","ARIMA"]:
    #for args.model in ["AVG"]:
    # for args.model in ["ARIMA"]:
    for args.model in ["MPNN"]:
    # for args.model in ["LSTM"]:
        # all_results_iter = []

        ferr.write(str(args)+"\n\n")

        # for it in range(1):
        all_results = []

        if(args.model=="PROPHET"):
            error, var = prophet(args.ahead,args.start_exp,n_samples,labels)
            count = len(range(args.start_exp,n_samples-args.ahead))
            for idx,e in enumerate(error):
                #fw.write(args.model+","+str(shift)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+"\n")
                fw.write("PROPHET,"+str(idx)+",{:.5f}".format(float(e/count)/n_nodes)+",{:.5f}".format(np.std(var[idx]))+"\n")
                all_results.append(float(e/count)/n_nodes)

            all_res_mean = np.mean(all_results)
            all_res_std = np.std(all_results)
            fw.write("Mean: "+",{:.5f}".format(all_res_mean)+",{:.5f}".format(all_res_std)+"\n")

            continue

        if(args.model=="ARIMA"):
            error, var = arima(args.ahead,args.start_exp,n_samples,labels)
            count = len(range(args.start_exp,n_samples-args.ahead))

            for idx,e in enumerate(error):
                fw.write("ARIMA,"+str(idx)+",{:.5f}".format(e/count)+",{:.5f}".format(np.std(var[idx]))+"\n")
                all_results.append(float(e/count)/n_nodes)

            all_res_mean = np.mean(all_results)
            all_res_std = np.std(all_results)
            fw.write("Mean: "+",{:.5f}".format(all_res_mean)+",{:.5f}".format(all_res_std)+"\n")

            continue


        if infra:
            irises_ind_tmp = []

            for node in nodes_iris:
                s_node = str(node)
                if len(s_node)==8:
                    s_node = "0" + s_node
                dep = irises_in[s_node]
                # import ipdb; ipdb.set_trace()
                # if dep in
                irises_ind_tmp.append(deps.index(dep))

            irises_ind = torch.LongTensor(irises_ind_tmp).to(device)

            irises_ind_batch = []
            for b in range(100):
                irises_ind_b = [(b*94)+ir for ir in irises_ind]
                irises_ind_batch.append(torch.LongTensor(irises_ind_b).to(device))

		#---- predict days ahead , 0-> next day etc.
        f = open("../results/hosp_preds_avg.txt","w")

        #f = open("../results/hosp_preds_avg_window.txt","w")

        for shift in list(range(0,args.ahead)):

            result = []
            exp = 0

            for test_sample in range(args.start_exp,n_samples-shift):#
                exp+=1
                print(test_sample)

                #----------------- Define the split of the data
                idx_train = list(range(args.window-1, test_sample-args.sep))

                idx_val = list(range(test_sample-args.sep,test_sample,2))

                idx_train = idx_train+list(range(test_sample-args.sep+1,test_sample,2))

                #--------------------- Baselines
                if(args.model=="AVG"):

                    avg = labels.iloc[:,:test_sample-1].mean(axis=1)
                    f.write(avg.to_string()+"\n")
                    targets_lab = labels.iloc[:,test_sample+shift]
                    f.write(targets_lab.to_string()+"\n\n")
                    error = np.sum(abs(avg - targets_lab))/n_nodes
                    print(error)
                    result.append(error)

                    continue


                if(args.model=="LAST_DAY"):
                    win_lab = labels.iloc[:,test_sample-1]
                    #print(win_lab[1])
                    targets_lab = labels.iloc[:,test_sample+shift]#:(test_sample+1)]
                    error = np.sum(abs(win_lab - targets_lab))#/avg)
                    if(not np.isnan(error)):
                        result.append(error)
                    else:
                        exp-=1
                    continue


                if(args.model=="AVG_WINDOW"):

                    win_lab = labels.iloc[:,(test_sample-args.window):test_sample]
                    f.write(win_lab.mean(1).to_string()+"\n")
                    targets_lab = labels.iloc[:,test_sample+shift]#:
                    f.write(targets_lab.to_string()+"\n\n")
                    error = np.sum(abs(win_lab.mean(1) - targets_lab))/n_nodes
                    if(not np.isnan(error)):
                        result.append(error)
                    else:
                        exp-=1

                    continue


                if(args.model=="LSTM"):
                    lstm_features = 1*n_nodes
                    adj_train, features_train, y_train = generate_batches_lstm(n_nodes, y, idx_train, args.window, shift,  args.batch_size, device, test_sample)
                    adj_val, features_val, y_val = generate_batches_lstm(n_nodes, y, idx_val, args.window, shift, args.batch_size, device, test_sample)
                    adj_test, features_test, y_test = generate_batches_lstm(n_nodes, y, [test_sample], args.window, shift,  args.batch_size, device, test_sample)


                elif(args.model=="MPNN_LSTM"):
                    adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train, args.graph_window, shift, args.batch_size,device,test_sample)
                    adj_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val, args.graph_window,  shift,args.batch_size, device,test_sample)
                    adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y,  [test_sample], args.graph_window,shift, args.batch_size, device,test_sample)

                else:
                    if infra:
                        adj_train, features_train, y_train, adj_train_iris, features_train_iris, y_train_iris = generate_new_batches_plus_iris(gs_adj, features, y, gs_adj_iris, features_iris, y_iris, irises_in, deps, nodes_iris, idx_train, 1, shift,args.batch_size,device,test_sample)
                        adj_val, features_val, y_val, adj_val_iris, features_val_iris, y_val_iris = generate_new_batches_plus_iris(gs_adj, features, y, gs_adj_iris, features_iris, y_iris, irises_in, deps, nodes_iris, idx_val, 1, shift,args.batch_size,device,test_sample)
                        adj_test, features_test, y_test, adj_test_iris, features_test_iris, y_test_iris = generate_new_batches_plus_iris(gs_adj, features, y, gs_adj_iris, features_iris, y_iris, irises_in, deps, nodes_iris, [test_sample], 1, shift,args.batch_size, device,-1)
                    else:
                        adj_train, features_train, y_train = generate_new_batches(gs_adj,features,y,idx_train,1,shift,args.batch_size,device,test_sample)
                        adj_val, features_val, y_val = generate_new_batches(gs_adj,features,y,idx_val,1,shift,args.batch_size,device,test_sample)
                        adj_test, features_test, y_test = generate_new_batches(gs_adj,features,y,[test_sample],1,shift,args.batch_size,device,-1)

                n_train_batches = ceil(len(idx_train)/args.batch_size)
                n_val_batches = 1
                n_test_batches = 1

                #-------------------- Training
                # Model and optimizer
                stop = False

                while(not stop):#
                    if(args.model=="LSTM"):
                        model = LSTM(nfeat=lstm_features, nhid=args.hidden, n_nodes=n_nodes, window=args.window, dropout=args.dropout,batch_size = args.batch_size, recur=args.recur).to(device)
                    elif(args.model=="MPNN_LSTM"):
                        model = MPNN_LSTM(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window, dropout=args.dropout).to(device)
                    elif(args.model=="MPNN"):
                        if infra:
                            model = MPNN(nfeat=nfeat, nfeat_iris=nfeat_iris, nhid=args.hidden, nout=1, dropout=args.dropout, n_nodes=n_nodes, n_nodes_iris=n_nodes_iris, iris=True).to(device)
                        else:
                            model = MPNN(nfeat=nfeat, nfeat_iris=1, nhid=args.hidden, nout=1, dropout=args.dropout, n_nodes=n_nodes, n_nodes_iris=1, iris=False).to(device)

                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

                    #------------------- Train
                    best_val_acc_cl = 0
                    best_val_acc = 1e8
                    val_among_epochs = []
                    train_among_epochs = []

                    stop = False

                    for epoch in range(args.epochs):
                        start = time.time()

                        model.train()
                        train_loss = AverageMeter()

                        # Train for one epoch
                        for batch in range(n_train_batches):
                            if infra:
                                b_size = int(len(y_train[batch])/94)
                                # print(len(y_train[batch]))
                                # print(b_size)

                                irises_ind_batch_train = irises_ind_batch[:b_size]

                                # irises_ind_batch = torch.tensor(irises_ind_batch).to(device)
                                #import ipdb; ipdb.set_trace()
                                output, loss = train_iris(epoch, adj_train[batch], features_train[batch], y_train[batch], adj_train_iris[batch], features_train_iris[batch], irises_ind_batch_train)
                            else:
                                output, loss = train(epoch, adj_train[batch], features_train[batch], y_train[batch])

                            train_loss.update(loss.data.item(), output.size(0))

                        # Evaluate on validation set
                        model.eval()

                        #for i in range(n_val_batches):
                        # irises_ind_batch = []
                        # for b in range(b_size):
                        #     irises_ind_b = [(b*94)+ir for ir in irises_ind]
                        #     irises_ind_batch.append(torch.tensor(irises_ind_b).to(device))
                        # irises_ind_val = 1 * [irises_ind_batch]
                        # output, val_loss = test(adj_val[0], features_val[0], y_val[0], adj_val_iris[0], features_val_iris[0], irises_ind_batch[:5], False)
                        # val_loss = float(val_loss.detach().cpu().numpy())

                        if infra:
                            output, val_loss = test_iris(adj_val[0], features_val[0], y_val[0], adj_val_iris[0], features_val_iris[0], irises_ind_batch[:5])
                        else:
                            output, val_loss = test(adj_val[0], features_val[0], y_val[0])
                        val_loss = float(val_loss.detach().cpu().numpy())


                        # Print results
                        if(epoch%50==0):
                            #print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg), "time=", "{:.5f}".format(time.time() - start))
                            print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),"val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))

                        train_among_epochs.append(train_loss.avg)
                        val_among_epochs.append(val_loss)

                        # if classification:
                        #     train_among_epochs_acc.append(train_acc.avg)
                        #     val_among_epochs_acc.append(val_acc)

                        #print(int(val_loss.detach().cpu().numpy()))

                        if(epoch<30 and epoch>10):
                            if(len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1 ):
                                #stuck= True
                                stop = False
                                break

                        if( epoch>args.early_stop):
                            if(len(set([round(val_e) for val_e in val_among_epochs[-50:]])) == 1):#
                                print("break")
                                #stop = True
                                break

                        stop = True

                        if val_loss < best_val_acc:
                            best_val_acc = val_loss
                            torch.save({
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                            }, 'model_best.pth.tar')

                        scheduler.step(val_loss)


                print("validation")
                #print(best_val_acc)
                #---------------- Testing
                test_loss = AverageMeter()
                # if classification:
                #     test_acc = AverageMeter()

                #print("Loading checkpoint!")
                checkpoint = torch.load('model_best.pth.tar')
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                model.eval()

                #error= 0
                #for batch in range(n_test_batches):
                # irises_ind_test = 1 * [irises_ind]
                # output, loss = test(adj_test[0], features_test[0], y_test[0], adj_test_iris[0], features_test_iris[0], irises_ind_test, True)

                if infra:
                    irises_ind_test = 1 * [irises_ind]
                    output, loss = test_iris(adj_test[0], features_test[0], y_test[0], adj_test_iris[0], features_test_iris[0], irises_ind_test)
                else:
                    output, loss = test(adj_test[0], features_test[0], y_test[0])

                if(args.model=="LSTM"):
                    o = output.view(-1).cpu().detach().numpy()
                    l = y_test[0].view(-1).cpu().numpy()
                else:
                    o = output.cpu().detach().numpy()
                    l = y_test[0].cpu().numpy()

                # average error per region
                # Print results
                ferr.write("Output:"+str(o)+"\n")
                ferr.write("Ground:"+str(l)+"\n")
                rel_error = np.sum((abs(l-o))/l)
                ferr.write("Relative:"+str(rel_error)+"\n\n")

                error = np.sum(abs(o-l))/n_nodes
                print("test error=", "{:.5f}".format(error))
                result.append(error)


            print("{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+",{:.5f}".format(np.sum(labels.iloc[:,args.start_exp:test_sample].mean(1))))
            all_results.append(np.mean(result))
            fw.write(str(args.model)+","+str(shift)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+"\n")

        all_res_mean = np.mean(all_results)
        all_res_std = np.std(all_results)
        fw.write("Mean: "+",{:.5f}".format(all_res_mean)+",{:.5f}".format(all_res_std)+"\n")
        f.close()
        # all_results_iter.append(all_res_mean)
        #
        # fw.write("All mean: "+",{:.5f}".format(np.mean(all_results_iter))+",{:.5f}".format(np.std(all_results_iter))+"\n")
    fw.close()
