import torch
import networkx as nx
# import igraph as ig
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import ceil
import glob
import unidecode
from datetime import date, timedelta
import itertools
from tqdm import tqdm
from sklearn import preprocessing
import ipdb
import os
import holidays

def read_meta_datasets(window, sdate, edate, infra, vac, pos_cases, holiday):
    meta_labs = []
    meta_graphs = []
    meta_features = []
    meta_y = []
    meta_vac = []

    meta_labs_iris = []
    meta_graphs_iris = []
    meta_features_iris = []
    meta_y_iris = []

    #---------------- France

    dropped_iris = []
    kept_iris = []

    if infra:
        # labels_iris = pd.read_csv("../data/infra_dep_data_labels_"+sdate.strftime("%Y-%m-%d")+"_"+edate.strftime("%Y-%m-%d")+".csv", delimiter=",", on_bad_lines='skip')
        labels_iris = pd.read_csv("../data/infra_dep_data_labels_2021-06-27.csv", delimiter=",")
        # labels_iris = pd.read_csv("../data/infra_dep_data_labels_2020-12-27_2021-03-27.csv", delimiter=",")
        # labels_iris = pd.read_csv("../data/epci_data_labels_2021-03-27.csv", delimiter=",")
        #del labels["id"]
        # labels_iris = labels_iris.fillna('')
        labels_iris = labels_iris.replace(r'^s*$', float('NaN'), regex=True)
        for index, row in labels_iris.iterrows():
            if row.isnull().any():
                # import ipdb; ipdb.set_trace()
                dropped_iris.append(row['name'])

        labels_iris = labels_iris.dropna(how='any', axis=0)
        kept_iris = labels_iris["name"]
        print("kept:"+str(len(kept_iris)))
        labels_iris = labels_iris.set_index("name")

        le = preprocessing.LabelEncoder()

        flat_list = [item for sublist in labels_iris.values.tolist() for item in sublist]
        le.fit(flat_list)
        for col in labels_iris.columns:
            labels_iris[col] = le.transform(labels_iris[col])


    #labels = pd.read_csv("../data/france_labels_"+sdate.strftime("%Y-%m-%d")+"_"+edate.strftime("%Y-%m-%d")+".csv")
    # labels = pd.read_csv("../data/france_labels_2020-12-27_2021-03-27.csv")
    labels = pd.read_csv("../data/hosp_france_labels_2020-12-27_2021-06-27.csv")
    #del labels["id"]
    deps = labels["name"].tolist()
    labels = labels.set_index("name")

    # sdate = date(2020, 3, 10)
    # edate = date(2020, 5, 12)

    #--- series of graphs and their respective dates
    delta = edate - sdate
    if infra:
        # dates_iris = []
        # for i in range(delta.days+1):
        #     first_date = sdate + timedelta(days=i)
        #     end_date = sdate + timedelta(days=i) + timedelta(days=6)
        #     if end_date<edate:
        #         dates_iris.append(first_date.strftime("%Y-%m-%d")+"-"+end_date.strftime("%Y-%m-%d"))

        dates_iris = [sdate + timedelta(days=i) for i in range(delta.days+1)]
        dates_iris = [str(date) for date in dates_iris]

    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]

    labels = labels.loc[:,dates]    #labels.sum(1).values>10

    if infra:
        labels_iris = labels_iris.loc[:,dates_iris]
        Gs, Gs_iris, irises_in, nodes_iris = generate_graphs_tmp(dates, dates_iris, infra, deps, kept_iris, "FR")
    else:
        dates_iris = []
        labels_iris = []
        Gs = generate_graphs_tmp(dates, dates_iris, infra, deps, kept_iris, "FR")

    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]
    labels = labels.loc[list(Gs[0].nodes()),:]

    # ipdb.set_trace()
    if infra:
        gs_adj_iris = [kgs.T for kgs in Gs_iris]
        labels_iris = labels_iris.loc[nodes_iris,:]

    vac_labels = None
    if vac:
        # if infra:
        #     vac_labels = pd.read_csv("../metadata/infra_vaccination_1dose_france_deps_cum_pop.csv")
        #     vac_labels = vac_labels.set_index("name")
        # else:
        vac_labels = pd.read_csv("../metadata/vaccination_1dose_france_deps_cum_pop_big.csv")
        vac_labels = vac_labels.set_index("dep")

    meta_labs.append(labels)
    meta_graphs.append(gs_adj)

    if infra:
        meta_labs_iris.append(labels_iris)
        meta_graphs_iris.append(gs_adj_iris)
        features, features_iris = generate_new_features_plus_iris(Gs, labels, Gs_iris, labels_iris, dates, dates_iris, nodes_iris, window, vac_labels=vac_labels, pos_cases=pos_cases)
        meta_features_iris.append(features_iris)
    else:
        features = generate_new_features(Gs, labels, dates, window, vac_labels=vac_labels, pos_cases=pos_cases, holiday=holiday)

    meta_features.append(features)

    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    meta_y.append(y)

    if infra:
        y_iris = list()
        for i,G in enumerate(Gs_iris):
            y_iris.append(list())
            for node in nodes_iris:
                y_iris[i].append(labels_iris.loc[node, dates_iris[i]])
        meta_y_iris.append(y_iris)

        return meta_labs, meta_labs_iris, meta_graphs, meta_graphs_iris, meta_features, meta_features_iris, meta_y, meta_y_iris, irises_in, deps, nodes_iris
    else:
        return meta_labs, meta_graphs, meta_features, meta_y


def generate_graphs_tmp(dates, dates_iris, infra, deps, kept_iris, country):

    dep_pops = {}
    dep_names = {}
    # df_com_pop = pd.read_csv('../metadata/Population_des_communes_Ile-de-France_INSEE.csv')
    df_dep_pop = pd.read_csv('../metadata/departements_population.csv', delimiter="\t")
    for index, row in df_dep_pop.iterrows():
        dep_code = str(row['INSEE Dept. No.'])
        dep_name =  unidecode.unidecode(str(row['Department'])).lower().replace("-","_").replace(" ","_")
        # com_pops[com] = pop
        # pop_row = dep_data[dep_data['Department']==deps[dep]]
        # import ipdb; ipdb.set_trace()
        pop_val = int(str(row['Legal population in 2013']).replace(",",""))
        dep_pops[dep_code] = pop_val
        dep_names[dep_code] = dep_name

    #labels = pd.read_csv("../data/france_labels_2020-12-27_2021-06-27.csv")
    labels = pd.read_csv("../data/hosp_france_labels_2020-12-27_2021-06-27.csv")
    #del labels["id"]
    deps = labels["name"].tolist()

    if infra:
        pop_irises = {}
        irises = {}
        irises_in = {}

        # if infra=="iris":
        df_iris_pop = pd.read_excel('../metadata/base-ic-evol-struct-pop-2016.xls',sheet_name='IRIS',header=5)
        for index, row in df_iris_pop.iterrows():
            dep = str(row['DEP'])
            code_iris = str(row['IRIS'])
            if len(code_iris)==8:
                code_iris = "0"+code_iris
            pop_iris = int(row['P16_POP'])
            # com = int(row['COM'])
            lib_iris = unidecode.unidecode(str(row['LIBIRIS']).lower().replace("-","_").replace(" ","_").replace("_st_","_saint_"))
            pop_irises[code_iris] = float(pop_iris)/dep_pops[dep]
            irises[code_iris] = unidecode.unidecode(lib_iris.lower().replace("-","_").replace(" ","_"))
            dep_name = unidecode.unidecode(dep_names[dep].strip().lower().replace("-","_").replace(" ","_"))
            if dep_name in deps:
                irises_in[code_iris] = dep_name
        # elif infra=="epci":

        # df_epci_pop = pd.read_excel('../metadata/Liste des groupements - France entière.xlsx')
        # for index, row in df_epci_pop.iterrows():
        #     epci_code = str(row['N° SIREN']).strip().replace("\n","")
        #     epci_name = str(row['Nom du groupement']).strip().replace("\n","")
        #
        #     dep = str(row['Département siège']).split(" - ")
        #     dep_code = str(dep[0]).strip().replace("\n","")
        #     # dep_code = str(row['dep_code']).strip()
        #     #dep_lib =
        #
        #     # pop_val = int(str(row['pop']).replace(",","").strip())
        #     pop_val = int(str(row['Population']).replace(",","").strip())
        #     pop_irises[epci_code] = pop_val
        #
        #     # epcis_pops[epci_code] = float(pop_val)/dep_pops[dep_code]
        #     irises[epci_code] = unidecode.unidecode(epci_name.lower().replace("-","_").replace(" ","_"))
        #     dep_name = unidecode.unidecode(dep_names[dep_code].strip().lower().replace("-","_").replace(" ","_"))
        #     if dep_name in deps:
        #         irises_in[epci_code] = dep_name


        Gs_iris = []
        data = []
        for dep in deps:
            print(dep)
            df = pd.read_csv("C:/Users/skian/Documents/ihu-covid/data/infra_graphs_big/"+dep+"/infra_"+country+"_"+dates_iris[0]+".csv",header=None, delimiter=",", on_bad_lines='skip', dtype={0: int, 1:int, 2:float})
            # df = pd.read_csv("C:/Users/skian/Documents/ihu-covid/data/epci_graphs/"+dep+"/epci_"+country+"_"+dates_iris[0]+".csv",header=None, delimiter=",", on_bad_lines='skip', dtype={0: int, 1:int, 2:float})
            df = df[df[0].isin(kept_iris)]
            df = df[df[1].isin(kept_iris)]
            data.append(df)

        d = pd.concat(data, ignore_index=True)
        G_iris_0 = nx.DiGraph()
        nodes_iris = list(set(d[0].unique()).union(set(d[1].unique())))
        print(nodes_iris)
        print("nodes iris: "+str(len(nodes_iris)))
        print("irises_in: "+str(len(irises_in)))
        # G_iris_0.add_nodes_from(nodes_0)
        # for row in df.iterrows():
        #     G_iris_0.add_edge(row[1][0], row[1][1], weight=row[1][2])

        for date in tqdm(dates_iris):
            data = []
            # fw = open("C:/Users/skian/Documents/ihu-covid/data/infra_graphs/all/infra_"+country+"_"+date+".csv","w")
            for dep in deps:
                # df = pd.read_csv("../data/infra_graphs/"+dep+"/infra_"+country+"_"+date+".csv",header=None, delimiter=",", on_bad_lines='skip')
                df = pd.read_csv("C:/Users/skian/Documents/ihu-covid/data/infra_graphs_big/"+dep+"/infra_"+country+"_"+date+".csv",header=None, delimiter=",", on_bad_lines='skip', dtype={0: int, 1:int, 2:float})
                # df = pd.read_csv("C:/Users/skian/Documents/ihu-covid/data/epci_graphs/"+dep+"/epci_"+country+"_"+date+".csv",header=None, delimiter=",", on_bad_lines='skip', dtype={0: int, 1:int, 2:float})
                # import ipdb; ipdb.set_trace()
                df = df[df[0].isin(kept_iris)]
                df = df[df[1].isin(kept_iris)]
                # df = df.reset_index(drop=True)
                data.append(df)
                # f = open("C:/Users/skian/Documents/ihu-covid/data/infra_graphs/"+dep+"/infra_"+country+"_"+date+".csv","r")
                # s = f.read()
                # f.close()
                # fw.write(s+"\n")
            # fw.close()

            d = pd.concat(data, ignore_index=True)
            d.columns = ['name1', 'name2', 'weight']
            # # G_iris = nx.from_pandas_edgelist(d, source=0, target=1, edge_attr=2, create_using=nx.DiGraph)
            # G_iris = ig.Graph.TupleList(d.itertuples(index=False), directed=True, weights=True)
            # G_iris = nx.DiGraph()
            nodes = set(d['name1'].unique()).union(set(d['name2'].unique()))
            # G_iris.add_nodes_from(nodes)
            nodes = sorted(nodes)
            # nodes = [(i, nodes[i]) for i in range(len(nodes))]
            nodes = {nodes[i]: i for i in range(len(nodes))}
            d['name1'] = d['name1'].apply(lambda x: nodes[x])
            d['name2'] = d['name2'].apply(lambda x: nodes[x])

            # for i in tqdm(range(len(nodes))):
            # #     print(i)
            #     d = d.replace(nodes[i][1], nodes[i][0])
            G_iris = sp.coo_matrix((d.iloc[:,2], (d.iloc[:,0], d.iloc[:,1])), shape=(len(nodes), len(nodes)))
            # import ipdb; ipdb.set_trace()
            nodes_iris = list(nodes.keys())
            # G_iris.add_nodes_from(nodes)
            # #
            # for row in d.iterrows():
            #     G_iris.add_edge(row[1][0], row[1][1], weight=row[1][2])
            # print("Number of edges:"+str(G_iris.number_of_edges()))
            # import ipdb; ipdb.set_trace()
            # G_iris = nx.read_edgelist("C:/Users/skian/Documents/ihu-covid/data/infra_graphs/all/infra_"+country+"_"+date+".csv", delimiter=",", create_using=nx.Digraph, nodetype=int, data=(("weight", float),))
            Gs_iris.append(G_iris)


    Gs = []
    for date in dates:
        # else:
        d = pd.read_csv("../data/graphs_big/vac_"+country+"_"+date+".csv",header=None)

        G = nx.DiGraph()
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        G.add_nodes_from(nodes)

        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])

        # if vac:
        #     graph_vac_labels = vac_labels[date]
        #     # import ipdb; ipdb.set_trace()
        #     vac_dict = {}
        #     vac_list = graph_vac_labels.tolist()
        #     for i, node in enumerate(G.nodes()):
        #         vac_dict[node] = vac_list[i]
        #     nx.set_node_attributes(G, "vac_label", vac_dict)

        Gs.append(G)


    if infra:
        return Gs, Gs_iris, irises_in, nodes_iris
    return Gs


def generate_new_features(Gs, labels, dates, window=7, scaled=False, vac_labels=None, pos_cases=False, holiday=False):
    """
    Generate node features
    Features[1] contains the features corresponding to y[1]
    e.g. if window = 7, features[7] = day0:day6, y[7] = day7
    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()

    labs = labels.copy()
    nodes = Gs[0].nodes()

    if pos_cases:
        pos_labels = pd.read_csv("../data/france_labels_2020-12-27_2021-06-27.csv")
        pos_labels = pos_labels.set_index("name")

    fr_holidays = holidays.country_holidays('FR')

    #print(n_departments)
    for idx, G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region
        if holiday:
            H = np.zeros([G.number_of_nodes(),2*window])

        if vac_labels is not None:
            if pos_cases:
                H = np.zeros([G.number_of_nodes(),3*window])
            else:
                H = np.zeros([G.number_of_nodes(),2*window]) #+3+n_departments])#])#])
        else:
            if holiday:
                H = np.zeros([G.number_of_nodes(),2*window])
            else:
                H = np.zeros([G.number_of_nodes(),window]) #+3+n_departments])#])#])
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1)+1

        if vac_labels is not None:
            vac_me = vac_labels.loc[:, dates[:(idx)]].mean(1)
            vac_sd = vac_labels.loc[:, dates[:(idx)]].std(1)+1

        if pos_cases:
            pos_me = pos_labels.loc[:, dates[:(idx)]].mean(1)
            pos_sd = pos_labels.loc[:, dates[:(idx)]].std(1)+1

        ### enumarate because H[i] and labs[node] are not aligned
        for i, node in enumerate(G.nodes()):
            #---- Past cases
            if(idx < window):# idx-1 goes before the start of the labels
                if(scaled):
                    #me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H[i,(window-idx):(window)] = (labs.loc[node, dates[0:(idx)]] - me[node])/sd[node]
                    if vac_labels is not None:
                        H[i,(window-idx)+window:(2*window)] = (vac_labels.loc[node, dates[0:(idx)]] - vac_me[node])/vac_sd[node]
                    if pos_cases:
                        H[i,(window-idx)+(2*window):(3*window)] = (pos_labels.loc[node, dates[0:(idx)]] - pos_me[node])/pos_sd[node]


                else:
                    H[i,(window-idx):(window)] = labs.loc[node, dates[0:(idx)]]
                    if vac_labels is not None:
                        H[i,(window-idx)+window:(2*window)] = vac_labels.loc[node, dates[0:(idx)]]
                    if pos_cases:
                        H[i,(window-idx)+(2*window):(3*window)] = pos_labels.loc[node, dates[0:(idx)]]

            elif idx >= window:
                if(scaled):
                    H[i,0:(window)] = (labs.loc[node, dates[(idx-window):(idx)]] - me[node])/sd[node]
                    if vac_labels is not None:
                        H[i,window:(2*window)] = (vac_labels.loc[node, dates[(idx-window):(idx)]] - vac_me[node])/vac_sd[node]
                    if pos_cases:
                        H[i,2*window:] = (pos_labels.loc[node, dates[(idx-window):(idx)]] - pos_me[node])/ pos_sd[node]

                else:
                    H[i,0:(window)] = labs.loc[node, dates[(idx-window):(idx)]]
                    if vac_labels is not None:
                        H[i,window:(2*window)] = vac_labels.loc[node, dates[(idx-window):(idx)]]
                    if pos_cases:
                        H[i,2*window:] = pos_labels.loc[node, dates[(idx-window):(idx)]]

                if holiday:
                    da = dates[(idx-window):(idx)]
                    hol = np.zeros(len(da))
                    for ind_d, d in enumerate(da):
                        ds = d.split("-")
                        d = date(int(ds[0]), int(ds[1]), int(ds[2]))
                        if d.weekday()==6 or d in fr_holidays:
                            hol[ind_d] = 1
                    H[i,window:(2*window)] = hol

        features.append(H)

    return features


def generate_new_features_plus_iris(Gs, labels, Gs_iris, labels_iris, dates, dates_iris, nodes_iris, window=7, scaled=False, vac_labels=None, pos_cases=False, holiday=False):
    """
    Generate node features
    Features[1] contains the features corresponding to y[1]
    e.g. if window = 7, features[7] = day0:day6, y[7] = day7
    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()

    labs = labels.copy()
    nodes = Gs[0].nodes()

    if pos_cases:
        pos_labels = pd.read_csv("../data/france_labels_2020-12-27_2021-06-27.csv")
        pos_labels = pos_labels.set_index("name")

    #--- one hot encoded the region
    #departments_name_to_id = dict()
    #for node in nodes:
    #    departments_name_to_id[node] = len(departments_name_to_id)

    #n_departments = len(departments_name_to_id)

    labs_iris = labels_iris.copy()
    features_iris = []

    #print(n_departments)
    for idx, G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region

        if vac_labels is not None:
            if pos_cases:
                H = np.zeros([G.number_of_nodes(),3*window])
            else:
                H = np.zeros([G.number_of_nodes(),2*window]) #+3+n_departments])#])#])
        else:
            H = np.zeros([G.number_of_nodes(),window]) #+3+n_departments])#])#])
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1)+1

        if vac_labels is not None:
            vac_me = vac_labels.loc[:, dates[:(idx)]].mean(1)
            vac_sd = vac_labels.loc[:, dates[:(idx)]].std(1)+1

        if pos_cases:
            pos_me = pos_labels.loc[:, dates[:(idx)]].mean(1)
            pos_sd = pos_labels.loc[:, dates[:(idx)]].std(1)+1

        ### enumarate because H[i] and labs[node] are not aligned
        for i, node in enumerate(G.nodes()):
            #---- Past cases
            if(idx < window):# idx-1 goes before the start of the labels
                if(scaled):
                    #me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H[i,(window-idx):(window)] = (labs.loc[node, dates[0:(idx)]] - me[node])/sd[node]
                    if vac_labels is not None:
                        H[i,(window-idx)+window:(2*window)] = (vac_labels.loc[node, dates[0:(idx)]] - vac_me[node])/vac_sd[node]
                    if pos_labels is not None:
                        H[i,(window-idx)+(2*window):(3*window)] = (pos_labels.loc[node, dates[0:(idx)]] - pos_me[node])/pos_sd[node]
                else:
                    H[i,(window-idx):(window)] = labs.loc[node, dates[0:(idx)]]
                    if vac_labels is not None:
                        H[i,(window-idx)+window:(2*window)] = vac_labels.loc[node, dates[0:(idx)]]
                    if pos_cases:
                        H[i,(window-idx)+(2*window):(3*window)] = pos_labels.loc[node, dates[0:(idx)]]

            elif idx >= window:
                if(scaled):
                    H[i,0:(window)] = (labs.loc[node, dates[(idx-window):(idx)]] - me[node])/ sd[node]
                    if vac_labels is not None:
                        H[i,window:(2*window)] = (vac_labels.loc[node, dates[(idx-window):(idx)]] - vac_me[node])/ vac_sd[node]
                    if pos_labels is not None:
                        H[i,2*window:] = (pos_labels.loc[node, dates[(idx-window):(idx)]] - pos_me[node])/ pos_sd[node]

                else:
                    H[i,0:(window)] = labs.loc[node, dates[(idx-window):(idx)]]
                    if vac_labels is not None:
                        H[i,window:(2*window)] = vac_labels.loc[node, dates[(idx-window):(idx)]]
                    if pos_cases:
                        H[i,2*window:] = pos_labels.loc[node, dates[(idx-window):(idx)]]


        features.append(H)

        H_iris = np.zeros([len(nodes_iris), window]) #+3+n_departments])#])#])
        me = labs_iris.loc[:, dates_iris[:(idx)]].mean(1)
        sd = labs_iris.loc[:, dates_iris[:(idx)]].std(1)+1

        ### enumerate because H[i] and labs[node] are not aligned
        # for i, node in enumerate(G.nodes()):
        for i, node in enumerate(nodes_iris):
            #---- Past cases
            if(idx < window):# idx-1 goes before the start of the labels
                if(scaled):
                    #me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H_iris[i,(window-idx):(window)] = (labs_iris.loc[node, dates_iris[0:(idx)]] - me[node])/sd[node]
                else:
                    H_iris[i,(window-idx):(window)] = labs_iris.loc[node, dates_iris[0:(idx)]]

            elif idx >= window:
                if(scaled):
                    H_iris[i,0:(window)] = (labs_iris.loc[node, dates_iris[(idx-window):(idx)]] - me[node])/sd[node]
                else:
                    H_iris[i,0:(window)] = labs_iris.loc[node, dates_iris[(idx-window):(idx)]]

        features_iris.append(H_iris)


    # for idx, G in enumerate(Gs_iris):
    #     #  Features = population, coordinates, d past cases, one hot region
    #
    #     H_iris = np.zeros([G.shape[0], window]) #+3+n_departments])#])#])
    #     me = labs_iris.loc[:, dates_iris[:(idx)]].mean(1)
    #     sd = labs_iris.loc[:, dates_iris[:(idx)]].std(1)+1
    #
    #     ### enumerate because H[i] and labs[node] are not aligned
    #     # for i, node in enumerate(G.nodes()):
    #     for i, node in enumerate(nodes_iris):
    #         #---- Past cases
    #         if(idx < window):# idx-1 goes before the start of the labels
    #             if(scaled):
    #                 #me = np.mean(labs.loc[node, dates[0:(idx)]]
    #                 H_iris[i,(window-idx):(window)] = (labs_iris.loc[node, dates_iris[0:(idx)]] - me[node])/sd[node]
    #             else:
    #                 H_iris[i,(window-idx):(window)] = labs_iris.loc[node, dates_iris[0:(idx)]]
    #
    #         elif idx >= window:
    #             if(scaled):
    #                 H_iris[i,0:(window)] = (labs_iris.loc[node, dates_iris[(idx-window):(idx)]] - me[node])/sd[node]
    #             else:
    #                 H_iris[i,0:(window)] = labs_iris.loc[node, dates_iris[(idx-window):(idx)]]
    #
    #     features_iris.append(H_iris)

    return features, features_iris



def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample):
    """
    Generate batches for graphs for MPNN
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]
    # n_nodes = Gs[0].number_of_nodes()

    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))

        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)

        #fill the input for each batch
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = idx[j]

            # Feature[10] containes the previous 7 cases of y[10]
            for e2,k in enumerate(range(val-graph_window+1,val+1)):

                adj_tmp.append(Gs[k-1].T)
                # each feature has a size of n_nodes
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]#-features[val-graph_window-1]


            if(test_sample>0):
                #--- val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]

                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]


            else:
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]

        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst


def generate_new_batches_plus_iris(Gs, features, y, Gs_iris, features_iris, y_iris, irises_in, deps, nodes_iris, idx, graph_window, shift, batch_size, device, test_sample):
    """
    Generate batches for graphs for MPNN
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]
    # n_nodes = Gs[0].number_of_nodes()

    adj_lst = list()
    features_lst = list()
    y_lst = list()

    n_nodes_iris = Gs_iris[0].shape[0]
    # n_nodes = Gs_iris[0].number_of_nodes()

    adj_lst_iris = list()
    features_lst_iris = list()
    y_lst_iris = list()

    # import ipdb; ipdb.set_trace()
    # for node in nodes_iris:
    #     s_node = str(node)
    #     if len(s_node)==8:
    #         s_node = "0" + s_node
    #     dep = irises_in[s_node]
    #     # import ipdb; ipdb.set_trace()
    #     # if dep in
    #     irises_ind_tmp.append(deps.index(dep))
    #
    # irises_ind = torch.LongTensor(irises_ind_tmp).to(device)


    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window
        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))
        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)


        n_nodes_iris_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes_iris
        step_iris = n_nodes_iris*graph_window
        adj_tmp_iris = list()
        features_tmp_iris = np.zeros((n_nodes_iris_batch, features_iris[0].shape[1]))
        y_tmp_iris = np.zeros((min(i+batch_size, N)-i)*n_nodes_iris)
        indices_tmp_iris = np.zeros(n_nodes_iris_batch)

        #fill the input for each batch
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = idx[j]

            # Feature[10] containes the previous 7 cases of y[10]
            for e2,k in enumerate(range(val-graph_window+1,val+1)):

                adj_tmp.append(Gs[k-1].T)
                # each feature has a size of n_nodes
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]#-features[val-graph_window-1]

                adj_tmp_iris.append(Gs_iris[k-1].T)
                # each feature has a size of n_nodes
                # try:
                features_tmp_iris[(e1*step_iris+e2*n_nodes_iris):(e1*step_iris+(e2+1)*n_nodes_iris),:] = features_iris[k]#-features[val-graph_window-1]


            if(test_sample>0):
                #--- val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                    y_tmp_iris[(n_nodes_iris*e1):(n_nodes_iris*(e1+1))] = y_iris[val+shift]
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                    y_tmp_iris[(n_nodes_iris*e1):(n_nodes_iris*(e1+1))] = y_iris[val]

            else:
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                y_tmp_iris[(n_nodes_iris*e1):(n_nodes_iris*(e1+1))] = y_iris[val+shift]


        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

        adj_tmp_iris = sp.block_diag(adj_tmp_iris)
        adj_lst_iris.append(sparse_mx_to_torch_sparse_tensor(adj_tmp_iris).to(device))
        features_lst_iris.append(torch.FloatTensor(features_tmp_iris).to(device))
        y_lst_iris.append(torch.FloatTensor(y_tmp_iris).to(device))

        # ## for iris
        #
        # # for i in range(0, N, batch_size):
        # n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes_iris
        # step = n_nodes_iris*graph_window
        #
        # adj_tmp_iris = list()
        # features_tmp_iris = np.zeros((n_nodes_batch, features_iris[0].shape[1]))
        #
        # y_tmp_iris = np.zeros((min(i+batch_size, N)-i)*n_nodes_iris)
        #
        # #fill the input for each batch
        # for e1, j in enumerate(range(i, min(i+batch_size, N) )):
        #     val = idx[j]
        #
        #     # Feature[10] containes the previous 7 cases of y[10]
        #     for e2,k in enumerate(range(val-graph_window+1, val+1)):
        #
        #         adj_tmp_iris.append(Gs_iris[k-1].T)
        #         # each feature has a size of n_nodes
        #         # try:
        #         features_tmp_iris[(e1*step+e2*n_nodes_iris):(e1*step+(e2+1)*n_nodes_iris),:] = features_iris[k]#-features[val-graph_window-1]
        #         # except:
        #         #     import ipdb; ipdb.set_trace()
        #
        #
        #     if(test_sample>0):
        #         #--- val is by construction less than test sample
        #         if(val+shift<test_sample):
        #             y_tmp_iris[(n_nodes_iris*e1):(n_nodes_iris*(e1+1))] = y_iris[val+shift]
        #
        #         else:
        #             y_tmp_iris[(n_nodes_iris*e1):(n_nodes_iris*(e1+1))] = y_iris[val]
        #
        #
        #     else:
        #         y_tmp_iris[(n_nodes_iris*e1):(n_nodes_iris*(e1+1))] = y_iris[val+shift]
        #
        # adj_tmp_iris = sp.block_diag(adj_tmp_iris)
        # adj_lst_iris.append(sparse_mx_to_torch_sparse_tensor(adj_tmp_iris).to(device))
        # features_lst_iris.append(torch.FloatTensor(features_tmp_iris).to(device))
        # y_lst_iris.append(torch.FloatTensor(y_tmp_iris).to(device))




    # import ipdb; ipdb.set_trace()

    return adj_lst, features_lst, y_lst, adj_lst_iris, features_lst_iris, y_lst_iris



def generate_batches_lstm(n_nodes, y, idx, window, shift, batch_size, device,test_sample):
    """
    Generate batches for graphs for the LSTM
    """
    N = len(idx)
    features_lst = list()
    y_lst = list()
    adj_fake = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*n_nodes*1
        #step = n_nodes#*window
        step = n_nodes*1

        adj_tmp = list()
        features_tmp = np.zeros((window, n_nodes_batch))#features.shape[1]))

        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)

        for e1,j in enumerate(range(i, min(i+batch_size, N))):
            val = idx[j]

            # keep the past information from val-window until val-1
            for e2,k in enumerate(range(val-window,val)):

                if(k==0):
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.zeros([n_nodes])#features#[k]
                else:
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.array(y[k])#.reshape([n_nodes,1])#

            if(test_sample>0):
                # val is by construction less than test sample
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]

            else:

                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]

        adj_fake.append(0)

        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append( torch.FloatTensor(y_tmp).to(device))

    return adj_fake, features_lst, y_lst




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
