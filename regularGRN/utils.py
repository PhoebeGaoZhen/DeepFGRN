import numpy as np
import pandas as pd
from math import log2
import re
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve



def transform_data_single_net(train_data):

    featuretf_exp = []
    featuretarget_exp = []
    net_tf_s = []
    net_tf_t = []
    net_target_s = []
    net_target_t = []
    label_ = []
    position = []
    for i in range(len(train_data)):
        featuretf_exp.append(train_data[i][0])
        featuretarget_exp.append(train_data[i][1])
        net_tf_s.append(train_data[i][2])
        net_tf_t.append(train_data[i][3])
        net_target_s.append(train_data[i][4])
        net_target_t.append(train_data[i][5])
        label_.append(train_data[i][6])
        position.append(train_data[i][7])

    featuretf_exp = np.array(featuretf_exp)
    featuretarget_exp = np.array(featuretarget_exp)
    net_tf_s = np.array(net_tf_s)
    net_tf_t = np.array(net_tf_t)
    net_target_s = np.array(net_target_s)
    net_target_t = np.array(net_target_t)

    dataX_tf = featuretf_exp[:,np.newaxis,:]
    dataX_target = featuretarget_exp[:,np.newaxis,:]
    net_tf_s = net_tf_s[:,np.newaxis,:]
    net_tf_t = net_tf_t[:,np.newaxis,:]
    net_target_s = net_target_s[:,np.newaxis,:]
    net_target_t = net_target_t[:,np.newaxis,:]
    print("the shape of dataX_tf: ",dataX_tf.shape)
    print("the shape of dataX_target: ",dataX_target.shape)
    print("the shape of net_tf_s: ", net_tf_s.shape)

    label_ = np.array(label_)

    labelY = to_categorical(label_,2)

    position = np.array(position)


    return dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t, labelY, position

def two_scores(y_test,y_pred,th=0.5):
    y_predlabel = [(0 if item<th else 1) for item in y_pred]
    tn,fp,fn,tp = confusion_matrix(y_test,y_predlabel).flatten()
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test,y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)
    return [Recall,SPE,Precision,F1,MCC,Acc,AUC,AUPR]


def readRawData_gene100(path, pathts):

    gene10 = pd.read_csv(path, sep='\t', header=None)
    gene10 = np.array(gene10)

    gene10_ts = pd.read_csv(pathts, sep='\t')
    gene10_ts = np.array(gene10_ts)
    gene10_ts = np.transpose(gene10_ts)

    return gene10, gene10_ts


def createGRN_gene100(gene10, gene10_ts):
    rowNumber = []
    colNumber = []
    for i in range(len(gene10)):
        row = gene10[i][0]
        rownum = re.findall("\d+",row)
        rownumber = int(np.array(rownum))
        rowNumber.append(rownumber)

        col = gene10[i][1]
        colnum = re.findall("\d+",col)
        colnumber = int(np.array(colnum))
        colNumber.append(colnumber)

    geneNetwork = np.zeros((gene10_ts.shape[0],gene10_ts.shape[0]))
    for i in range(len(rowNumber)):
        r = rowNumber[i]-1
        c = colNumber[i]-1
        geneNetwork[r][c] = 1
    return geneNetwork


def createSamples_gene100_1(gene10_ts, GRN_embedding_s, GRN_embedding_t,geneNetwork):
    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []

    sample_cold_pos_1_net_tf_s = []
    sample_cold_pos_1_net_tf_t = []
    sample_cold_pos_1_net_target_s = []
    sample_cold_pos_1_net_target_t = []
    sample_cold_pos_0_net_tf_s = []
    sample_cold_pos_0_net_tf_t = []
    sample_cold_pos_0_net_target_s = []
    sample_cold_pos_0_net_target_t = []

    labels_pos_1 = []
    labels_neg_0 = []
    positive_1_position = []
    negative_0_position = []
    for i in range(100):
        for j in range(100):
            tf1 = gene10_ts[i]  # (24,)
            tf_s = GRN_embedding_s[i]  # (32,)
            tf_t = GRN_embedding_t[i]  # (32,)
            target1 = gene10_ts[j]  # (24,)
            target_s = GRN_embedding_s[j]  # (32,)
            target_t = GRN_embedding_t[j]  # (32,)

            label = int(geneNetwork[i][j])

            if label == 1:

                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))

            elif label == 0:
                sample_cold_neg_0_tf.append(tf1)
                sample_cold_neg_0_target.append(target1)

                sample_cold_pos_0_net_tf_s.append(tf_s)
                sample_cold_pos_0_net_tf_t.append(tf_t)
                sample_cold_pos_0_net_target_s.append(target_s)
                sample_cold_pos_0_net_target_t.append(target_t)

                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)

    # bind  feature (sample) and  label
    positive1_data = list(
        zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, sample_cold_pos_1_net_tf_s, sample_cold_pos_1_net_tf_t,
            sample_cold_pos_1_net_target_s, sample_cold_pos_1_net_target_t, labels_pos_1, positive_1_position))  # len
    negative0_data = list(
        zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, sample_cold_pos_0_net_tf_s, sample_cold_pos_0_net_tf_t,
            sample_cold_pos_0_net_target_s, sample_cold_pos_0_net_target_t, labels_neg_0, negative_0_position))  # len

    feature_size_tf = sample_cold_pos_1_tf[0].shape[0]
    feature_size_target = sample_cold_pos_1_target[0].shape[0]
    feature_size_tf_nets = sample_cold_pos_1_net_tf_s[0].shape[0]

    return positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets





def minmaxstandard(rawdata):
    Standard_data = MinMaxScaler().fit_transform(rawdata)
    return Standard_data

def read_graph_DREAM4(path,pathnode):
    nodes = set()
    nodes_s = set()
    egs = []
    graph = [{}, {}]

    egs_nodes = pd.read_csv(path, sep='\t', header=None)

    for i in range(egs_nodes.shape[0]):

        source_node = egs_nodes[0][i]
        target_node = egs_nodes[1][i]
        source_node = [i for i in source_node if i.isnumeric()]
        source_node = ''.join(source_node)
        target_node = [i for i in target_node if i.isnumeric()]
        target_node = ''.join(target_node)

        source_node = int(source_node)-1
        target_node = int(target_node)-1

        nodes.add(source_node)
        nodes.add(target_node)
        nodes_s.add(source_node)
        egs.append([source_node, target_node])

        if source_node not in graph[0]:
            graph[0][source_node] = []
        if target_node not in graph[1]:
            graph[1][target_node] = []

        graph[0][source_node].append(target_node)
        graph[1][target_node].append(source_node)

    genename = pd.read_csv(pathnode, sep='\t', engine='python')
    nodes = genename.ids
    n_node = len(nodes)
    return graph, n_node, list(nodes), list(nodes_s), egs
