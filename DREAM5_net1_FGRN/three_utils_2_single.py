import numpy as np
from math import log2
import re
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
# from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
import random


def get_GRN(Ecoli_GRN_known,genename):


    rowNumber = []
    colNumber = []
    regulation_types = []
    TF_name = np.array(Ecoli_GRN_known[0])
    target_name = np.array(Ecoli_GRN_known[1])
    regulation_type = np.array(Ecoli_GRN_known[2])

    genename2 = genename.tolist()

    for i in range(len(TF_name)):
        rowNumber.append(genename2.index(TF_name[i]))

    for i in range(len(target_name)):
        colNumber.append(genename2.index(target_name[i]))

    for i in range(len(regulation_type)):
        regulation_types.append(regulation_type[i])

    num_activator = 0
    num_repressor = 0
    num_unknown = 0
    geneNetwork = np.zeros((len(genename2), len(genename2)))

    for i in range(len(regulation_types)):
        r = rowNumber[i]
        c = colNumber[i]
        if regulation_types[i] == 'activator':
            geneNetwork[r][c] = int(2.0)
            # num_activator += 1
        elif regulation_types[i] == 'repressor':
            geneNetwork[r][c] = int(1.0)
            # num_repressor += 1
        else:
            geneNetwork[r][c] = int(0.0)

    for i in range(geneNetwork.shape[0]):
        for j in range(geneNetwork.shape[0]):
            if geneNetwork[i][j] == 2:
                num_activator += 1
            elif geneNetwork[i][j] == 1:
                num_repressor += 1
            else:
                num_unknown += 1
    return geneNetwork, num_activator, num_repressor, num_unknown

def createGRN_gene100(Ecoli_GRN_known,genename):
    rowNumber = []
    colNumber = []
    for i in range(len(Ecoli_GRN_known)):
        row = Ecoli_GRN_known[i][0]
        rownum = re.findall("\d+",row)
        rownumber = int(np.array(rownum))
        rowNumber.append(rownumber)

        col = Ecoli_GRN_known[i][1]
        colnum = re.findall("\d+",col)
        colnumber = int(np.array(colnum))
        colNumber.append(colnumber)

    geneNetwork = np.zeros((genename.shape[0],genename.shape[0]))
    for i in range(len(rowNumber)):
        r = rowNumber[i]-1
        c = colNumber[i]-1
        geneNetwork[r][c] = 1

    return geneNetwork
def standard(rawdata):
    new_data1 = np.zeros((rawdata.shape[0],rawdata.shape[1]))
    for i in range(rawdata.shape[0]):
        for j in range(rawdata.shape[1]):
            new_data1[i][j] = log2(rawdata[i][j]+1)
    Standard_data = MinMaxScaler().fit_transform(new_data1)
    return Standard_data

def create_samples_concatenate(EXP_cold, Ecoli_GRN):

    EXP_cold = standard(EXP_cold)
    sample_cold_pos_2 = []
    sample_cold_pos_1 = []
    sample_cold_neg_0 = []
    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(2205):
        for j in range(2205):

            tf1 = EXP_cold[i]

            target1 = EXP_cold[j]
            temp = np.hstack((tf1, target1))

            label = int(Ecoli_GRN[i][j])

            if label == 2:
                sample_cold_pos_2.append(temp)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                sample_cold_pos_1.append(temp)
                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                sample_cold_neg_0.append(temp)
                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i,j)
    positive2_data = list(zip(sample_cold_pos_2, labels_pos_2, positive_2_position))
    positive1_data = list(zip(sample_cold_pos_1, labels_pos_1, positive_1_position))
    negative0_data = list(zip(sample_cold_neg_0, labels_neg_0, negative_0_position))


    feature_size = sample_cold_pos_2[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size


def create_samples_single(EXP_cold, Ecoli_GRN):

    EXP_cold = standard(EXP_cold)
    sample_cold_pos_2_tf = []
    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_2_target = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []
    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(2205):
        for j in range(2205):

            tf1 = EXP_cold[i]

            target1 = EXP_cold[j]

            label = int(Ecoli_GRN[i][j])


            if label == 2:
                # sample_cold_pos_2.append(temp)
                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)
                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:
                # sample_cold_pos_1.append(temp)
                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)
                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                # sample_cold_neg_0.append(temp)
                sample_cold_neg_0_tf.append(tf1)
                sample_cold_neg_0_target.append(target1)
                labels_neg_0.append(label)
                negative_0_position.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)

    positive2_data = list(zip(sample_cold_pos_2_tf, sample_cold_pos_2_target, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, labels_neg_0, negative_0_position))  # len


    feature_size_tf = sample_cold_pos_2_tf[0].shape[0]
    feature_size_target = sample_cold_pos_2_target[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target

def create_samples_single_net(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t,num_negative):

    EXP_cold = standard(EXP_cold)
    sample_cold_pos_2_tf = []
    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_2_target = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []

    sample_cold_pos_2_net_tf_s = []
    sample_cold_pos_2_net_tf_t = []
    sample_cold_pos_2_net_target_s = []
    sample_cold_pos_2_net_target_t = []
    sample_cold_pos_1_net_tf_s = []
    sample_cold_pos_1_net_tf_t = []
    sample_cold_pos_1_net_target_s = []
    sample_cold_pos_1_net_target_t = []
    sample_cold_pos_0_net_tf_s = []
    sample_cold_pos_0_net_tf_t = []
    sample_cold_pos_0_net_target_s = []
    sample_cold_pos_0_net_target_t = []

    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_positions = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:

                tf1 = EXP_cold[i]

                tf_s = GRN_embedding_s[i]

                tf_t = GRN_embedding_t[i]

                target1 = EXP_cold[j]

                target_s = GRN_embedding_s[j]

                target_t = GRN_embedding_t[j]

                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)

                sample_cold_pos_2_net_tf_s.append(tf_s)
                sample_cold_pos_2_net_tf_t.append(tf_t)
                sample_cold_pos_2_net_target_s.append(target_s)
                sample_cold_pos_2_net_target_t.append(target_t)

                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:

                tf1 = EXP_cold[i]

                tf_s = GRN_embedding_s[i]

                tf_t = GRN_embedding_t[i]

                target1 = EXP_cold[j]

                target_s = GRN_embedding_s[j]

                target_t = GRN_embedding_t[j]

                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                negative_0_positions.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    random.shuffle(negative_0_positions)
    negative_0_position = negative_0_positions[0:num_negative]
    for k in range(len(negative_0_position)):
        i = negative_0_position[k][0]
        j = negative_0_position[k][1]

        tf1 = EXP_cold[i]

        tf_s = GRN_embedding_s[i]

        tf_t = GRN_embedding_t[i]

        target1 = EXP_cold[j]

        target_s = GRN_embedding_s[j]

        target_t = GRN_embedding_t[j]

        sample_cold_neg_0_tf.append(tf1)
        sample_cold_neg_0_target.append(target1)

        sample_cold_pos_0_net_tf_s.append(tf_s)
        sample_cold_pos_0_net_tf_t.append(tf_t)
        sample_cold_pos_0_net_target_s.append(target_s)
        sample_cold_pos_0_net_target_t.append(target_t)

        labels_neg_0.append(0)


    positive2_data = list(zip(sample_cold_pos_2_tf, sample_cold_pos_2_target, sample_cold_pos_2_net_tf_s, sample_cold_pos_2_net_tf_t, sample_cold_pos_2_net_target_s, sample_cold_pos_2_net_target_t, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, sample_cold_pos_1_net_tf_s, sample_cold_pos_1_net_tf_t, sample_cold_pos_1_net_target_s, sample_cold_pos_1_net_target_t, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, sample_cold_pos_0_net_tf_s, sample_cold_pos_0_net_tf_t, sample_cold_pos_0_net_target_s, sample_cold_pos_0_net_target_t, labels_neg_0, negative_0_position))  # len


    feature_size_tf = sample_cold_pos_2_tf[0].shape[0]
    feature_size_target = sample_cold_pos_2_target[0].shape[0]
    feature_size_tf_nets = sample_cold_pos_2_net_tf_s[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets

def create_samples_dream5(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t,num_negative):

    sample_cold_pos_2_tf = []
    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_2_target = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []

    sample_cold_pos_2_net_tf_s = []
    sample_cold_pos_2_net_tf_t = []
    sample_cold_pos_2_net_target_s = []
    sample_cold_pos_2_net_target_t = []
    sample_cold_pos_1_net_tf_s = []
    sample_cold_pos_1_net_tf_t = []
    sample_cold_pos_1_net_target_s = []
    sample_cold_pos_1_net_target_t = []
    sample_cold_pos_0_net_tf_s = []
    sample_cold_pos_0_net_tf_t = []
    sample_cold_pos_0_net_target_s = []
    sample_cold_pos_0_net_target_t = []

    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_positions = []

    for i in range(EXP_cold.shape[0]):
        for j in range(EXP_cold.shape[0]):

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:

                tf1 = EXP_cold[i]

                tf_s = GRN_embedding_s[i]

                tf_t = GRN_embedding_t[i]

                target1 = EXP_cold[j]

                target_s = GRN_embedding_s[j]

                target_t = GRN_embedding_t[j]


                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)

                sample_cold_pos_2_net_tf_s.append(tf_s)
                sample_cold_pos_2_net_tf_t.append(tf_t)
                sample_cold_pos_2_net_target_s.append(target_s)
                sample_cold_pos_2_net_target_t.append(target_t)

                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:

                tf1 = EXP_cold[i]

                tf_s = GRN_embedding_s[i]

                tf_t = GRN_embedding_t[i]

                target1 = EXP_cold[j]

                target_s = GRN_embedding_s[j]

                target_t = GRN_embedding_t[j]


                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:

                negative_0_positions.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    random.shuffle(negative_0_positions)
    negative_0_position = negative_0_positions[0:num_negative]
    for k in range(len(negative_0_position)):
        i = negative_0_position[k][0]
        j = negative_0_position[k][1]

        tf1 = EXP_cold[i]

        tf_s = GRN_embedding_s[i]

        tf_t = GRN_embedding_t[i]

        target1 = EXP_cold[j]

        target_s = GRN_embedding_s[j]

        target_t = GRN_embedding_t[j]

        sample_cold_neg_0_tf.append(tf1)
        sample_cold_neg_0_target.append(target1)

        sample_cold_pos_0_net_tf_s.append(tf_s)
        sample_cold_pos_0_net_tf_t.append(tf_t)
        sample_cold_pos_0_net_target_s.append(target_s)
        sample_cold_pos_0_net_target_t.append(target_t)

        labels_neg_0.append(0)

    positive2_data = list(zip(sample_cold_pos_2_tf, sample_cold_pos_2_target, sample_cold_pos_2_net_tf_s, sample_cold_pos_2_net_tf_t, sample_cold_pos_2_net_target_s, sample_cold_pos_2_net_target_t, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, sample_cold_pos_1_net_tf_s, sample_cold_pos_1_net_tf_t, sample_cold_pos_1_net_target_s, sample_cold_pos_1_net_target_t, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, sample_cold_pos_0_net_tf_s, sample_cold_pos_0_net_tf_t, sample_cold_pos_0_net_target_s, sample_cold_pos_0_net_target_t, labels_neg_0, negative_0_position))  # len


    feature_size_tf = sample_cold_pos_2_tf[0].shape[0]
    feature_size_target = sample_cold_pos_2_target[0].shape[0]
    feature_size_tf_nets = sample_cold_pos_2_net_tf_s[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets

def create_samples_human_counts(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t,num_negative):

    EXP_cold = standard(EXP_cold)
    sample_cold_pos_2_tf = []
    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_2_target = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []

    sample_cold_pos_2_net_tf_s = []
    sample_cold_pos_2_net_tf_t = []
    sample_cold_pos_2_net_target_s = []
    sample_cold_pos_2_net_target_t = []
    sample_cold_pos_1_net_tf_s = []
    sample_cold_pos_1_net_tf_t = []
    sample_cold_pos_1_net_target_s = []
    sample_cold_pos_1_net_target_t = []
    sample_cold_pos_0_net_tf_s = []
    sample_cold_pos_0_net_tf_t = []
    sample_cold_pos_0_net_target_s = []
    sample_cold_pos_0_net_target_t = []

    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_positions = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):

            label = int(Ecoli_GRN[i][j])
            # print(label)

            if label == 2:

                tf1 = EXP_cold[i]

                tf_s = GRN_embedding_s[i]

                tf_t = GRN_embedding_t[i]

                target1 = EXP_cold[j]

                target_s = GRN_embedding_s[j]

                target_t = GRN_embedding_t[j]

                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)

                sample_cold_pos_2_net_tf_s.append(tf_s)
                sample_cold_pos_2_net_tf_t.append(tf_t)
                sample_cold_pos_2_net_target_s.append(target_s)
                sample_cold_pos_2_net_target_t.append(target_t)

                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:

                tf1 = EXP_cold[i]

                tf_s = GRN_embedding_s[i]

                tf_t = GRN_embedding_t[i]

                target1 = EXP_cold[j]

                target_s = GRN_embedding_s[j]

                target_t = GRN_embedding_t[j]

                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:

                labels_neg_0.append(label)
                negative_0_positions.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    random.shuffle(negative_0_positions)
    negative_0_position = negative_0_positions[0:num_negative]
    for k in range(len(negative_0_position)):
        i = negative_0_position[k][0]
        j = negative_0_position[k][1]


        tf1 = EXP_cold[i]

        tf_s = GRN_embedding_s[i]

        tf_t = GRN_embedding_t[i]

        target1 = EXP_cold[j]

        target_s = GRN_embedding_s[j]

        target_t = GRN_embedding_t[j]

        sample_cold_neg_0_tf.append(tf1)
        sample_cold_neg_0_target.append(target1)

        sample_cold_pos_0_net_tf_s.append(tf_s)
        sample_cold_pos_0_net_tf_t.append(tf_t)
        sample_cold_pos_0_net_target_s.append(target_s)
        sample_cold_pos_0_net_target_t.append(target_t)

        labels_neg_0.append(0)

    positive2_data = list(zip(sample_cold_pos_2_tf, sample_cold_pos_2_target, sample_cold_pos_2_net_tf_s, sample_cold_pos_2_net_tf_t, sample_cold_pos_2_net_target_s, sample_cold_pos_2_net_target_t, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, sample_cold_pos_1_net_tf_s, sample_cold_pos_1_net_tf_t, sample_cold_pos_1_net_target_s, sample_cold_pos_1_net_target_t, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, sample_cold_pos_0_net_tf_s, sample_cold_pos_0_net_tf_t, sample_cold_pos_0_net_target_s, sample_cold_pos_0_net_target_t, labels_neg_0, negative_0_position))  # len


    feature_size_tf = sample_cold_pos_2_tf[0].shape[0]
    feature_size_target = sample_cold_pos_2_target[0].shape[0]
    feature_size_tf_nets = sample_cold_pos_2_net_tf_s[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets

def create_samples_human_FPKM(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t):

    EXP_cold = MinMaxScaler().fit_transform(EXP_cold)
    sample_cold_pos_2_tf = []
    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_2_target = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []

    sample_cold_pos_2_net_tf_s = []
    sample_cold_pos_2_net_tf_t = []
    sample_cold_pos_2_net_target_s = []
    sample_cold_pos_2_net_target_t = []
    sample_cold_pos_1_net_tf_s = []
    sample_cold_pos_1_net_tf_t = []
    sample_cold_pos_1_net_target_s = []
    sample_cold_pos_1_net_target_t = []
    sample_cold_pos_0_net_tf_s = []
    sample_cold_pos_0_net_tf_t = []
    sample_cold_pos_0_net_target_s = []
    sample_cold_pos_0_net_target_t = []

    labels_pos_2 = []
    labels_pos_1 = []
    labels_neg_0 = []
    positive_2_position = []
    positive_1_position = []
    negative_0_position = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):

            tf1 = EXP_cold[i]

            tf_s = GRN_embedding_s[i]

            tf_t = GRN_embedding_t[i]

            target1 = EXP_cold[j]

            target_s = GRN_embedding_s[j]

            target_t = GRN_embedding_t[j]

            label = int(Ecoli_GRN[i][j])


            if label == 2:

                sample_cold_pos_2_tf.append(tf1)
                sample_cold_pos_2_target.append(target1)

                sample_cold_pos_2_net_tf_s.append(tf_s)
                sample_cold_pos_2_net_tf_t.append(tf_t)
                sample_cold_pos_2_net_target_s.append(target_s)
                sample_cold_pos_2_net_target_t.append(target_t)

                labels_pos_2.append(label)
                positive_2_position.append((i, j))
            elif label == 1:

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

    positive2_data = list(zip(sample_cold_pos_2_tf, sample_cold_pos_2_target, sample_cold_pos_2_net_tf_s, sample_cold_pos_2_net_tf_t, sample_cold_pos_2_net_target_s, sample_cold_pos_2_net_target_t, labels_pos_2, positive_2_position))  # len
    positive1_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, sample_cold_pos_1_net_tf_s, sample_cold_pos_1_net_tf_t, sample_cold_pos_1_net_target_s, sample_cold_pos_1_net_target_t, labels_pos_1, positive_1_position))  # len
    negative0_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, sample_cold_pos_0_net_tf_s, sample_cold_pos_0_net_tf_t, sample_cold_pos_0_net_target_s, sample_cold_pos_0_net_target_t, labels_neg_0, negative_0_position))  # len

    feature_size_tf = sample_cold_pos_2_tf[0].shape[0]
    feature_size_target = sample_cold_pos_2_target[0].shape[0]
    feature_size_tf_nets = sample_cold_pos_2_net_tf_s[0].shape[0]

    return positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets

def transform_data_single(train_data):

    featuretf = []
    featuretarget = []
    label_ = []
    position = []
    for i in range(len(train_data)):
        featuretf.append(train_data[i][0])
        featuretarget.append(train_data[i][1])
        label_.append(train_data[i][2])
        position.append(train_data[i][3])

    featuretf = np.array(featuretf)
    featuretarget = np.array(featuretarget)

    dataX_tf = featuretf[:,np.newaxis,:]
    dataX_target = featuretarget[:,np.newaxis,:]
    print("the shape of dataX_tf: ",dataX_tf.shape)
    print("the shape of dataX_target: ",dataX_target.shape)


    label_ = np.array(label_)

    labelY = to_categorical(label_,3)

    position = np.array(position)

    return dataX_tf, dataX_target, labelY, position


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

    labelY = to_categorical(label_,3)

    position = np.array(position)

    return dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t, labelY, position

