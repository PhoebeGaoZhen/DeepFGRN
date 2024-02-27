
import pandas as pd
import numpy as np
import random, os,time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score,recall_score,precision_score,matthews_corrcoef
import three_utils_2_single, dggan_embedding_param
import corrresnet_pred_160 as corrresnet_pred


iteration = 10
nb_classes = 3
num_not_regulate = 7000


dim_net = 160
dim_exp = 10
a = dim_exp
b = a + dim_exp
c = b + dim_net
d = c + dim_net
e = d + dim_net
f = e + dim_net

path_network_name_type = 'traindataHuman\\final_GRN\\new_GRN_Liver_GEN_counts_genename.csv'

path_expression = 'traindataHuman\\final_expression\\Liver_GEN_counts.csv'
path_network_ids = 'traindataHuman\\final_GRN_ids\\new_GRN_Liver_GEN_counts_genename_ids.tsv'
path_node = 'traindataHuman\\final_genelist_txt\\new_exp_Liver_GEN_counts_genename_ids.txt'

output_directory = '.\\output_directory\\'
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
logTime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
network_dict_name = 'Liver_counts_'  + str(dim_net) + '_' + logTime
save_index_path = '.\\results\\'
if not os.path.isdir(save_index_path):
    os.makedirs(save_index_path)

EXP_cold_raw = pd.read_csv(path_expression, sep='\,', header=None,engine='python')
EXP_cold = EXP_cold_raw.loc[1:,1:]
EXP_cold = np.array(EXP_cold)
EXP_cold_new = np.zeros((EXP_cold.shape[0],EXP_cold.shape[1]))
for i in range(EXP_cold.shape[0]):
    for j in range(EXP_cold.shape[1]):
        EXP_cold_new[i][j] = float(EXP_cold[i][j])

genename = EXP_cold_raw.loc[1:,0]
genename = np.array(genename)

Ecoli_GRN_known = pd.read_csv(path_network_name_type, sep='\,', header=None,engine='python')

Ecoli_GRN, num_activator, num_repressor, num_unknown = three_utils_2_single.get_GRN(Ecoli_GRN_known,genename)

GRN_embedding_s, GRN_embedding_t = dggan_embedding_param.dggan(path_network_ids, path_node)

network_dict = {"AUROC mean": 0,
                 "AUROC std": 0,
                 "Recall mean": 0,
                 "Recall std": 0,
                 "Precision mean": 0,
                 "Precision std": 0,
                 "F1 mean": 0,
                 "F1 std": 0,
                "MCC mean": 0,
                "MCC std": 0}
all_network_dict = {"AUROC": 0,
                 "Recall": 0,
                 "Precision": 0,
                 "F1": 0,
                "MCC": 0}
kf = KFold(n_splits=5, shuffle=True)
netavgAUROCs = []
netavgAUPRs = []
netavgRecalls = []
netavgSPEs = []
netavgPrecisions = []
netavgF1s = []
netavgMCCs = []

for ki in range(iteration):
    print('\n')
    print("\nthe {}th five-fold cross-validation..........\n".format(ki + 1))

    positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets = three_utils_2_single.create_samples_human_counts(
        EXP_cold_new, Ecoli_GRN, GRN_embedding_s, GRN_embedding_t, num_not_regulate)

    random.shuffle(negative0_data)

    alldata = np.vstack((positive2_data, positive1_data))
    alldata = np.vstack((alldata, negative0_data))
    random.shuffle(alldata)

    dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t, labelY, position = three_utils_2_single.transform_data_single_net(alldata)  # 获取特征和标签

    AUROCs = []
    AUPRs = []
    Recalls = []
    SPEs = []
    Precisions = []
    F1s = []
    MCCs = []

    dataX = []
    for i in range(dataX_tf.shape[0]):
        temp = np.hstack((dataX_tf[i], dataX_target[i]))
        temp = np.hstack((temp, net_tf_s[i]))
        temp = np.hstack((temp, net_tf_t[i]))
        temp = np.hstack((temp, net_target_s[i]))
        temp = np.hstack((temp, net_target_t[i]))
        dataX.append(temp)

    dataX = np.array(dataX)
    for train_index, test_index in kf.split(dataX, labelY):

        trainX, testX = dataX[train_index], dataX[test_index]
        trainY, testY = labelY[train_index], labelY[test_index]

        (trainXX, testXX, trainYY, testYY) = train_test_split(trainX, trainY, test_size=0.2, random_state=1,
                                                              shuffle=True)

        trainXX_tf = trainXX[:,:,0:a]
        trainXX_target = trainXX[:,:,a:b]
        trainXX_net_tf_s = trainXX[:, :, b:c]
        trainXX_net_tf_t = trainXX[:, :, c:d]
        trainXX_net_target_s = trainXX[:, :, d:e]
        trainXX_net_target_t = trainXX[:, :, e:f]

        testXX_tf = testXX[:,:,0:a]
        testXX_target = testXX[:,:,a:b]
        testXX_net_tf_s = testXX[:, :, b:c]
        testXX_net_tf_t = testXX[:, :, c:d]
        testXX_net_target_s = testXX[:, :, d:e]
        testXX_net_target_t = testXX[:, :, e:f]


        testX_tf = testX[:,:,0:a]
        testX_target = testX[:,:,a:b]
        testX_net_tf_s = testX[:, :, b:c]
        testX_net_tf_t = testX[:, :, c:d]
        testX_net_target_s = testX[:, :, d:e]
        testX_net_target_t = testX[:, :, e:f]


        classifier = corrresnet_pred.Classifier_corrResNET_pred(output_directory, nb_classes, trainXX_tf, trainXX_target, trainXX_net_tf_s, trainXX_net_tf_t, trainXX_net_target_s,trainXX_net_target_t, verbose=True, patience=5)


        score_1, score_int = classifier.fit_5CV(trainXX_tf, trainXX_target,  trainXX_net_tf_s, trainXX_net_tf_t, trainXX_net_target_s,trainXX_net_target_t,trainYY,
                                     testXX_tf, testXX_target, testXX_net_tf_s, testXX_net_tf_t, testXX_net_target_s, testXX_net_target_t, testYY,
                                     testX_tf, testX_target, testX_net_tf_s, testX_net_tf_t, testX_net_target_s, testX_net_target_t)


        testY_int = np.argmax(testY, axis=1)

        cm = confusion_matrix(testY_int, score_int)
        conf_matrix = pd.DataFrame(cm)

        AUC = roc_auc_score(testY, score_1, multi_class='ovo')
        Recall = recall_score(testY_int, score_int, average='weighted')
        Precision = precision_score(testY_int, score_int, average='weighted')
        F1 = f1_score(testY_int, score_int, average='weighted')
        MCC = matthews_corrcoef(testY_int, score_int)

        AUROCs.append(AUC)
        Recalls.append(Recall)
        Precisions.append(Precision)
        F1s.append(F1)
        MCCs.append(MCC)

    avg_AUROC = np.mean(AUROCs)
    avg_Recalls = np.mean(Recalls)
    avg_Precisions = np.mean(Precisions)
    avg_F1s = np.mean(F1s)
    avg_MCCs = np.mean(MCCs)

    print("\nAUROC of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_AUROC))
    print("\nMCC of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_MCCs))
    print("\nRecall of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_Recalls))
    print("\nPrecision of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_Precisions))
    print("\nF1 of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_F1s))

    netavgAUROCs.append(avg_AUROC)
    netavgRecalls.append(avg_Recalls)
    netavgPrecisions.append(avg_Precisions)
    netavgF1s.append(avg_F1s)
    netavgMCCs.append(avg_MCCs)


AUROC_mean = np.mean(netavgAUROCs)
AUROC_std = np.std(netavgAUROCs, ddof=1)
Recall_mean = np.mean(netavgRecalls)
Recall_std = np.std(netavgRecalls)
Precision_mean = np.mean(netavgPrecisions)
Precision_std = np.std(netavgPrecisions)
F1_mean = np.mean(netavgF1s)
F1_std = np.std(netavgF1s)
MCC_mean = np.mean(netavgMCCs)
MCC_std = np.std(netavgMCCs)

AUROC_mean = float('{:.4f}'.format(AUROC_mean))
AUROC_std = float('{:.4f}'.format(AUROC_std))
Recall_mean = float('{:.4f}'.format(Recall_mean))
Recall_std = float('{:.4f}'.format(Recall_std))
Precision_mean = float('{:.4f}'.format(Precision_mean))
Precision_std = float('{:.4f}'.format(Precision_std))
F1_mean = float('{:.4f}'.format(F1_mean))
F1_std = float('{:.4f}'.format(F1_std))
MCC_mean = float('{:.4f}'.format(MCC_mean))
MCC_std = float('{:.4f}'.format(MCC_std))


network_dict["AUROC mean"] = AUROC_mean
network_dict["AUROC std"] = AUROC_std
network_dict["Recall mean"] = Recall_mean
network_dict["Recall std"] = Recall_std
network_dict["Precision mean"] = Precision_mean
network_dict["Precision std"] = Precision_std
network_dict["F1 mean"] = F1_mean
network_dict["F1 std"] = F1_std
network_dict["MCC mean"] = MCC_mean
network_dict["MCC std"] = MCC_std


filename = open(save_index_path + network_dict_name + '_avg.csv', 'w')
for k, v in network_dict.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()

all_network_dict["AUROC"] = netavgAUROCs
all_network_dict["Recall"] = netavgRecalls
all_network_dict["Precision"] = netavgPrecisions
all_network_dict["F1"] = netavgF1s
all_network_dict["MCC"] = netavgMCCs

filename = open(save_index_path + network_dict_name+ '_all.csv', 'w')
for k, v in all_network_dict.items():
    filename.write(k + ':' + str(v))
    filename.write('\n')
filename.close()











