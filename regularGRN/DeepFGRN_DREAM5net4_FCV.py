import tensorflow as tf
import time
import numpy as np
import random, os
from sklearn.model_selection import train_test_split, KFold
import utils, embedding_DREAM5
import corrresnet_pred as corrresnet_pred

if __name__ == '__main__':
    iteration = 5
    nb_classes = 2
    dim_net = 224
    dim_exp = 536
    a = dim_exp
    b = a + dim_exp
    c = b + dim_net
    d = c + dim_net
    e = d + dim_net
    f = e + dim_net
    path_figure = '.\\results\\'
    logTime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
    network_dict_name = 'DREAM5 network4_' + str(dim_net) + '_' + logTime
    if not os.path.isdir(path_figure):
        os.makedirs(path_figure)
    negNum = 3940
    network_dict = {"AUROC mean": 0,
                    "AUROC std": 0,
                    "AUPR mean": 0,
                    "AUPR std": 0,
                    "Recall mean": 0,
                    "Recall std": 0,
                    "SPE mean": 0,
                    "SPE std": 0,
                    "Precision mean": 0,
                    "Precision std": 0,
                    "F1 mean": 0,
                    "F1 std": 0,
                    "MCC mean": 0,
                    "MCC std": 0,
                    "Acc mean": 0,
                    "Acc std": 0}
    print('DREAM5 network4 is training............................................................')
    path = '.\\traindataDREAM5\\DREAM5_NetworkInference_GoldStandard_Network4_Scerevisiae.tsv'
    pathts = '.\\traindataDREAM5\\net4_expression_data.tsv'

    pathnode = '.\\traindataDREAM5\\net4_node.txt'
    output_directory = '.\\output_directory\\DREAM5network4\\'
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    gene10, gene10_ts = utils.readRawData_gene100(path, pathts)

    geneNetwork = utils.createGRN_gene100(gene10, gene10_ts)

    GRN_embedding_s, GRN_embedding_t = embedding_DREAM5.dggan(path, pathnode)
    positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets = utils.createSamples_gene100_1(
        gene10_ts, GRN_embedding_s, GRN_embedding_t,
        geneNetwork)

    with tf.Session() as sess:
        kf = KFold(n_splits=5, shuffle=True)
        netavgAUROCs = []
        netavgAUPRs = []
        netavgRecalls = []
        netavgSPEs = []
        netavgPrecisions = []
        netavgF1s = []
        netavgMCCs = []
        netavgAccs = []

        for ki in range(iteration):
            print('\n')
            print("\nthe {}th five-fold cross-validation..........\n".format(ki + 1))

            random.shuffle(positive1_data)
            random.shuffle(negative0_data)
            print('the number of negative set is: ' + str(negNum))
            alldata = np.vstack((positive1_data, negative0_data[0:negNum]))
            random.shuffle(alldata)
            dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t, labelY, position = utils.transform_data_single_net(
                alldata)

            # 5CV
            AUROCs = []
            AUPRs = []
            Recalls = []
            SPEs = []
            Precisions = []
            F1s = []
            MCCs = []
            Accs = []

            dataX = []
            for i in range(dataX_tf.shape[0]):
                temp = np.hstack((dataX_tf[i], dataX_target[i]))
                temp = np.hstack((temp, net_tf_s[i]))
                temp = np.hstack((temp, net_tf_t[i]))
                temp = np.hstack((temp, net_target_s[i]))
                temp = np.hstack((temp, net_target_t[i]))
                dataX.append(temp)
            dataX = np.array(dataX)  #
            for train_index, test_index in kf.split(dataX, labelY):

                trainX, testX = dataX[train_index], dataX[test_index]
                trainY, testY = labelY[train_index], labelY[test_index]

                (trainXX, testXX, trainYY, testYY) = train_test_split(trainX, trainY, test_size=0.2, random_state=1,
                                                                      shuffle=True)

                trainXX_tf = trainXX[:, :, 0:a]
                trainXX_target = trainXX[:, :, a:b]
                trainXX_net_tf_s = trainXX[:, :, b:c]
                trainXX_net_tf_t = trainXX[:, :, c:d]
                trainXX_net_target_s = trainXX[:, :, d:e]
                trainXX_net_target_t = trainXX[:, :, e:f]

                testXX_tf = testXX[:, :, 0:a]
                testXX_target = testXX[:, :, a:b]
                testXX_net_tf_s = testXX[:, :, b:c]
                testXX_net_tf_t = testXX[:, :, c:d]
                testXX_net_target_s = testXX[:, :, d:e]
                testXX_net_target_t = testXX[:, :, e:f]

                testX_tf = testX[:, :, 0:a]
                testX_target = testX[:, :, a:b]
                testX_net_tf_s = testX[:, :, b:c]
                testX_net_tf_t = testX[:, :, c:d]
                testX_net_target_s = testX[:, :, d:e]
                testX_net_target_t = testX[:, :, e:f]

                classifier = corrresnet_pred.Classifier_corrResNET_pred(output_directory, nb_classes, trainXX_tf,
                                                                        trainXX_target, trainXX_net_tf_s,
                                                                        trainXX_net_tf_t, trainXX_net_target_s,
                                                                        trainXX_net_target_t, verbose=True,
                                                                        patience=5)

                score_1, score_int = classifier.fit_5CV(trainXX_tf, trainXX_target, trainXX_net_tf_s,
                                                        trainXX_net_tf_t, trainXX_net_target_s,
                                                        trainXX_net_target_t, trainYY,
                                                        testXX_tf, testXX_target, testXX_net_tf_s, testXX_net_tf_t,
                                                        testXX_net_target_s, testXX_net_target_t, testYY,
                                                        testX_tf, testX_target, testX_net_tf_s, testX_net_tf_t,
                                                        testX_net_target_s, testX_net_target_t)

                new_score_1 = score_1

                Recall, SPE, Precision, F1, MCC, ACC, AUC, AUPR = utils.two_scores(testY[:, 1],
                                                                                                  new_score_1[:, 1],
                                                                                                  th=0.5)
                AUROCs.append(AUC)
                AUPRs.append(AUPR)
                SPEs.append(SPE)
                Recalls.append(Recall)
                Precisions.append(Precision)
                F1s.append(F1)
                MCCs.append(MCC)
                Accs.append(ACC)

            avg_AUROC = np.mean(AUROCs)
            avg_AUPR = np.mean(AUPRs)
            avg_Recalls = np.mean(Recalls)
            avg_SPEs = np.mean(SPEs)
            avg_Precisions = np.mean(Precisions)
            avg_F1s = np.mean(F1s)
            avg_MCCs = np.mean(MCCs)
            avg_Accs = np.mean(Accs)

            netavgAUROCs.append(avg_AUROC)
            netavgAUPRs.append(avg_AUPR)
            netavgRecalls.append(avg_Recalls)
            netavgSPEs.append(avg_SPEs)
            netavgPrecisions.append(avg_Precisions)
            netavgF1s.append(avg_F1s)
            netavgMCCs.append(avg_MCCs)
            netavgAccs.append(avg_Accs)
    AUROC_mean = np.mean(netavgAUROCs)
    AUROC_std = np.std(netavgAUROCs, ddof=1)
    AUPR_mean = np.mean(netavgAUPRs)
    AUPR_std = np.std(netavgAUPRs)
    Recall_mean = np.mean(netavgRecalls)
    Recall_std = np.std(netavgRecalls)
    SPE_mean = np.mean(netavgSPEs)
    SPE_std = np.std(netavgSPEs)
    Precision_mean = np.mean(netavgPrecisions)
    Precision_std = np.std(netavgPrecisions)
    F1_mean = np.mean(netavgF1s)
    F1_std = np.std(netavgF1s)
    MCC_mean = np.mean(netavgMCCs)
    MCC_std = np.std(netavgMCCs)
    Acc_mean = np.mean(netavgAccs)
    Acc_std = np.std(netavgAccs)

    AUROC_mean = float('{:.4f}'.format(AUROC_mean))
    AUROC_std = float('{:.4f}'.format(AUROC_std))
    AUPR_mean = float('{:.4f}'.format(AUPR_mean))
    AUPR_std = float('{:.4f}'.format(AUPR_std))
    Recall_mean = float('{:.4f}'.format(Recall_mean))
    Recall_std = float('{:.4f}'.format(Recall_std))
    SPE_mean = float('{:.4f}'.format(SPE_mean))
    SPE_std = float('{:.4f}'.format(SPE_std))
    Precision_mean = float('{:.4f}'.format(Precision_mean))
    Precision_std = float('{:.4f}'.format(Precision_std))
    F1_mean = float('{:.4f}'.format(F1_mean))
    F1_std = float('{:.4f}'.format(F1_std))
    MCC_mean = float('{:.4f}'.format(MCC_mean))
    MCC_std = float('{:.4f}'.format(MCC_std))
    Acc_mean = float('{:.4f}'.format(Acc_mean))
    Acc_std = float('{:.4f}'.format(Acc_std))

    network_dict["AUROC mean"] = AUROC_mean
    network_dict["AUROC std"] = AUROC_std
    network_dict["AUPR mean"] = AUPR_mean
    network_dict["AUPR std"] = AUPR_std
    network_dict["Recall mean"] = Recall_mean
    network_dict["Recall std"] = Recall_std
    network_dict["SPE mean"] = SPE_mean
    network_dict["SPE std"] = SPE_std
    network_dict["Precision mean"] = Precision_mean
    network_dict["Precision std"] = Precision_std
    network_dict["F1 mean"] = F1_mean
    network_dict["F1 std"] = F1_std
    network_dict["MCC mean"] = MCC_mean
    network_dict["MCC std"] = MCC_std
    network_dict["Acc mean"] = Acc_mean
    network_dict["Acc std"] = Acc_std


    filename = open(path_figure + network_dict_name + '.csv', 'w')
    for k, v in network_dict.items():
        filename.write(k + ':' + str(v))
        filename.write('\n')
    filename.close()






