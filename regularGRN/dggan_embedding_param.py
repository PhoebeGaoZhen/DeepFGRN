"""
作用：利用GAN为Ecoli的网络生成GRN的嵌入向量矩阵
E.coli的cold heat oxid的原始网络相同，所以没必要生成三个
GAN: 2个生成器，1个判别器
需要修改的参数，
嵌入维度：n_emb = 32 64 128
训练批次
    n_epoch = 30
    d_epoch = 15
    g_epoch = 10
batch_size
    g_batch_size = 4
    d_batch_size = 4

"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import math
from detailGRN_twoclass import dggan_utils
from detailGRN_twoclass.dggan5_discriminator import Discriminator

from detailGRN_twoclass.dggan5_generator_112 import Generator
from detailGRN_twoclass import dggan_config_112 as dggan_config



class Model():
    # path 是main中传过来的参数，是调控关联的文件  TF-gene
    def __init__(self,pathnetwork, pathnode):
        tf.reset_default_graph() # 因为要循环训练好几次，如果不重置，那有些变量比如node_embedding_matrix就无法初始化
        t = time.time()
        print('reading graph...')
        # 这里要选一个 数据集,main中设置了for循环遍历5个数据集
        #               2033 / 2205个节点                            4329条边
        self.graph, self.n_node, self.node_list, self.node_list_s, self.egs = dggan_utils.read_graph_Ecoli(pathnetwork, pathnode)
        print()
        self.node_emd_shape = [2, self.n_node, dggan_config.n_emb]  # [2, 100, 64 ]
        print('[%.2f] reading graph finished. #node = %d' % (time.time() - t, self.n_node))

        self.dis_node_embed_init = None
        self.gen_node_embed_init = None

        print('building DGGAN model...')
        self.discriminator = None
        self.generator = None

        self.build_generator()
        self.build_discriminator() # 同上

        self.config = tf.ConfigProto()   # 在创建session的时候对session进行参数配置
        # self.config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.config.gpu_options.allow_growth = True  # 用于动态申请显存，从少到多慢慢增加gpu容量,避免溢出
        self.sess = tf.Session(config = self.config)
        self.saver = tf.train.Saver(max_to_keep=0)  # 将训练好的模型参数保存起来，以便以后进行验证或测试，max_to_keep=0表示每训练一代（epoch)就保存一次模型

        print('initial...')
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  # 执行组合操作
        self.sess.run(self.init_op)  # 开始初始化
        print(' initial sucessful------------------------------')

    def build_discriminator(self):
        #with tf.variable_scope("discriminator"):
        self.discriminator = Discriminator(n_node = self.n_node,  # 100
                                           node_emd_init = self.dis_node_embed_init,  # self.dis_node_embed_init = None
                                           config = dggan_config)
    def build_generator(self):
        #with tf.variable_scope("generator"):
        self.generator = Generator(n_node = self.n_node,  # 100
                                   node_emd_init = self.gen_node_embed_init,  # self.gen_node_embed_init = None
                                   config = dggan_config)
    '''
    dis_loss = 0.0
    dis_pos_loss = 0.0
    dis_neg_loss = [0.0, 0.0, 0.0, 0.0]
    dis_cnt = 0
    '''
    def train_dis(self, dis_loss, pos_loss, neg_loss, dis_cnt):
        np.random.shuffle(self.egs)  # 176 条已知边打乱了
        # print('here is train_dis----------------------------')
        info = ''
        # 这里划分了 batchsize = 16
        for index in range(math.floor(len(self.egs) / dggan_config.d_batch_size)):  # range(176/16)
            # len=16（节点编号）；len=16（节点编号）；len=2(详细见下面)
            # print(" prepare data for d--")
            pos_node_ids, pos_node_neighbor_ids, fake_node_embedding = self.prepare_data_for_d(index, self.egs)
            # print(" prepare data for d++")

            '''
            fake_node_embedding, len=2, [[emb1, emb2],[emb1, emb2]]  
            fake_node_embedding[0]是源节点，fake_node_embedding[1]是邻居节点
            fake_node_embedding[0][0]  源节点, direction为0 , shape= (16, 64)
            fake_node_embedding[0][1]  源节点, direction为1, shape= (16, 64)
            fake_node_embedding[1][0]  邻居节点，direction为0, shape= (16, 64)
            fake_node_embedding[1][1]  邻居节点，direction为1, shape= (16, 64)
            '''
            _, _loss, _pos_loss, _neg_loss = self.sess.run([self.discriminator.d_updates, self.discriminator.loss, \
                                            self.discriminator.pos_loss, self.discriminator.neg_loss],
                                            feed_dict = {self.discriminator.pos_node_ids : np.array(pos_node_ids),
                                                         self.discriminator.pos_node_neighbor_ids : np.array(pos_node_neighbor_ids),
                                                         self.discriminator.fake_node_embedding : np.array(fake_node_embedding)})
            # print("optimize discriminator finished")
            dis_loss += _loss
            pos_loss += _pos_loss
            for i in range(4):
                neg_loss[i] += _neg_loss[i]
            dis_cnt += 1
            info = 'dis_loss=%.4f pos_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f neg_loss_2=%.4f neg_loss_3=%.4f' % \
                (dis_loss / dis_cnt, pos_loss / dis_cnt, neg_loss[0] / dis_cnt, neg_loss[1] / dis_cnt, \
                 neg_loss[2] / dis_cnt, neg_loss[3] / dis_cnt)
            avg_dis_loss = dis_loss / dis_cnt
            self.my_print(info, True, 1)  # 展示 总loss  pos loss和 四个 neg loss
        # return (dis_loss, pos_loss, neg_loss, dis_cnt)
        return (dis_loss, pos_loss, neg_loss, dis_cnt, avg_dis_loss)

    def train_gen(self, gen_loss, neg_loss, gen_cnt):
        # self.node_list:   list类型，大小为100，graph中所有节点列表，不分源节点靶节点，[0,1,2....23165]
        np.random.shuffle(self.node_list)
        # print('here is train_gen-----------------------------------')

        info = ''
        # range(100/16)
        for index in range(math.floor(len(self.node_list) / dggan_config.g_batch_size)):
            # len=16    (2,16,64)        (2,16,64)
            node_ids, noise_embedding, dis_node_embedding = self.prepare_data_for_g(index, self.node_list)
            _, _loss, _neg_loss = self.sess.run([self.generator.g_updates, self.generator.loss, self.generator.neg_loss],
                                                 feed_dict = {self.generator.node_ids : np.array(node_ids),
                                                              self.generator.noise_embedding : np.array(noise_embedding),
                                                              self.generator.dis_node_embedding : np.array(dis_node_embedding)})

            gen_loss += _loss
            for i in range(2):
                neg_loss[i] += _neg_loss[i]
            gen_cnt += 1
            # info = 'gen_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f' % (gen_loss / gen_cnt, neg_loss[0] / gen_cnt, neg_loss[1] / gen_cnt)
            avg_gen_loss = gen_loss / gen_cnt
            self.my_print(info, True, 1) # 展示 总loss 两个 neg loss
            # print('here is train_gen-------------- finished ---------------------')
        return (gen_loss, neg_loss, gen_cnt,avg_gen_loss)

    def train(self):
        print('start traning...')
        for epoch in range(dggan_config.n_epoch):
            info = 'epoch %d' % epoch
            self.my_print(info, False, 1)

            dis_loss = 0.0
            dis_pos_loss = 0.0
            dis_neg_loss = [0.0, 0.0, 0.0, 0.0]
            dis_cnt = 0

            gen_loss = 0.0
            gen_neg_loss = [0.0, 0.0]
            gen_cnt = 0

            epochs_d = []
            dis_loss_c = []
            epochs_g = []
            gen_loss_c = []

            # D-step               d_epoch = 15
            for d_epoch in range(dggan_config.d_epoch):
                dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt,avg_dis_loss = self.train_dis(dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt)
                # dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt = self.train_dis(dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt)

                self.my_print('', False, 1)
                # auc = self.evaluate()
                print("dis_loss=%.4f " % avg_dis_loss)
                dis_loss_c.append(avg_dis_loss)
                epochs_d.append(d_epoch)

            # path_figure = 'D:\pycharmProjects\DeepGRN\DeepGRN6\\results\\0929\\'
            # if not os.path.isdir(path_figure):
            #     os.makedirs(path_figure)
            #
            # loss_name = path_figure + 'Discriminator Loss.png'
            # loss_title = 'Loss of Discriminator on E.coli'
            # plt.figure()
            # plt.plot(epochs_d, dis_loss_c, label='loss')
            # plt.title(loss_title)
            # plt.legend(loc='lower right')
            # plt.savefig(loss_name, dpi=600)

            # G-step               g_epoch = 5
            for g_epoch in range(dggan_config.g_epoch):
                gen_loss, gen_neg_loss, gen_cnt,avg_gen_loss = self.train_gen(gen_loss, gen_neg_loss, gen_cnt)
                print("gen_loss=%.4f " % avg_gen_loss)
                gen_loss_c.append(avg_gen_loss)
                epochs_g.append(g_epoch)
                self.my_print('', False, 1)
            # loss_name = path_figure + 'Generator Loss.png'
            # loss_title = 'Loss of Generator on E.coli '
            # plt.figure()
            # plt.plot(epochs_g, utils.smooth_curve(gen_loss_c, factor=0.8), label='Training average loss')
            # plt.title(loss_title)
            # plt.legend(loc='lower right')
            # plt.savefig(loss_name, dpi=600)

        print('training finished.')
        embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        self.sess.close()
        return embedding_matrix

    '''
    此时传入的45750条边在上边被打乱了
    传入边，可以利用generator生成源节点的假嵌入和邻居节点的假嵌入
    '''
    def prepare_data_for_d(self, index, egs):
        # print('here is prepare_data_for_d----------------------')
        pos_node_ids = []  # 真实边的源节点
        pos_node_neighbor_ids = []  # 真实边的邻居节点
        '''
        egs 是真实的边（大小为 176），所以都是pos
        d_batch_size = 16    egs[index*16:(index + 1)*16]  也就是选了16条边
        
            eg 是一条边 : [4, 15]
                 node_id：1
        node_neighbor_id：15
        '''
        for eg in egs[index * dggan_config.d_batch_size : (index + 1) * dggan_config.d_batch_size]:
            node_id, node_neighbor_id = eg
            pos_node_ids.append(node_id)   # 真实的节点，正样本，len为16
            pos_node_neighbor_ids.append(node_neighbor_id)  # 节点真实的邻居节点，正样本，len为16
        # print(pos_node_ids)
        # generate fake node
        fake_node_embedding = []  # [2,2,16,64]
        '''
        np.random.normal(0.0, config.sig, (2, len(pos_node_ids), config.n_emb))
        正态分布，       均值 0， 方差 1        （2,16,64)  
        noise_embedding 类型是array， shape: (2, 16, 64)
        生成的噪声
        '''
        noise_embedding = np.random.normal(0.0, dggan_config.sig, (2, len(pos_node_ids), dggan_config.n_emb))  # (2, 16, 64)
        # print(noise_embedding.shape)
        '''
        generator有三个参数可以传过去，node_ids noise_embedding  dis_node_embedding
        根据generator.fake_node_embedding的计算过程可以发现需要两个参数 node_ids，noise_embedding，因此传入了这两个参数
        self.generator.node_ids : np.array(pos_node_ids), pos_node_ids是列表，len为 16，
        self.generator.noise_embedding : np.array(noise_embedding)，shape为(2, 16, 64)
        '''
        # print('here is fake_node_embedding first ----------------------')
        # 利用generator为真实边的源节点生成假嵌入
        fake_node_embedding.append(self.sess.run(self.generator.fake_node_embedding,
                                            feed_dict = {self.generator.node_ids : np.array(pos_node_ids),
                                                         self.generator.noise_embedding : np.array(noise_embedding)}))

        # print(len(fake_node_embedding[0]))
        # (2, 16, 64) 与上面的 noise_embedding 一样
        noise_embedding = np.random.normal(0.0, dggan_config.sig, (2, len(pos_node_ids), dggan_config.n_emb))
        '''
        与上面不同的是 generator.node_ids : np.array(pos_node_neighbor_ids)
        pos_node_neighbor_ids 与 pos_node_ids 长度一样，128，上面是源节点，这里是靶节点（邻居节点）
        '''
        # print('here is fake_node_embedding second ----------------------')
        # 利用generator为真实边的邻居节点生成假嵌入
        fake_node_embedding.append(self.sess.run(self.generator.fake_node_embedding,
                                            feed_dict = {self.generator.node_ids : np.array(pos_node_neighbor_ids),
                                                         self.generator.noise_embedding : np.array(noise_embedding)}))

        '''
        至此，fake_node_embedding 是一个长度为2的列表，第一个是 选定的结点（源节点），第二个是邻居节点
        两个元素都是Tensor("LeakyRelu:0", shape=(?, 128), dtype=float32)
        也就是两个节点的 embedding（128维）
        也就是说，这个函数为128条边生成了嵌入，
        首先得到这个边的两个节点（pos_node_ids, pos_node_neighbor_ids）
        然后利用GAN的生成器得到了fake_node_embedding
        '''
        return pos_node_ids, pos_node_neighbor_ids, fake_node_embedding


    def prepare_data_for_g(self, index, node_list):
        # print('here is prepare_data_for_g------------------------------')
        node_ids = []  # 存储了 16 个节点
        '''
        node_list 是打乱了的节点列表，长度 100
        g_batch_size = 16    node_list[index*16:(index + 1)*16]  也就是选了 16 个节点

        '''
        for node_id in node_list[index * dggan_config.g_batch_size : (index + 1) * dggan_config.g_batch_size]:
            node_ids.append(node_id)
        '''
        正态分布，均值0 方差1，（2,16,64）
        noise_embedding 类型是array， shape: (2, 16, 64)
        '''
        noise_embedding = np.random.normal(0.0, dggan_config.sig, (2, len(node_ids), dggan_config.n_emb))

        '''
        从node_embedding_matrix中根据 node_ids 得到的 pos_node_embedding，pos_node_neighbor_embedding
        '''
        dis_node_embedding = []
        dis_node_embedding1 = self.sess.run([self.discriminator.pos_node_embedding_1],
                                             feed_dict = {self.discriminator.pos_node_ids : np.array(node_ids)})
        dis_node_embedding2 = self.sess.run([self.discriminator.pos_node_neighbor_embedding_1],
                                             feed_dict = {self.discriminator.pos_node_neighbor_ids : np.array(node_ids)})

        dis_node_embedding = np.vstack([dis_node_embedding1, dis_node_embedding2])
        '''
        node_ids： 128个节点id
        noise_embedding：正态分布生成的噪声嵌入，(2, 128, 128)
        dis_node_embedding： 从discriminator中得到的  ??????
        '''
        return node_ids, noise_embedding, dis_node_embedding

    def my_print(self, info, r_flag, verbose):
        if verbose == 1 and dggan_config.verbose == 0:
            return
        if r_flag:
            print('\r%s' % info, end='')
        else:
            print('%s' % info)


# if __name__ == '__main__':
# def dggan(pathnetwork, savePathsource, savePathtarget):
#
#     model = Model(pathnetwork)
#
#     node_embedding_matrix = model.train()  # (2,100,64)
#     node_embedding_matrix_0 = utils.minmaxstandard(node_embedding_matrix[0])
#     node_embedding_matrix_1 = utils.minmaxstandard(node_embedding_matrix[1])
#
#     data1 = pd.DataFrame(node_embedding_matrix_0) # (100,64)
#     data1.to_csv(savePathsource)
#     data2 = pd.DataFrame(node_embedding_matrix_1) # (100,64)
#     data2.to_csv(savePathtarget)
#     print('successfully saved.')
def dggan(pathnetwork, pathnode):

    model = Model(pathnetwork, pathnode)

    node_embedding_matrix = model.train()  # (2,100,64)
    node_embedding_matrix_0 = dggan_utils.minmaxstandard(node_embedding_matrix[0])
    node_embedding_matrix_1 = dggan_utils.minmaxstandard(node_embedding_matrix[1])
    print('successfully saved.')
    return node_embedding_matrix_0, node_embedding_matrix_1


# pathnetwork = 'D:\\pycharmProjects\\DeepGRN\data\\Ecoil\\integrated_gold_network_new2.tsv'  # 黄金网络的路径 id-id
# path_figure = '.\\embedding\\a1\\'
# if not os.path.isdir(path_figure):
#     os.makedirs(path_figure)
# savePathsource = path_figure + 'Ecoli_net_embedding_s.csv'
# savePathtarget = path_figure + 'Ecoli_net_embedding_t.csv'
#
# dggan(pathnetwork, savePathsource, savePathtarget)


