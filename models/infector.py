import json
import math
import time

import numpy as np
import tensorflow as tf


class Infector:
    def __init__(self, fn, learning_rate, n_epochs, embedding_size, num_samples):
        self.fn = fn
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.embedding_size = embedding_size
        self.num_samples = num_samples
        self.file_Sn = "/media/yuting/TOSHIBA EXT/weibo/weibodata/processed4maxmization/weibo/embeddings/source_gender_fps+fac_v2_new.txt"
        self.file_Tn = "/media/yuting/TOSHIBA EXT/weibo/weibodata/processed4maxmization/weibo/embeddings/target_gender_fps+fac_v2_new.txt"
        # self.file_Sn = "/media/yuting/TOSHIBA EXT/digg/sampled/embeddings/source_age_fac.txt"
        # self.file_Tn = "/media/yuting/TOSHIBA EXT/digg/sampled/embeddings/target_age_fac.txt"

        self.min = np.inf
        self.max = 0#i#
        self.range = None
        self.dic_in = {}
        self.vocabulary_size = None
        self.dic_out = {}
        self.target_size = None
        self.graph = None

        self.loss1 = None
        # self.loss2 = None
        self.loss3 = None
        self.train_step1 = None
        # self.train_step2 = None
        self.train_step3 = None
        self.Sn = None
        self.Tn = None

    def create_dicts(self):
        """
        Min max normalization of cascade length and source-target dictionaries
        """
        initiators = []
        # with open(self.fn.capitalize() + "/Init_Data/train_set.txt", "r") as f:
        with open('/media/yuting/TOSHIBA EXT/weibo/weibodata/processed4maxmization/weibo/train_set_fair_gender_fps_v2_new.txt', "r") as f:
        # with open('/media/yuting/TOSHIBA EXT/digg/sampled/trainset_fair_age_fac.txt', "r") as f:
            for l in f:
                parts = l.split(",")
                initiators.append(parts[0])
                t = int(parts[2])
                if t < self.min:
                    self.min = t
                if t > self.max:
                    self.max = t
        self.range = self.max - self.min

        # ----------------- Source node dictionary
        initiators = np.unique(initiators)

        self.dic_in = {initiators[i]: i for i in range(0, len(initiators))}
        self.vocabulary_size = len(self.dic_in)
        print(self.vocabulary_size)
        # ----------------- Target node dictionary
        # with open(self.fn.capitalize() + "/Init_Data/" + self.fn + "_incr_dic.json", "r") as f:
        # with open("/media/yuting/TOSHIBA EXT/digg/sampled/digg_sampled_incr_dic.json", "r") as f:
        with open("/media/yuting/TOSHIBA EXT/weibo/weibodata/processed4maxmization/weibo/weibo_incr_dic.json", "r") as f:
            self.dic_out = json.load(f)
        self.target_size = len(self.dic_out)
        print(self.target_size)
        # with open(self.fn.capitalize() + "/" + self.fn + "_sizes.txt", "w") as f:
        # with open("/media/yuting/TOSHIBA EXT/digg/sampled/digg_sampled_size_age_fac.txt","w") as f:
        with open("/media/yuting/TOSHIBA EXT/weibo/weibodata/processed4maxmization/weibo/weibo_sizes_gender_fps+fac_v2_new.txt", "w") as f:
            f.write(f"{str(self.target_size)}\n")
            f.write(str(self.vocabulary_size))

    def model(self):
        """
        The multi-task learning NN to classify influenced nodes and predict cascade length
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            # ---- Batch size depends on the cascade
            u = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name="u")
            v = tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name="v")

            # ------------ Source (hidden layer embeddings)
            S = tf.Variable(tf.random.uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name="S")
            u2 = tf.squeeze(u)
            Su = tf.nn.embedding_lookup(S, u2, name="Su")
            # ------------- First task
            # ------------ Target (hidden and output weights)
            T = tf.Variable(tf.random.truncated_normal([self.target_size, self.embedding_size],
                                                       stddev=1.0 / math.sqrt(self.embedding_size)), name="T")

            # ---- Noise contrastive loss function
            nce_biases = tf.Variable(tf.zeros([self.target_size]))

            self.loss1 = tf.reduce_mean(
                tf.nn.nce_loss(weights=T,
                               biases=nce_biases,
                               labels=v,
                               inputs=Su,
                               num_sampled=self.num_samples,
                               num_classes=self.target_size))

            # ------------- Second task
            # ---- Cascade length
            c = tf.compat.v1.placeholder(tf.float32, name="c")

            # ------------ Cascade length weights (output layer of cascade length prediction)
            C = tf.constant(np.repeat(1, self.embedding_size).reshape((self.embedding_size, 1)), tf.float32, name="C")

            # ------------ Bias for cascade length
            b_c = tf.Variable(tf.zeros((1, 1)), name="b_c")

            # ------------- Third task
            # ---- Fairness Score
            fair = tf.compat.v1.placeholder(tf.float32, name="fair")

            # ------------ Fairness weights (output layer of fairness prediction)
            Fair = tf.constant(np.repeat(1, self.embedding_size).reshape((self.embedding_size, 1)), tf.float32,
                               name="Fair")

            # ------------ Bias for fairness score
            b_fair = tf.Variable(tf.zeros((1, 1)), name="b_fair")

            # ------------ Loss2
            alpha = 1.0  ## why alpha == 0.0
            tmp = tf.tensordot(Su, C, 1)
            o2 = tf.sigmoid(tmp + b_c)
            self.loss2 = alpha * tf.square((o2 - c))

            # ------------ Loss3
            beta = 1.0
            tmp2 = tf.tensordot(Su, Fair, 1)
            o_f_2 = tf.sigmoid(tmp2 + b_fair)
            self.loss3 = beta * tf.square(o_f_2 - fair)

            # ---- To retrieve the embedding-node pairs after training
            n_in = tf.compat.v1.placeholder(tf.int32, shape=[1], name="n_in")
            self.Sn = tf.nn.embedding_lookup(S, n_in, name="Sn")

            n_out = tf.compat.v1.placeholder(tf.int32, shape=[1], name="n_out")
            self.Tn = tf.nn.embedding_lookup(T, n_out, name="Tn")

            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            # --- Separate optimizations, for joint to loss1+loss2
            self.train_step1 = optimizer.minimize(self.loss1)
            # self.train_step2 = optimizer.minimize(self.loss2)
            self.train_step3 = optimizer.minimize(self.loss3)

    def train(self):
        """ Train the model """
        l1s, l2s, l3s = [], [], []
        with tf.compat.v1.Session(graph=self.graph) as sess:
            sess.run(tf.compat.v1.initialize_all_variables())
            for epoch in range(self.n_epochs):
                # --------- Train
                # with open(self.fn.capitalize() + "/Init_Data/train_set.txt", "r") as f:
                with open('/media/yuting/TOSHIBA EXT/weibo/weibodata/processed4maxmization/weibo/train_set_fair_gender_fps_v2_new.txt',"r") as f:
                # with open('/media/yuting/TOSHIBA EXT/digg/sampled/trainset_fair_age_fac.txt',"r") as f:

                    idx, init, inputs, labels = 0, -1, [], []
                    # ---- Build the input batch
                    for line in f:
                        # ---- input node, output node, copying_time, cascade_length, 10 negative samples
                        sample = line.replace("\r", "").replace("\n", "").split(",")
                        try:
                            original = self.dic_in[sample[0]]
                            label = self.dic_out[sample[1]]
                        except:
                            continue
                        # ---- check if we are at the same cascade
                        if (init == original) or (init < 0):
                            init = original
                            inputs.append(original)
                            labels.append(label)
                            casc_len = int(sample[2])
                            casc = (float(sample[2]) - self.min) / self.range
                            fairness_score = float(float(sample[3]) * 1)

                        # ---- New cascade, train on the previous one
                        else:
                            # ---------- Run one training batch
                            # --- Train for target nodes
                            if len(inputs) < 2:
                                inputs.append(inputs[0])
                                labels.append(labels[0])
                            inputs = np.asarray(inputs).reshape((len(inputs), 1))
                            labels = np.asarray(labels).reshape((len(labels), 1))

                            sess.run(self.train_step1,
                                     feed_dict={"u:0": inputs, "v:0": labels, "c:0": [[0]], "fair:0": [[0]]})

                            # # --- Train for cascade length
                            # sess.run(self.train_step2,
                            #          feed_dict={"u:0": inputs[0].reshape(1, 1), "v:0": labels, "c:0": [[casc]],
                            #                     "fair:0": [[0]]})
                            #
                            # --- Train for fairness score
                            for i in range(casc_len):
                                sess.run(self.train_step3,
                                         feed_dict={"u:0": inputs[0].reshape(1, 1), "v:0": labels, "c:0": [[0]],
                                                    "fair:0": [[fairness_score]]})

                            idx += 1

                            if idx % 1000 == 0:  # Collecting losses to see if the values are decreasing

                                l1 = sess.run(self.loss1, feed_dict={"u:0": inputs, "v:0": labels, "c:0": [[casc]],
                                                                     "fair:0": [[fairness_score]]})
                                # l2 = sess.run(self.loss2,
                                #               feed_dict={"u:0": inputs[0].reshape(1, 1), "v:0": labels, "c:0": [[casc]],
                                #                          "fair:0": [[fairness_score]]})

                                l3 = sess.run(self.loss3,
                                              feed_dict={"u:0": inputs[0].reshape(1, 1), "v:0": labels, "c:0": [[casc]],
                                                         "fair:0": [[fairness_score]]})
                                l1s.append(l1)
                                # l2s.append(l2)
                                l3s.append(l3)
                                print("epoch: ", epoch)
                                print(f'Loss 1 at step {idx}: {l1}')
                                # print(f'Loss 2 at step {idx}: {l2}')
                                print(f'Loss 3 at step {idx}: {l3}')

                            # ---- Arrange for the next batch
                            inputs, labels = [], []
                            inputs.append(original)
                            labels.append(label)
                            casc = (float(sample[2]) - self.min) / self.range
                            init = original

            with open(self.file_Sn, "w") as fsn:
                # ---------- Store the source embedding of each node
                for node in self.dic_in.keys():
                    emb_Sn = sess.run("Sn:0", feed_dict={"n_in:0": np.asarray([self.dic_in[node]])})
                    fsn.write(node + ":" + ",".join([str(s) for s in list(emb_Sn)]) + "\n")

            with open(self.file_Tn, "w") as ftn:
                # ---------- Store the target embedding of each node
                for node in self.dic_out.keys():
                    emb_Tn = sess.run("Tn:0", feed_dict={"n_out:0": np.asarray([self.dic_out[node]])})
                    ftn.write(node + ":" + ",".join([str(s) for s in list(emb_Tn)]) + "\n")

            return l1s, l2s, l3s


def run(fn, learning_rate, n_epochs, embedding_size, num_neg_samples, log):
    start = time.time()
    infector = Infector(fn, learning_rate, n_epochs, embedding_size, num_neg_samples)

    infector.create_dicts()
    infector.model()

    l1s, l2s, l3s = infector.train()

    log.write(f"Time taken for the {fn} infector:{str(time.time() - start)}\n")
    print(f"Time taken for the {fn} infector:{str(time.time() - start)}\n")


if __name__ == '__main__':
    with open("time_log.txt", "a") as log:
        input_fn = 'weibo'
        learning_rate = 0.1
        n_epochs = 10
        embedding_size = 50
        num_neg_samples = 10
        sampling_perc = 120
        run(input_fn, learning_rate, n_epochs, embedding_size, num_neg_samples, log)