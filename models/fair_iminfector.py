import time
from collections import defaultdict
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class IMInfector:
    def __init__(self, fn, embedding_size):
        self.fn = fn
        self.embedding_size = embedding_size
        # self.file_Sn = fn.capitalize() + "/FAC/Embeddings/infector_source3.txt"
        # self.file_Tn = fn.capitalize() + "/FAC/Embeddings/infector_target3.txt"
        self.file_Sn = "/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Embeddings/source_gender_fps+fac_v2_new.txt"
        self.file_Tn = "/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Embeddings/target_gender_fps+fac_v2_new.txt"
        # self.file_Sn = "/media/yuting/TOSHIBA EXT/digg/sampled/embeddings/source_gender_fac.txt"
        # self.file_Tn = "/media/yuting/TOSHIBA EXT/digg/sampled/embeddings/target_gender_fac.txt"
        # self.train_set_file = '/media/yuting/TOSHIBA EXT/digg/sampled/trainset_fair_gender_fac.txt'
        self.train_set_file = '/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/train_set_fair_gender_fps_v4.txt'
        # self.size = 500
        # self.P = 10
        # self.P = 100
        self.alpha = 0
        if(fn=="digg"):
            self.size=200
            self.P = 40
        elif(fn=="weibo"):
            self.size=1000
            self.P = 20
        else:
            self.size=10000
            self.P = 10

        self.target_size = None
        self.input_size = None
        self.chosen = None
        self.bins = None

        self.D = None
        self.S = None

    def get_fair_score(self):
        '''
        :param file_train_set: load the nodes and fair score
        :return:
        '''
        with open(self.train_set_file,'r') as f:
            fair_dict = defaultdict(list)

            for line in f:
                sample = line.replace("\r", "").replace("\n", "").split(",")
                fair_dict[int(sample[0])].append(float(sample[3]))
        fair_dict_avg = {k : np.mean(v) for k, v in fair_dict.items()}

        return fair_dict_avg

    def infl_set(self, candidate, size, uninfected):
        return np.argpartition(self.D[candidate, uninfected], -size)[-size:] # take the index of maximin influence probability with size

    def infl_spread(self, candidate, size, uninfected):
        return sum(np.partition(self.D[candidate, uninfected], -size)[-size:]) # take the maximin influence probability with size

    def embedding_matrix(self, var):
        """
        Derive the matrix embeddings vector from the file
        """
        if var == "T":
            embedding_file = self.file_Tn
            embed_dim = [self.target_size, self.embedding_size]
        else:
            embedding_file = self.file_Sn
            embed_dim = [self.input_size, self.embedding_size]

        nodes, i = [], 0
        f = open(embedding_file, "r")
        emb = np.zeros((embed_dim[0], embed_dim[1]), dtype=np.float64)

        for l in f:
            if "[" in l:
                combined = ""
            if "]" in l:
                combined = combined + " " + l.replace("\n", "").replace("[", "").replace("]", "")
                parts = combined.split(":")
                nodes.append(int(parts[0]))
                emb[i] = np.asarray([float(p.strip()) for p in parts[1].split(" ") if p != ""], dtype=np.float64)
                i += 1
            combined = combined + " " + l.replace("\n", "").replace("[", "").replace("]", "")
        return nodes, emb

    def read_sizes(self):
        # with open(self.fn.capitalize() + "/" + self.fn + "_sizes.txt", "r") as f:
        with open("/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/weibo_sizes_gender_fps+fac_v2_new.txt", "r") as f:
        # with open("/media/yuting/TOSHIBA EXT/digg/sampled/digg_sampled_size_gender_fac.txt", "r") as f:
            self.target_size = int(next(f).strip())
            self.input_size = int(next(f).strip())

    def compute_D(self, S, T):
        """
        Derive matrix D and vector E
        """
        print(S.shape[0])
        perc = int(self.P * S.shape[0] / 100)
        norm = np.apply_along_axis(lambda x: sum(x ** 2), 1, S)
        self.chosen = np.argsort(-norm)[0:perc]
        norm = norm[self.chosen]
        bins = self.target_size * norm / sum(norm)
        self.bins = [int(i) for i in np.rint(bins)]
        # np.save(self.fn.capitalize() + "/E", self.bins)
        np.save("E_digg",self.bins)

        self.D = np.dot(np.around(S[self.chosen], 4), np.around(T.T, 4))

    def process_D(self):
        """
        Derive the diffusion probabilities. Had to be separated with compute_D, because of memory
        """
        self.D = np.apply_along_axis(lambda x: x - abs(max(x)), 1, self.D)
        self.D = np.apply_along_axis(softmax, 1, self.D)
        self.D = np.around(self.D, 3)
        self.D = abs(self.D)
        # np.save(self.fn.capitalize() + "/D", self.D)
        np.save("D_digg", self.D)

    def run_method(self, init_idx):
        """
        IMINFECTOR algorithm
        """
        q = []
        self.S = []
        nid = 0
        mg = 1
        iteration = 3
        infected = np.zeros(self.D.shape[1])
        total = set([i for i in range(self.D.shape[1])])
        uninfected = list(total - set(np.where(infected)[0]))

        dict_fairscore = self.get_fair_score()

        # ----- Initialization
        for u in range(self.D.shape[0]):
            temp_l = [u]
            spr = self.infl_spread(u, int(self.bins[u]), uninfected)
            temp_l.append(spr)
            temp_l.append(dict_fairscore[init_idx[self.chosen[u]]])
            temp_l.append(0)
            q.append(temp_l)

        # Do not sort
        # with open(self.fn.capitalize() + "/FAC/Seeds/final_seeds.txt", "w") as ftp:
        with open("/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/FAC/Seeds/final_seedds_gender_im_v2_new.txt", "w") as ftp:
        # with open("/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/processed4maxmization/weibo/train_set_fair_gender_fps_v2_new.txt", "w") as ftp:

            while len(self.S) < self.size:
                if len(q) <= 0:
                    break
                u = q[0]
                new_s = u[nid]
                if u[iteration] == len(self.S):
                    influenced = self.infl_set(new_s, int(self.bins[new_s]), uninfected)
                    infected[influenced] = 1
                    uninfected = list(total - set(np.where(infected)[0]))

                    # ----- Store the new seed
                    ftp.write(f"{str(init_idx[self.chosen[new_s]])}\n")
                    self.S.append(new_s)
                    if len(self.S) % 50 == 0:
                        print(len(self.S))
                    # ----- Delete uid
                    q = [l for l in q if l[0] != new_s]

                else:
                    # ------- Keep only the number of nodes influenced to rank the candidate seed
                    spr = self.infl_spread(new_s, int(self.bins[new_s]), uninfected)
                    u[mg] = spr
                    if u[mg] < 0:
                        print("Something is wrong")
                    u[iteration] = len(self.S)
                    q = sorted(q, key=lambda x: (1-self.alpha)*x[1]+self.alpha*x[2], reverse=True)


def run(fn, embedding_size, log):
    start = time.time()
    iminfector = IMInfector(fn, embedding_size)
    iminfector.read_sizes()

    nodes_idx, T = iminfector.embedding_matrix("T")
    init_idx, S = iminfector.embedding_matrix("S")

    iminfector.compute_D(S, T)
    del T, S, nodes_idx
    iminfector.process_D()
    iminfector.run_method(init_idx)

    log.write(f"Time taken for the {fn} IMInfector: {str(time.time() - start)}\n")
    print(f"Time taken for the {fn} IMInfector: {str(time.time() - start)}\n")

# TODO scaling of the fair and influence score

if __name__ == '__main__':
    with open("time_log.txt", "a") as log:
        input_fn = 'weibo'
        embedding_size = 50
        run(input_fn, embedding_size, log)