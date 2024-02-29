"""
Compute kcore and avg cascade length
Extract the train set for INFECTOR
"""

import json
import os
import time
from datetime import datetime
from typing import Dict

import igraph as ig
import numpy as np
import pandas as pd
from collections import defaultdict
import math


def remove_duplicates(cascade_nodes, cascade_times):
    """
    Some tweets have more then one retweets from the same person
    Keep only the first retweet of that person
    """
    duplicates = set([x for x in cascade_nodes if cascade_nodes.count(x) > 1])
    for d in duplicates:
        to_remove = [v for v, b in enumerate(cascade_nodes) if b == d][1:]
        cascade_nodes = [b for v, b in enumerate(cascade_nodes) if v not in to_remove]
        cascade_times = [b for v, b in enumerate(cascade_times) if v not in to_remove]

    return cascade_nodes, cascade_times

def mapped_uid():
    '''
    map user id from uidlist.txt
    :return: dict of user id map
    '''
    file_path = '/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/uidlist.txt'
    with open(file_path, "r", encoding="gbk") as f:
        lines_uid = f.readlines()
    uid_map = {}
    for idx, uid in enumerate(lines_uid):
        uid_map[uid.strip()] = idx

    return uid_map



def get_attribute_dict(fn:str, path: str, attribute: str) -> Dict:
    """
    This function creates a gender dictionary using the profile_gender.csv if the file is available. If the file
    isn't available, it calls the generate_profile_gender_csv() function to generate the CSV and then builds the
    dictionary.

    :param path: path to profile_gender.csv
    :return: gender_dict: dictionary with user IDs as keys and 0 or 1 values indicating that the user is female or male
    """

    try:
        with open(path, 'r', encoding="ISO-8859-1") as f:
            contents = f.read()
    except:
        # path_user_profile = '/'.join(path.split("/")[:-1]) + "/userProfile/"
        path_user_profile = '/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/userProfile/' ####

        txt_files = [os.path.join(path_user_profile, f) for f in os.listdir(path_user_profile) if
                     os.path.isfile(os.path.join(path_user_profile, f))]
        user_profile_df = pd.DataFrame()
        for t in txt_files:
            with open(t, 'r', encoding="ISO-8859-1") as f:
                contents = f.read()
            split_content = contents.split('\n')[:-1]

            reshaped_content = np.reshape(split_content, (int(len(split_content) / 15), 15))
            df = pd.DataFrame(reshaped_content)
            df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            df.columns = df.columns.str.lstrip("# ")
            user_profile_df = user_profile_df.append(df)

        attribute_df = user_profile_df[["id", attribute]].reset_index(drop=True)
        uid_map = mapped_uid()
        attribute_df.id = attribute_df.id.map(uid_map)  # mapping user id

        if attribute == 'gender' and fn == 'weibo':
            gender_conversion_dict = {"m": 1, "f": 0}
            attribute_df[attribute] = attribute_df[attribute].map(gender_conversion_dict)

        attribute_df.to_csv(path, index=False)  # store the processed data

        attribute_dict = pd.Series(attribute_df[attribute].values, index=attribute_df.id).to_dict()

        return attribute_dict

    split_content, split_content_list = contents.split('\n')[0:-1], []
    for i in split_content:
        split_content_list.append(i.split(","))
    split_content_list = split_content_list[1:]

    attribute_dict = {}
    for split_data in split_content_list:
        attribute_dict[split_data[0]] = int(split_data[1])

    return attribute_dict

def compute_coef(L):
    sigma = np.sqrt(np.mean([(L[i]-np.mean(L))**2 for i in range(len(L))])) # strandard deviation
    coef = sigma/np.mean(L) # coef of variation
    sigmoid = 1 / (1 + math.e ** -coef)
    return  2*(1-sigmoid)# sigmod function


def compute_fair(node_list, attribute_dict, grouped, attribute='gender'):
    '''
    :param node_list: cascade nodes
    :param attribute_dict: original attribute dict
    :param grouped: statistics of attribute dict
    :return: fairness score
    '''

    # influenced statistics
    influenced_attribute_dict = {k: attribute_dict[k] for k in node_list}
    T_grouped = defaultdict(list)
    for k, v in influenced_attribute_dict.items():
        T_grouped[v].append(k)

    ratio = [len(T_grouped[k]) / len(grouped[k]) for k in grouped.keys()]

    score = compute_coef(ratio)
    if attribute == 'province':
        min_f = 0.00537
        k = 0.566 # coefficient of scaling get from distribution [0.5,1] a=0.5, b=1, k = (b-a)/(max(score)-min(score))
        score = 0.5 + k * (score-min_f) # 0.5 min scaling border

    return score

def store_samples(fn, cascade_nodes, cascade_times, initiators, train_set, op_time, attribute_dict, grouped, attribute, sampling_perc=120):
    """
    Store the samples  for the train set as described in the node-context pair creation process for INFECTOR
    """
    # ---- Inverse sampling based on copying time
    no_samples = round(len(cascade_nodes) * sampling_perc / 100)
    casc_len = len(cascade_nodes)
    times = [1.0 / (abs((cascade_times[i] - op_time)) + 1) for i in range(0, casc_len)]
    s_times = sum(times)

    f_score = compute_fair(cascade_nodes,attribute_dict,grouped, attribute)
    if (f_score is not None) and (not np.isnan(f_score)):
        if s_times == 0:
            samples = []
        else:
            print("out")
            probs = [float(i) / s_times for i in times]
            samples = np.random.choice(a=cascade_nodes, size=round((no_samples)*f_score), p=probs)  # multiplied by fair score for fps
            # samples = np.random.choice(a=cascade_nodes, size=round((no_samples) * f_score), p=probs) # direct sampling for fac
        # ----- Store train set
        op_id = initiators[0]
        for i in samples:
            train_set.write(str(op_id) + "," + i + "," + str(casc_len) + "," + str(f_score) + "\n")


def run(fn, attribute,sampling_perc, log):
    print("Reading the network")
    cwd = os.getcwd()
    # txt_file_path = '/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/weibo_network.txt' ###
    txt_file_path = '/opt/data/weibo_network.txt' ###
    # txt_file_path = '/media/yuting/TOSHIBA EXT/digg/sampled/digg_network_sampled.txt'  ###
    g = ig.Graph.Read_Ncol(txt_file_path)
    print("Completed reading the network.")

    train_set_file = '/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/train_set_fair_gender_fps_v4.txt'  # set the train_set file according to different attribute
    # train_set_file = '/media/yuting/TOSHIBA EXT/digg/sampled/trainset_fair_age_fps.txt'  # set the train_set file according to different attribute
    attribute_csv = '/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/profile_gender.csv'      #  !!! use v6  set attribute csv file to write corresponding attribute
    # attribute_csv = '/media/yuting/TOSHIBA EXT/digg/profile_age_v3.csv'
    user_attribute_dict = get_attribute_dict(fn, attribute_csv, attribute)

    # group statistics
    attribute_grouped = defaultdict(list)
    for k, v in user_attribute_dict.items():
        attribute_grouped[v].append(k)
    print('generate grouped nodes')

    with open('/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/train_cascades.txt', "r") as f, open(train_set_file, "w") as train_set:
    # with open('/media/yuting/TOSHIBA EXT/digg/sampled/train_cascades_sampled.txt', "r") as f, open(train_set_file, "w") as train_set:
        # ----- Initialize features
        deleted_nodes = []
        g.vs["Cascades_started"] = 0
        g.vs["Cumsize_cascades_started"] = 0
        g.vs["Cascades_participated"] = 0
        log.write(f" net:{fn}\n")
        idx = 0

        start = time.time()
        # ---------------------- Iterate through cascades to create the train set
        for line in f:

            cascade = line.replace("\n", "").split(";")
            if fn == 'weibo':
                cascade_nodes = list(map(lambda x: x.split(" ")[0], cascade[1:]))
                cascade_times = list(map(lambda x: int(((datetime.strptime(x.replace("\r", "").split(" ")[1],
                                                                           '%Y-%m-%d-%H:%M:%S') - datetime.strptime(
                    "2011-10-28", "%Y-%m-%d")).total_seconds())), cascade[1:]))
            else:
                cascade_nodes = list(map(lambda x:  x.split(" ")[0],cascade))
                cascade_times = list(map(lambda x:  int(x.replace("\r","").split(" ")[1]),cascade))

            # ---- Remove retweets by the same person in one cascade
            cascade_nodes, cascade_times = remove_duplicates(cascade_nodes, cascade_times)

            # ---------- Dictionary nodes -> cascades
            op_id, op_time = cascade_nodes[0], cascade_times[0]

            try:
                g.vs.find(name=op_id)["Cascades_started"] += 1
                g.vs.find(op_id)["Cumsize_cascades_started"] += len(cascade_nodes)
            except:
                deleted_nodes.append(op_id)
                continue

            if len(cascade_nodes) < 3:
                continue
            initiators = [op_id]

            store_samples(fn, cascade_nodes[1:], cascade_times[1:], initiators, train_set, op_time,
                          user_attribute_dict, attribute_grouped, attribute, sampling_perc)

            idx += 1
            if idx % 1000 == 0:
                print("-------------------", idx)

        print(f"Number of nodes not found in the graph: {len(deleted_nodes)}")
    log.write(f"Feature extraction time:{str(time.time() - start)}\n")

    print("Evaluating fairness score of each influencer in train_cascades")

    kcores = g.shell_index()
    log.write(f"K-core time:{str(time.time() - start)}\n")
    a = np.array(g.vs["Cumsize_cascades_started"], dtype=np.float)
    b = np.array(g.vs["Cascades_started"], dtype=np.float)

    np.seterr(divide='ignore', invalid='ignore')

    # ------ Store node characteristics
    #node_feature_fair_gender = '/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/FPS/node_features.csv'
    node_feature_fair_age = '/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/FPS/node_feature_age_fps.csv'
    # node_feature_fair_age = '/media/yuting/TOSHIBA EXT/digg/sampled/node_feature_age_fps.csv'
    pd.DataFrame({"Node": g.vs["name"],
                  "Kcores": kcores,
                  "Participated": g.vs["Cascades_participated"],
                  "Avg_Cascade_Size": a / b}).to_csv(node_feature_fair_age, index=False)

    print("Finished storing node characteristics")

    # # ------ Derive incremental node dictionary
    # graph = pd.read_csv('/media/yuting/TOSHIBA EXT/digg/' + fn + "_network.txt", sep=" ")
    # graph.columns = ["node1", "node2", "weight"]
    # all = list(set(graph["node1"].unique()).union(set(graph["node2"].unique())))
    # dic = {int(all[i]): i for i in range(0, len(all))}
    # with open('/media/yuting/TOSHIBA EXT/digg/' + fn + "_incr_dic.json", "w") as json_file:
    #     json.dump(dic, json_file)

if __name__ == '__main__':
    with open("time_log.txt", "a") as log:
        input_fn = 'weibo'
        sampling_perc = 120
        run(input_fn, 'gender', sampling_perc, log)