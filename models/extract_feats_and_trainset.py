"""
Compute kcore and avg cascade length
Extract the train set for INFECTOR
"""

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict

import igraph as ig
import numpy as np
import pandas as pd


def remove_duplicates(cascade_nodes, cascade_times):
    """
    Summary:
    Removes duplicate cascade nodes while preserving corresponding times.

    Args:
      cascade_nodes: List of cascade nodes (e.g., user IDs).
      cascade_times: List of corresponding timestamps for each cascade node.

    Returns:
      A tuple containing the filtered lists: (unique_cascade_nodes, unique_cascade_times).

    This implementation removes duplicates in one pass through cascade nodes and times.
    """
    seen = set()
    unique_cascade_nodes = []
    unique_cascade_times = []
    for node, t in zip(cascade_nodes, cascade_times):
        if node not in seen:
            seen.add(node)
            unique_cascade_nodes.append(node)
            unique_cascade_times.append(t)
    return unique_cascade_nodes, unique_cascade_times


def mapped_uid():
    """
    map user id from uidlist.txt
    :return: dict of user id map
    """
    file_path = "/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/uidlist.txt"
    with open(file_path, "r", encoding="gbk") as f:
        uid_map = {uid.strip(): idx for idx, uid in enumerate(f)}

    return uid_map


def get_attribute_dict(fn: str, path: str, attribute: str) -> Dict:
    """
    This function creates a gender dictionary using the profile_gender.csv if the file is available. If the file
    isn't available, it calls the generate_profile_gender_csv() function to generate the CSV and then builds the
    dictionary.

    :param path: path to profile_gender.csv
    :return: gender_dict: dictionary with user IDs as keys and 0 or 1 values indicating that the user is female or male
    """

    try:
        with open(path, "r", encoding="ISO-8859-1") as f:
            contents = f.read()
    except:
        # path_user_profile = '/'.join(path.split("/")[:-1]) + "/userProfile/"
        path_user_profile = "/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/userProfile/"  ####

        txt_files = [
            os.path.join(path_user_profile, f)
            for f in os.listdir(path_user_profile)
            if os.path.isfile(os.path.join(path_user_profile, f))
        ]
        user_profile_df = pd.DataFrame()
        for t in txt_files:
            with open(t, "r", encoding="ISO-8859-1") as f:
                contents = f.read()
            split_content = contents.split("\n")[:-1]

            reshaped_content = np.reshape(
                split_content, (int(len(split_content) / 15), 15)
            )
            df = pd.DataFrame(reshaped_content)
            df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            df.columns = df.columns.str.lstrip("# ")
            user_profile_df = user_profile_df.append(df)

        attribute_df = user_profile_df[["id", attribute]].reset_index(drop=True)
        uid_map = mapped_uid()
        attribute_df.id = attribute_df.id.map(uid_map)  # mapping user id

        if attribute == "gender" and fn == "weibo":
            gender_conversion_dict = {"m": 1, "f": 0}
            attribute_df[attribute] = attribute_df[attribute].map(
                gender_conversion_dict
            )

        attribute_df.to_csv(path, index=False)  # store the processed data

        attribute_dict = pd.Series(
            attribute_df[attribute].values, index=attribute_df.id
        ).to_dict()

        return attribute_dict

    split_content, split_content_list = contents.split("\n")[0:-1], []
    for i in split_content:
        split_content_list.append(i.split(","))
    split_content_list = split_content_list[1:]

    attribute_dict = {}
    for split_data in split_content_list:
        attribute_dict[split_data[0]] = int(split_data[1])

    return attribute_dict


def compute_coef(L):
    sigma = np.sqrt(
        np.mean([(L[i] - np.mean(L)) ** 2 for i in range(len(L))])
    )  # strandard deviation
    coef = sigma / np.mean(L)  # coef of variation
    sigmoid = 1 / (1 + math.e**-coef)
    return 2 * (1 - sigmoid)  # sigmod function


def compute_fair(node_list, attribute_dict, grouped, attribute="gender"):
    """
    :param node_list: cascade nodes
    :param attribute_dict: original attribute dict
    :param grouped: statistics of attribute dict
    :return: fairness score
    """

    # influenced statistics
    influenced_attribute_dict = {k: attribute_dict[k] for k in node_list}
    T_grouped = defaultdict(list)
    for k, v in influenced_attribute_dict.items():
        T_grouped[v].append(k)

    ratio = [len(T_grouped[k]) / len(grouped[k]) for k in grouped.keys()]

    score = compute_coef(ratio)
    if attribute == "province":
        min_f = 0.00537
        k = 0.566  # coefficient of scaling get from distribution [0.5,1] a=0.5, b=1, k = (b-a)/(max(score)-min(score))
        score = 0.5 + k * (score - min_f)  # 0.5 min scaling border

    return score


def store_samples(
    fn,
    cascade_nodes,
    cascade_times,
    initiators,
    train_set,
    op_time,
    attribute_dict,
    grouped,
    attribute,
    sampling_perc=120,
):
    """
    Store the samples  for the train set as described in the node-context pair creation process for INFECTOR
    """
    # ---- Inverse sampling based on copying time
    no_samples = round(len(cascade_nodes) * sampling_perc / 100)
    casc_len = len(cascade_nodes)
    times = [1.0 / (abs((cascade_times[i] - op_time)) + 1) for i in range(0, casc_len)]
    s_times = sum(times)

    f_score = compute_fair(cascade_nodes, attribute_dict, grouped, attribute)
    if (f_score is not None) and (not np.isnan(f_score)):
        if s_times == 0:
            samples = []
        else:
            print("out")
            probs = [float(i) / s_times for i in times]
            samples = np.random.choice(
                a=cascade_nodes, size=round((no_samples) * f_score), p=probs
            )  # multiplied by fair score for fps
            # samples = np.random.choice(a=cascade_nodes, size=round((no_samples) * f_score), p=probs) # direct sampling for fac
        # ----- Store train set
        op_id = initiators[0]
        for i in samples:
            train_set.write(
                str(op_id) + "," + i + "," + str(casc_len) + "," + str(f_score) + "\n"
            )


######## BEGIN SECTION IS FROM SPRING 2024 ###########


def process_csv(input_file, output_file):
    with open(input_file, "r") as csv_in:
        reader = csv.reader(csv_in)
        rows = list(reader)  # Read all rows into a list

    # Process rows and set third column based on probability
    for row in rows[1:]:
        prob = random.random()
        if prob < 0.325:
            row[1] = 0
        elif prob < 0.552:
            row[1] = 1
        elif prob < 0.745:
            row[1] = 2
        elif prob < 0.849:
            row[1] = 3
        elif prob < 0.882:
            row[1] = 4
        elif prob < 0.936:
            row[1] = 5
        else:
            row[1] = 6

    with open(output_file, "w", newline="") as csv_out:
        writer = csv.writer(csv_out)
        writer.writerows(rows)


def randomly_assign_age_group():
    prob = random.random()
    if prob < 0.325:
        return 0
    elif prob < 0.552:
        return 1
    elif prob < 0.745:
        return 2
    elif prob < 0.849:
        return 3
    elif prob < 0.882:
        return 4
    elif prob < 0.936:
        return 5
    else:
        return 6


def randomly_assign_political_lean():
    # Source: Gallop https://news.gallup.com/poll/15370/party-affiliation.aspx
    # 2024 March 1-20: 30% Repulican, 41% Independent, 28% Democrat.
    # Independent is split evenly between Republican and Democrat
    prob = random.random()
    if prob < 0.555:
        return 0  # Simulates Republican
    else:
        return 1  # Simulates Democrat


def randomly_assign_age_x_pol_affln():
    """
    Population source: https://www.statista.com/statistics/296974/us-population-share-by-generation/
    Political affiliation source: https://www.statista.com/statistics/319068/party-identification-in-the-united-states-by-generation/
    Note: this evenly splits independent across the Republican and Democrat parties evently.
    - silent generation: 5.49%
      - 39 + 13 = 52 % Republican, 35 + 13 = 48% Democrat (26% independent)
    - baby boomers: 20.58%
      - 35 + 16.5 = 51.5 % Republican, 32 + 16.5 = 48.5% Democrat (33% independent)
    - generation x: 19.61%
      - 30 + 22 = 52 % Republican, 27 + 22 = 49% Democrat (44% independent)
    - millenials: 21.67%
      - 21 + 26 = 47 % Republican, 27 + 26 = 53% Democrat (52% independent)
    - generation z: 20.88%
      - 17 + 26 = 43 % Republican, 31 + 26 = 57% Democrat (52% independent)
    =========
    0: silent republican = .0549 * .52 = .0264
    1: silent democrat = .0549 * .48 = .0263
    2: baby boomers republican = .2058 * .515 = .1060
    3: baby boomers democrat = .2058 * .485 = .0998
    4: generation x republican = .1961 * .52 = .1020
    5: generation x democrat = .1961 * .49 = .0961
    6: millenials republican = .2167 * .47 = .1018
    7: millenials democrat = .2167 * .53 = .1149
    8: generation z republican = .2088 * .43 = .0897
    9: generation z democrat = .2088 * .57 = .119
    """
    groups = {
        0: 0.0264,
        1: 0.0263,
        2: 0.1060,
        3: 0.0998,
        4: 0.1020,
        5: 0.0961,
        6: 0.1018,
        7: 0.1149,
        8: 0.0897,
        9: 0.119,
    }
    prob = random.random() * sum(groups.values())
    cumulative_prob = 0
    for key, value in groups.items():
        cumulative_prob += value
        if prob <= cumulative_prob:
            return key


def randomly_assign_gender_x_pol_affln():
    """
    2017 political affiliation by gender source: https://www.pewresearch.org/politics/2018/03/20/1-trends-in-party-affiliation-among-demographic-groups/

    0: male republicans = .48
    1: male democrats = .44
    3: female republicans = .56
    4: female republicans = 37
    """
    groups = {0: 0.48, 1: 0.44, 3: 0.56, 4: 0.37}
    prob = random.random() * sum(groups.values())
    cumulative_prob = 0
    for key, value in groups.items():
        cumulative_prob += value
        if prob <= cumulative_prob:
            return key


def randomly_assign_age_x_pol_affln_with_noise(noise_std_dev=0.1):
    """
    Population source: https://www.statista.com/statistics/296974/us-population-share-by-generation/
    Political affiliation source: https://www.statista.com/statistics/319068/party-identification-in-the-united-states-by-generation/
    Note: this evenly splits independent across the Republican and Democrat parties evently.
    - silent generation: 5.49%
      - 39 + 13 = 52 % Republican, 35 + 13 = 48% Democrat (26% independent)
    - baby boomers: 20.58%
      - 35 + 16.5 = 51.5 % Republican, 32 + 16.5 = 48.5% Democrat (33% independent)
    - generation x: 19.61%
      - 30 + 22 = 52 % Republican, 27 + 22 = 49% Democrat (44% independent)
    - millenials: 21.67%
      - 21 + 26 = 47 % Republican, 27 + 26 = 53% Democrat (52% independent)
    - generation z: 20.88%
      - 17 + 26 = 43 % Republican, 31 + 26 = 57% Democrat (52% independent)
    =========
    Extra noise is then added based on each bucket having an equal chance of being chosen.
    0: silent republican = .0549 * .52 = .0264
    1: silent democrat = .0549 * .48 = .0263
    2: baby boomers republican = .2058 * .515 = .1060
    3: baby boomers democrat = .2058 * .485 = .0998
    4: generation x republican = .1961 * .52 = .1020
    5: generation x democrat = .1961 * .49 = .0961
    6: millenials republican = .2167 * .47 = .1018
    7: millenials democrat = .2167 * .53 = .1149
    8: generation z republican = .2088 * .43 = .0897
    9: generation z democrat = .2088 * .57 = .119
    """
    groups = {
        0: 0.0264,
        1: 0.0263,
        2: 0.1060,
        3: 0.0998,
        4: 0.1020,
        5: 0.0961,
        6: 0.1018,
        7: 0.1149,
        8: 0.0897,
        9: 0.119,
    }
    # make noise positive to follow a proper cdf
    noise = abs(random.normalvariate(0, noise_std_dev) / 10)
    # Add noise to the key (age group) based on a normal distribution
    prob = random.random() * (sum(groups.values()) + noise)
    cumulative_prob = 0
    for key, value in groups.items():
        cumulative_prob += value + noise
        if prob <= cumulative_prob:
            return key


def get_attribute_to_users_dict(user_attribute_dict):
    dic = defaultdict(list)
    for k, v in user_attribute_dict.items():
        dic[v].append(k)

    return dic


def get_user_to_attribute_dict(fn: str, path: str, attribute: str) -> Dict:
    """
    Summary:
    This function parses the user profile partitions and creates an attribute dic
    for the selected attribute

    If "gender" is the attribute and it has already been pre-processed - provide it as a path
    path: path to profile_gender.csv

    If the file isn't available, it will generate an attribute dictionary for the attribute

    Return:
    WORKS ONLY FOR GENDER
    gender_dict: dictionary with user IDs as keys and 0 (female) or 1 (male) values
    """

    try:
        # with open(path, 'r', encoding="ISO-8859-1") as f:
        #     contents = f.read()
        print(f"Trying to load {path}")
        attribute_df = pd.read_csv(path)
    except Exception as e:
        print(f"ERROR cannot find: {path} -> {e}")
        DATA_INIT_PATH = (
            "/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/userProfile/"
        )
        path_user_profile = DATA_INIT_PATH + "userProfile"

        # get all user profile partition files
        txt_files = [
            os.path.join(path_user_profile, f)
            for f in os.listdir(path_user_profile)
            if os.path.isfile(os.path.join(path_user_profile, f))
        ]

        # there are 14 features between one user row and the next (+1 new line for next member)
        USER_ROW_LINE_BUFFER = 15
        user_profile_df = pd.DataFrame()

        # parse through all user partitions
        for t in txt_files:
            with open(t, "r", encoding="ISO-8859-1") as f:
                contents = f.read()

            # split on line since each feature is in new line
            split_content = contents.split("\n")[:-1]

            reshaped_content = np.reshape(
                split_content,
                (int(len(split_content) / USER_ROW_LINE_BUFFER), USER_ROW_LINE_BUFFER),
            )
            df = pd.DataFrame(reshaped_content)
            df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            df.columns = df.columns.str.lstrip("# ")
            user_profile_df = user_profile_df.append(df)
        print("user_profile_df columns: ", user_profile_df.columns)
        if attribute in user_profile_df.columns:
            attribute_df = user_profile_df[["id", attribute]].reset_index(drop=True)
        else:
            print("Attribute not found in user_profile dfs.")
            attribute_df = user_profile_df[["id"]].reset_index(drop=True)
        uid_map = mapped_uid()
        attribute_df.id = attribute_df.id.map(uid_map)  # mapping user id

        # gender conversion for weibo into categorical 1 or 0
        if attribute == "gender" and fn == "weibo":
            gender_conversion_dict = {"m": 1, "f": 0}
            attribute_df[attribute] = attribute_df[attribute].map(
                gender_conversion_dict
            )
        elif attribute == "age" and fn == "weibo":
            attribute_df[attribute] = attribute_df["id"].apply(
                lambda x: randomly_assign_age_group()
            )
        elif attribute == "political_lean" and fn == "weibo":
            attribute_df[attribute] = attribute_df["id"].apply(
                lambda x: randomly_assign_political_lean()
            )
        elif attribute == "age_x_pol_affln" and fn == "weibo":
            attribute_df[attribute] = attribute_df["id"].apply(
                lambda x: randomly_assign_age_x_pol_affln()
            )
        elif attribute == "gender_x_pol_affln" and fn == "weibo":
            attribute_df[attribute] = attribute_df["id"].apply(
                lambda x: randomly_assign_gender_x_pol_affln()
            )
        elif attribute == "age_x_pol_affln_with_noise" and fn == "weibo":
            attribute_df[attribute] = attribute_df["id"].apply(
                lambda x: randomly_assign_age_x_pol_affln_with_noise()
            )
        # store the processed data as csv
        print("Dumping attribute_df to ", path)
        attribute_df.to_csv(path, index=False)

    attribute_dict = pd.Series(
        attribute_df[attribute].values, index=attribute_df.id
    ).to_dict()

    return attribute_dict


######## END SECTION IS FROM SPRING 2024 ###########


def run(fn, attribute, sampling_perc, log):
    print("Reading the network")
    cwd = os.getcwd()
    # txt_file_path = '/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/weibo_network.txt' ###
    txt_file_path = "/opt/data/weibo_network.txt"  ###
    # txt_file_path = '/media/yuting/TOSHIBA EXT/digg/sampled/digg_network_sampled.txt'  ###
    g = ig.Graph.Read_Ncol(txt_file_path)
    print("Completed reading the network.")

    train_set_file = "/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/train_set_fair_gender_fps_v4.txt"  # set the train_set file according to different attribute
    # train_set_file = '/media/yuting/TOSHIBA EXT/digg/sampled/trainset_fair_age_fps.txt'  # set the train_set file according to different attribute
    attribute_csv = "/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/profile_gender.csv"  #  !!! use v6  set attribute csv file to write corresponding attribute
    # attribute_csv = '/media/yuting/TOSHIBA EXT/digg/profile_age_v3.csv'
    user_attribute_dict = get_attribute_dict(fn, attribute_csv, attribute)

    # group statistics
    attribute_grouped = defaultdict(list)
    for k, v in user_attribute_dict.items():
        attribute_grouped[v].append(k)
    print("generate grouped nodes")

    with open(
        "/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/train_cascades.txt",
        "r",
    ) as f, open(train_set_file, "w") as train_set:
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
            if fn == "weibo":
                cascade_nodes = list(map(lambda x: x.split(" ")[0], cascade[1:]))
                cascade_times = list(
                    map(
                        lambda x: int(
                            (
                                (
                                    datetime.strptime(
                                        x.replace("\r", "").split(" ")[1],
                                        "%Y-%m-%d-%H:%M:%S",
                                    )
                                    - datetime.strptime("2011-10-28", "%Y-%m-%d")
                                ).total_seconds()
                            )
                        ),
                        cascade[1:],
                    )
                )
            else:
                cascade_nodes = list(map(lambda x: x.split(" ")[0], cascade))
                cascade_times = list(
                    map(lambda x: int(x.replace("\r", "").split(" ")[1]), cascade)
                )

            # ---- Remove retweets by the same person in one cascade
            cascade_nodes, cascade_times = remove_duplicates(
                cascade_nodes, cascade_times
            )

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

            store_samples(
                fn,
                cascade_nodes[1:],
                cascade_times[1:],
                initiators,
                train_set,
                op_time,
                user_attribute_dict,
                attribute_grouped,
                attribute,
                sampling_perc,
            )

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

    np.seterr(divide="ignore", invalid="ignore")

    # ------ Store node characteristics
    # node_feature_fair_gender = '/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/FPS/node_features.csv'
    node_feature_fair_age = "/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/FPS/node_feature_age_fps.csv"
    # node_feature_fair_age = '/media/yuting/TOSHIBA EXT/digg/sampled/node_feature_age_fps.csv'
    pd.DataFrame(
        {
            "Node": g.vs["name"],
            "Kcores": kcores,
            "Participated": g.vs["Cascades_participated"],
            "Avg_Cascade_Size": a / b,
        }
    ).to_csv(node_feature_fair_age, index=False)

    print("Finished storing node characteristics")

    # # ------ Derive incremental node dictionary
    # graph = pd.read_csv('/media/yuting/TOSHIBA EXT/digg/' + fn + "_network.txt", sep=" ")
    # graph.columns = ["node1", "node2", "weight"]
    # all = list(set(graph["node1"].unique()).union(set(graph["node2"].unique())))
    # dic = {int(all[i]): i for i in range(0, len(all))}
    # with open('/media/yuting/TOSHIBA EXT/digg/' + fn + "_incr_dic.json", "w") as json_file:
    #     json.dump(dic, json_file)


if __name__ == "__main__":
    with open("time_log.txt", "a") as log:
        input_fn = "weibo"
        sampling_perc = 120
        run(input_fn, "gender", sampling_perc, log)
