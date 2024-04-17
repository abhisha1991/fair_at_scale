"""
Evaluate seed sets based on DNI and precision
"""

import glob
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict

import numpy as np
import pandas as pd
from extract_feats_and_trainset import (
    compute_fair,
    get_attribute_dict,
    get_attribute_to_users_dict,
    get_user_to_attribute_dict,
)
from fair_iminfector import softmax
from variables import (
    ATTRIBUTE,
    DATA_SEEDS_PATH,
    GENDER_ATTRIBUTE_CSV,
    INPUT_FN,
    TEST_CASCADES,
)


def count_distinct_nodes_influenced(seed_set_cascades: Dict) -> int:
    """
    Measure the number of distinct nodes in the test cascades started of the seed set
    """
    combined = set()
    for v in seed_set_cascades.values():
        combined = combined.union(set().union(*v))
    return len(combined)


######## BEGIN SECTION IS FROM SPRING 2024 ###########


def process_batch(batch_D):
    """
    Processes a single batch of data.

    Args:
        batch_D: A numpy array representing a batch of data.

    Returns:
        The processed batch of data.
    """
    # Normalize & apply softmax
    batch_D -= np.abs(batch_D.max(axis=1, keepdims=True))
    batch_D = softmax(batch_D)

    # Round and convert to absolute values
    batch_D = np.around(batch_D, 3)
    batch_D = np.abs(batch_D)

    return batch_D


def batched_fair_im_process_D_parallel(D, batch_size=1000):
    print(f"Processing {len(D)} batches of {batch_size}")
    with Pool() as pool:
        processed_batches = pool.map(
            process_batch,
            [(D[i : i + batch_size]) for i in range(0, len(D), batch_size)],
        )
    return np.concatenate(processed_batches, axis=0)


def count_distinct_nodes_influenced_in_cascades(seed_to_cascades: Dict) -> int:
    """
    Summary:
    Finds the DNI score for the seed set cascades - this is the main metric
    used to evaluate the spread from the seed cascade

    Returns:
    Returns the number of distinct nodes influenced by the seed set cascades

    """
    combined_dni_all_cascades = set()
    for v in seed_to_cascades.values():
        dni_in_cascade_for_seed = set().union(*v)
        combined_dni_all_cascades = combined_dni_all_cascades.union(
            dni_in_cascade_for_seed
        )

    return len(combined_dni_all_cascades)


def get_distinct_nodes_influenced_in_cascades(seed_to_cascades: Dict) -> set:
    """
    Summary:
    Finds the DNI score for the seed set cascades - this is the main metric
    used to evaluate the spread from the seed cascade

    Returns:
    Returns the distinct nodes influenced by the seed set cascades

    """
    combined_dni_all_cascades = set()
    for v in seed_to_cascades.values():
        dni_in_cascade_for_seed = set().union(*v)
        combined_dni_all_cascades = combined_dni_all_cascades.union(
            dni_in_cascade_for_seed
        )

    return combined_dni_all_cascades


def eval_run2():
    """
    Runs eval_run but includes the fairness metric.
    """
    user_attribute_dict = get_user_to_attribute_dict(
        INPUT_FN, GENDER_ATTRIBUTE_CSV, ATTRIBUTE
    )
    attr_users_dict = get_attribute_to_users_dict(user_attribute_dict)
    print("generate grouped nodes")

    # go through all seed files generated under seeds folder
    for seed_set_file in glob.glob(DATA_SEEDS_PATH + "*.txt"):
        print(f"\n\nRunning eval for {seed_set_file}")
        print("------------------")
        # Compute precision
        with open(seed_set_file, "a") as current_seed_file, open(
            seed_set_file, "r"
        ) as current_seed_file_read:
            current_seed_file.write(seed_set_file + "\n")
            l = current_seed_file_read.read().replace("\n", " ")
            seed_set_all = [x for x in l.split(" ") if x != ""]

            # Estimate the spread of that seed set in the test cascades
            spreading_of_set, f_scores, step, upper_limit = {}, {}, 50, 1100

            for seed_set_size in range(step, upper_limit, step):
                curr_seeds = seed_set_all[0:seed_set_size]

                # Init empty list of cascades for each seed
                curr_seed_cascades, seed_set = {str(s): [] for s in curr_seeds}, set()

                # Fill the seed_cascades
                with open(TEST_CASCADES) as test_cascades:
                    for line in test_cascades:
                        cascade = line.split(";")
                        op_id = cascade[1].split(" ")[0]
                        cascade = set(map(lambda x: x.split(" ")[0], cascade[2:]))
                        if op_id in curr_seed_cascades:
                            curr_seed_cascades[op_id].append(cascade)
                            seed_set.add(op_id)

                # Fill the seed_cascades
                seed_cascades_dic = {
                    seed: curr_seed_cascades[seed]
                    for seed in seed_set
                    if len(curr_seed_cascades[seed]) > 0
                }

                print(f"Seeds found in current seed group: {len(seed_cascades_dic)}")
                current_seed_file.write(str(len(seed_cascades_dic)) + "\n")
                # ------- compute fair
                spreading_of_set[seed_set_size] = (
                    count_distinct_nodes_influenced_in_cascades(seed_cascades_dic)
                )
                influenced_nodes = get_distinct_nodes_influenced_in_cascades(
                    seed_cascades_dic
                )
                f_scores[seed_set_size] = compute_fair(
                    influenced_nodes, user_attribute_dict, attr_users_dict, "age"
                )

            df = pd.DataFrame(
                {
                    "Influencer set size": list(spreading_of_set.keys()),
                    "DNI": list(spreading_of_set.values()),
                    "FScore": list(f_scores.values()),
                }
            )
            df.to_csv(
                seed_set_file.replace("Seeds", "Spreading").replace(
                    "Seeds/", "Spreading/"
                ),
                index=False,
            )


######## END SECTION IS FROM SPRING 2024 ###########


def run(fn, log):
    # for seed_set_file in glob.glob(fn.capitalize() + "/FAC/Seeds/*"):
    for seed_set_file in glob.glob(
        "/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/FPS/Seeds/*.txt"
    ):
        print(seed_set_file)
        # --- Compute precision
        print("------------------")
        with open(seed_set_file, "a") as current_seed_file, open(
            seed_set_file, "r"
        ) as current_seed_file_read:
            current_seed_file.write(seed_set_file + "\n")
            l = current_seed_file_read.read().replace("\n", " ")
            seed_set_all = [x for x in l.split(" ") if x != ""]

            # ------- Estimate the spread of that seed set in the test cascades
            spreading_of_set, step, upper_limit = {}, 50, 1100

            for seed_set_size in range(step, upper_limit, step):
                seeds = seed_set_all[0:seed_set_size]

                # ------- List of cascades for each seed
                seed_cascades, seed_set = {str(s): [] for s in seeds}, set()

                # ------- Fill the seed_cascades
                # with open(f"{fn.capitalize()}/Init_Data/test_cascades.txt") as test_cascades:
                with open(
                    "/gdrive/MyDrive/FairInfluenceMaximization/data/Data/Weibo/Init_Data/test_cascades.txt"
                ) as test_cascades:
                    for line in test_cascades:
                        cascade = line.split(";")
                        op_id = cascade[1].split(" ")[0]
                        cascade = set(map(lambda x: x.split(" ")[0], cascade[2:]))
                        if op_id in seed_cascades:
                            seed_cascades[op_id].append(cascade)
                            seed_set.add(op_id)

                # ------- Fill the seed_cascades
                seed_set_cascades = {
                    seed: seed_cascades[seed]
                    for seed in seed_set
                    if len(seed_cascades[seed]) > 0
                }
                print(f"Seeds found: {len(seed_set_cascades)}")
                current_seed_file.write(str(len(seed_set_cascades)) + "\n")

                spreading_of_set[seed_set_size] = count_distinct_nodes_influenced(
                    seed_set_cascades
                )
            pd.DataFrame(
                {
                    "Feature": list(spreading_of_set.keys()),
                    "Cascade Size": list(spreading_of_set.values()),
                }
            ).to_csv(
                seed_set_file.replace("Seeds", "Spreading").replace(
                    "Seeds/", "Spreading/"
                ),
                index=False,
            )


def run2(fn, log, attribute):

    # attribute_csv = '/media/yuting/TOSHIBA EXT/digg/profile_'+ attribute + '_v6.csv'
    attribute_csv = (
        "/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/processed4maxmization/weibo/profile_"
        + attribute
        + "v3.csv"
    )
    user_attribute_dict = get_attribute_dict("weibo", attribute_csv, attribute)

    # group statistics
    attribute_grouped = defaultdict(list)
    for k, v in user_attribute_dict.items():
        attribute_grouped[v].append(k)
    print("generate grouped nodes")

    # load test cascade
    seed_cascades, seed_set_total = defaultdict(list), set()
    # with open("/media/yuting/TOSHIBA EXT/digg/sampled/test_cascades_sampled.txt") as test_cascades:
    with open(
        "/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/processed4maxmization/test_cascades.txt"
    ) as test_cascades:
        for line in test_cascades:
            cascade = line.split(";")
            op_id = cascade[1].split(" ")[0]
            cascade = set(map(lambda x: x.split(" ")[0], cascade[2:]))
            if len(cascade) <= 10:
                continue
            seed_cascades[op_id].extend(cascade)
            seed_set_total.add(op_id)

    # for seed_set_file in glob.glob(fn.capitalize() + "/FAC/Seeds/*"):
    # for seed_set_file in glob.glob("Data/seeds_digg/final_seeds_sampled_" +"*" + attribute + "*.txt"):
    for seed_set_file in glob.glob("Data/seeds/final_seedds_" + "*" + "v2_new" + "*"):
        # for seed_set_file in glob.glob("Data/seeds/final_seeds_kcore.txt"):
        print(seed_set_file)
        # --- Compute precision
        print("------------------")
        # with open(seed_set_file, "a") as current_seed_file, open(seed_set_file, "r") as current_seed_file_read:
        with open(seed_set_file, "r") as current_seed_file_read, open(
            "results_flipped.txt", "a"
        ) as result_file:
            # current_seed_file.write(seed_set_file + "\n")
            result_file.write(seed_set_file + "_" + attribute + "\n")
            l = current_seed_file_read.read().replace("\n", " ")
            seed_set_all = [x for x in l.split(" ") if x != ""]

            # ------- Estimate the spread of that seed set in the test cascades
            # spreading_of_set, step, upper_limit = {}, 20, 120
            spreading_of_set, step, upper_limit = {}, 50, 1050

            for seed_set_size in range(step, upper_limit, step):
                seeds = seed_set_all[0:seed_set_size]

                # ------- List of cascades for each seed
                # seed_cascades, seed_set = {str(s): [] for s in seeds}, set()

                # # ------- Fill the seed_cascades
                # # with open(f"{fn.capitalize()}/Init_Data/test_cascades.txt") as test_cascades:
                # with open("/gdrive/MyDrive/FairInfluenceMaximization/data/weibodata/processed4maxmization/test_cascades.txt") as test_cascades:
                #     for line in test_cascades:
                #         cascade = line.split(";")
                #         op_id = cascade[1].split(" ")[0]
                #         cascade = set(map(lambda x: x.split(" ")[0], cascade[2:]))
                #         if op_id in seed_cascades:
                #             seed_cascades[op_id].append(cascade)
                #             seed_set.add(op_id)

                # ------- Fill the seed_cascades
                # seed_set_cascades = {seed: seed_cascades[seed] for seed in seed_set if len(seed_cascades[seed]) > 0}
                # print(f"Seeds found: {len(seed_set_cascades)}")
                # current_seed_file.write(str(len(seed_set_cascades)) + "\n")

                seed_set_cascades = {
                    seed: seed_cascades[seed]
                    for seed in seed_set_total.intersection(seeds)
                }
                influenced_nodes = set.union(
                    *map(set, list(seed_set_cascades.values()))
                )

                # spreading_of_set[seed_set_size] = count_distinct_nodes_influenced(seed_set_cascades)
                # spreading_of_set[seed_set_size] = len(influenced_nodes)

                # ------- compute fair
                f_score = compute_fair(
                    influenced_nodes, user_attribute_dict, attribute_grouped, attribute
                )
                # fairscore_of_set = {}
                # fairscore_of_set[seed_set_size] = f_score

                result_file.write(
                    str(seed_set_size)
                    + " "
                    + str(len(influenced_nodes))
                    + " "
                    + str(f_score)
                    + "\n"
                )


if __name__ == "__main__":

    with open("time_log.txt", "a") as log:
        fn = "weibo"
        attribute = "gender"
        run2(fn, log, attribute)
