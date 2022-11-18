# -*- coding: utf-8 -*-
import argparse
import os
import time

import evaluation
import extract_feats_and_trainset
import iminfector
import infector
import preprocess_for_imm
import preprocessing
import rank_nodes


def get_parameters() -> [int, float, int, int, int]:
    """
    This function creates and gets arguments for sampling percentage, learning rate, number of epochs,
    embedding size, and number of negative samples.

    :return: list with argument values for sampling percentage, learning rate, number of epochs, embeddings size, and number of negative samples
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling-perc', type=int, default=120, help='')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='')
    parser.add_argument('--n-epochs', type=int, default=10, help='')
    parser.add_argument('--embedding-size', type=int, default=50, help='')
    parser.add_argument('--num-neg-samples', type=int, default=10, help='')
    args = parser.parse_args()

    return int(args.sampling_perc), float(args.learning_rate), int(args.n_epochs), int(args.embedding_size), int(
        args.num_neg_samples)


def run_processes(input_log, input_fn="weibo"):
    """
    This function kicks off all the processes in the project.

    :param input_log: log file
    :param input_fn: name of dataset/process
    """
    sampling_perc, learning_rate, n_epochs, embedding_size, num_neg_samples = get_parameters()
    print(sampling_perc, learning_rate, n_epochs, embedding_size, num_neg_samples)

    # if (not os.path.isfile(os.getcwd() + "/Weibo/Init_Data/train_cascades.txt")) or (
    #         not os.path.isdir(os.getcwd() + "/Weibo/Init_Data/FAC")):
    #     preprocessing.run(input_fn, input_log)
    # extract_feats_and_trainset.run(input_fn, 'region', sampling_perc, input_log)
    # preprocess_for_imm.run(input_fn, input_log)
    # rank_nodes.run(input_fn)
    # infector.run(input_fn, learning_rate, n_epochs, embedding_size, num_neg_samples, input_log)
    # iminfector.run(input_fn, embedding_size, input_log)
    evaluation.run2(input_fn, input_log, 'region')


if __name__ == '__main__':
    start = time.time()
    if not os.path.isdir("Data"):
        os.makedirs("Data")
    abspath = os.path.abspath(__file__)
    directory_name = os.path.dirname(abspath)
    # os.chdir(os.path.join(directory_name, "..", "Data"))
    os.chdir(os.path.join(directory_name, "Data"))
    print('Current working directory for the code', os.getcwd())

    with open("time_log.txt", "a") as log:
        run_processes(log)
