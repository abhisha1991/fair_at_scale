"""
Weigh all networks based on weighted cascade, and derive the attribute file required for IMM
"""

import json
import time

import pandas as pd


def run(fn, log):
    start = time.time()
    # --- Read graph
    with open(fn.capitalize() + "/wc_" + fn + "_attribute.txt", "w") as attribute_file:
        graph = pd.read_csv(fn.capitalize() + "/Init_Data/" + fn + "_network.txt", sep=" ")
        if graph.shape[1] > 2:
            graph = graph.drop(graph.columns[2], 1)
        graph.columns = ["node1", "node2"]

        # --- Compute influence weight
        outdegree = graph.groupby("node1").agg('count').reset_index()
        outdegree.columns = ["node1", "outdegree"]

        outdegree["outdegree"] = 1 / outdegree["outdegree"]
        outdegree["outdegree"] = outdegree["outdegree"].apply(lambda x: float('%s' % float('%.6f' % x)))

        # --- Assign it
        graph = graph.merge(outdegree, on="node1")

        # --- Find all nodes to create incremental ids for IMM
        all_nodes = list(set(graph["node1"].unique()).union(set(graph["node2"].unique())))

        dic = {int(all_nodes[i]): i for i in range(0, len(all_nodes))}
        graph['node1'] = graph['node1'].map(dic)
        graph['node2'] = graph['node2'].map(dic)

        # --- Store the ids to translate the seeds_result of IMM
        with open(fn.capitalize() + "/" + fn + "_incr_dic.json", "w") as json_file:
            json.dump(dic, json_file)

        # --- Store
        graph = graph[["node2", "node1", "outdegree"]]
        graph.to_csv(fn.capitalize() + "/wc_" + fn + "_network.txt", header=False, index=False, sep=" ")
        log.write(f"Time for wc {fn} network:{str(time.time() - start)}\n")

        attribute_file.write(f"n={str(len(all_nodes) + 1)}\n")
        attribute_file.write(f"m={str(graph.shape[0])}\n")
