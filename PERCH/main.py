import os
import time

import numpy as np

from ExperimentTool.Entity.Basis import Task, Record, RecordType
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing, print_estimate
from models.PNode import PNode
from models.pruning_heuristics import pick_k_min_dist, \
    pick_k_max_dist, \
    pick_k_point_counter, pick_k_local_mean_cost, pick_k_global_k_mean_cost, \
    pick_k_approx_km_cost


def get_assignment(cluster):
    # Write clustering
    label_pred = []
    label_true = []
    idx = 0
    c_idx = 0
    for c in cluster:
        pts = [pt for l in c.leaves() for pt in l.pts]
        for pt in pts:
            label_pred.append(c_idx)
            label_true.append(pt[1])
            idx += 1
        c_idx += 1
    return label_true, label_pred


def PERCH(data, label, K):
    clustering_time_start = time.time()
    clustering_time_per_point = []
    counter = 0
    L = float('INf')
    L_was_defined = False
    collapsibles = [] if L < float("Inf") else None
    root = PNode(exact_dist_thres=10)
    for i in range(len(label)):
        pt = (np.array(data[i]), label[i], i)
        root = root.insert(pt, collapsibles=collapsibles, L=L)

    pick_k_method = None
    clustering_time_end = time.time()
    runTime = clustering_time_end - clustering_time_start
    # First save the tree structure to a file to evaluate dendrogram purity.
    pick_k_time = 0
    if K:
        start_pick_k = time.time()
        if collapsibles is None:
            collapsibles = root.find_collapsibles()
        if not L_was_defined:
            L = root.point_counter

        pick_k_method = 'approxKM'
        if pick_k_method == 'approxKM':
            pick_k_approx_km_cost(root, collapsibles, L, K)
        elif pick_k_method == 'pointCounter':
            pick_k_point_counter(root, collapsibles, K)
        elif pick_k_method == 'globalKM':
            pick_k_global_k_mean_cost(root, collapsibles, L, K)
        elif pick_k_method == 'localKM':
            pick_k_local_mean_cost(root, collapsibles, L, K)
        elif pick_k_method == 'maxD':
            pick_k_max_dist(root, collapsibles, L, K)
        elif pick_k_method == 'minD':
            pick_k_min_dist(root, collapsibles, L, K)
        else:
            print('UNKNOWN PICK K METHOD USING approxKM')
            pick_k_approx_km_cost(root, collapsibles, L, K)

        end_pick_k = time.time()
        pick_k_time = end_pick_k - start_pick_k
        label_true, label_pred = get_assignment(root.clusters())
    return label_true, label_pred, runTime


@ExpMonitor(expId='PERCH', algorithmName='PERCH', storgePath=os.pardir + '/Experiment')
def run(task,k):
    record = Record()
    data, label,K = data_processing(task.filePath,drop_duplicates=False)
    label_true, label_pred, runTime = PERCH(data, label, k)
    print(len(label_true))
    print_estimate(label_true, label_pred, task.dataName, int(task.iterIndex), 0, runTime)
    record.save_time(runTime)
    record.save_output(RecordType.assignment, label_true, label_pred)
    return {'record': record}


if __name__ == "__main__":
    path = '../data/small'
    for file in os.listdir(path):
        data, label, K = data_processing(path + '/' + file,drop_duplicates=False)
        # for k in range(K - 5 if K - 5 > 1 else 1, K + 5):
        for testIndex in range(50):
            task = Task(K, testIndex, file.split('.')[0], path + '/' + file)
            run(task=task, k=10)
