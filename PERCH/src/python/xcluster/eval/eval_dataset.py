"""
Copyright (C) 2017 University of Massachusetts Amherst.
This file is part of "xcluster"
http://github.com/iesl/xcluster
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import errno
import time

import numpy as np

from Algorithm.PERCH.src.python.xcluster.models.PNode import PNode
from Algorithm.PERCH.src.python.xcluster.models.pruning_heuristics import pick_k_min_dist, \
    pick_k_max_dist, \
    pick_k_point_counter, pick_k_local_mean_cost, pick_k_global_k_mean_cost, \
    pick_k_approx_km_cost
from Algorithm.PERCH.src.python.xcluster.utils.serialize_trees import \
    serliaze_collapsed_tree_to_file_with_point_ids


def mkdir_p_safe(dir):
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def load_data(filename):
    with open(filename, 'r') as f:
        index = 0
        for line in f:
            splits = line.strip().split('\t')
            pid, l, vec = index, splits[-1], np.array([float(x)
                                                       for x in splits[:-1]])
            index += 1
            yield ((vec, l, pid))


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Evaluate PERCH clustering.')
    # parser.add_argument('--input', '-i', type=str,
    #                     help='Path to the dataset.')
    # parser.add_argument('--outdir', '-o', type=str,
    #                     help='the output directory')
    # parser.add_argument('--algorithm', '-a', type=str,
    #                     help='The name of the algorithm to evaluate.')
    # parser.add_argument('--dataset', '-n', type=str,
    #                     help='The name of the dataset.')
    # parser.add_argument('--max_leaves', '-L', type=str,
    #                     help='The maximum number of leaves.', default=None)
    # parser.add_argument('--clusters', '-k', type=str,
    #                     help='The number of clusters to pick.', default=None)
    # parser.add_argument('--pick_k', '-m', type=str,
    #                     help='The heuristic by which to pick clusters',
    #                     default=None)
    # parser.add_argument('--exact_dist_thres', '-e', type=int,
    #                     help='# of points to search using exact dist threshold',
    #                     default=10)
    #
    # args = parser.parse_args()
    input = 'E:/Project/Python/data/small/glass.txt'
    output = 'F:/project/Experiment/PERCH'
    algorithm = 'PERCH'
    dataset = 'glass'
    max_leaves = float('INf')
    clusters = None
    pick_k = None
    exact_dist_thres = 10

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    exp_dir_base = os.path.join(output)
    mkdir_p_safe(exp_dir_base)

    clustering_time_start = time.time()
    clustering_time_per_point = []
    counter = 0
    L = float('INf')
    L_was_defined = False
    collapsibles = [] if L < float("Inf") else None
    root = PNode(exact_dist_thres=10)
    data = load_data(input)
    for pt in data:
        root = root.insert(pt, collapsibles=collapsibles, L=L)

    clustering_time_end = time.time()
    clustering_time_elapsed = clustering_time_end - clustering_time_start
    # First save the tree structure to a file to evaluate dendrogram purity.
    pick_k_time = 0
    K = 4
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

    # Write clustering
    clustering = root.clusters()
    predicted_clustering = []
    gold_clustering = []
    idx = 0
    c_idx = 0
    for c in clustering:
        pts = [pt for l in c.leaves() for pt in l.pts]
        for pt in pts:
            predicted_clustering.append((idx, c_idx))
            gold_clustering.append((idx, pt[1]))
            idx += 1
        c_idx += 1

    with open(os.path.join(exp_dir_base, 'predicted.txt'), 'w') as fout:
        for p in predicted_clustering:
            fout.write('{}\t{}\n'.format(p[0], p[1]))
    with open(os.path.join(exp_dir_base, 'gold.txt'), 'w') as fout:
        for g in gold_clustering:
            fout.write('{}\t{}\n'.format(g[0], g[1]))
    serliaze_collapsed_tree_to_file_with_point_ids(
        root, os.path.join(exp_dir_base, 'tree-pick-k.tsv'))
