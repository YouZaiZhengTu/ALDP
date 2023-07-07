# Copyright 2021 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import time

import numpy as np

from ExperimentTool.Entity.Basis import Record, RecordType, Task
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing, print_estimate
from src.IKMapper import IKMapper
from src.INode import INode


def build_streKhc_tree(data, label, m, psi, t):
    """Create trees over the same points.
    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.

    Args:
        data_path - path to dataset.
        m - numuber of point to intitial ik metrix
        psi - particial size  to build isolation kernel mapper
        t - sample size to build isolation kernel mapper

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """
    root = INode()
    train_dataset = []
    L = len(data)
    vec = []
    for i, d in enumerate(data):
        vec.append((label[i], i, d))
    for i, pt in enumerate(vec, start=1):
        if i <= m:
            train_dataset.append(pt)
            if i == m:
                ik_mapper = IKMapper(t=t, psi=psi)
                ik_mapper = ik_mapper.fit(np.array(
                    [pt[2] for pt in train_dataset]))
                for j, train_pt in enumerate(train_dataset, start=1):
                    l, pid, ikv = train_pt[0], train_pt[1], ik_mapper.embeding_mat[j - 1]
                    root = root.insert((l, pid, ikv), L=L,
                                       t=t, delete_node=True)
        else:
            l, pid = pt[:2]
            root = root.insert((l, pid, ik_mapper.transform(
                pt[2])), L=L, t=t, delete_node=True)
    return root


def save_data(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, 'score.tsv')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'purity',
                "max_psi",
            ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%.2f\t%s\n' % (
            args['dataset'],
            args['algorithm'],
            args['purity'],
            args["max_psi"],
        ))


def save_grid_data(args, exp_dir_base):
    file_path = os.path.join(exp_dir_base, 'grid_score.tsv')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'purity',
                "psi",
            ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%.2f\t%s\n' % (
            args['dataset'],
            args['algorithm'],
            args['purity'],
            args["psi"],
        ))


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate StreaKHC clustering.')
    parser.add_argument('--input', '-i', type=str,
                        help='<Required> Path to the dataset.', required=True)
    parser.add_argument('--outdir', '-o', type=str,
                        help='<Required> The output directory', required=True)
    parser.add_argument('--dataset', '-n', type=str,
                        help='<Required> The name of the dataset', required=True)
    parser.add_argument('--sample_size', '-t', type=int, default=300,
                        help='<Required> Sample size for isolation kernel mapper')
    parser.add_argument('--psi', '-p', nargs='+', type=int, required=True,
                        help='<Required> Particial size for isolation kernel mapper')
    parser.add_argument('--train_size', '-m', type=int, required=True,
                        help='<Required> Initial used data size to build Isolation Kernel Mapper')
    args = parser.parse_args()
    # grid_search_inode(data_path=args.input, m=args.train_size, t=args.sample_size, psi=args.psi,
    #                   file_name=args.dataset, exp_dir_base=args.outdir)
def getAssignment(root, k, node_num):
    label_true = [-1 for i in range(node_num)]
    label_pred = [-1 for i in range(node_num)]
    roots = [root]
    patitions = []
    patitions.append(root)
    flag = 0
    while len(patitions) < k:
        bigest_c_size = 0
        bigest_c_index = 0
        for index in range(len(patitions)):
            if len(patitions[index].pts) > bigest_c_size:
                bigest_c_size = len(patitions[index].pts)
                bigest_c_index = index
        root = patitions.pop(bigest_c_index)
        if root.children[0] == None and root.children[1] == None:
            patitions.append(root)
            flag += 1
            if flag > 2:
                print('circle!')
        else:
            flag = 0
            if root.children[0] != None:
                patitions.append(root.children[0])
            if root.children[1] != None:
                patitions.append(root.children[1])
    for i, root in enumerate(patitions):
        for pt in root.pts:
            label_true[pt[1]] = pt[0]
            label_pred[pt[1]] = i
    # print('true: %s, pred: %s'%( str(-1 in label_true), str(-1 in label_pred)))
    return label_true, label_pred


def StreaKHC(data, label, ps, t, m, k):
    root = build_streKhc_tree(data, label, m, ps, t)
    label_true, label_pred = getAssignment(root, k, len(data))
    return label_true, label_pred

@ExpMonitor(expId='streaKHC', algorithmName='streaKHC', storgePath=os.pardir + '/Experiment')
def run(task):
    record = Record()
    data, label, K = data_processing(task.filePath)
    start = time.time()
    for ps in [3, 5, 10, 17, 21, 25]:
        label_true, label_pred = StreaKHC(data, label, ps, K, len(data), task.params)
        # print_estimate(label_true, label_pred, task.dataName, int(task.iterIndex), 0, time.time() - start,str(task.params))
        record.save_time(time.time() - start)
        record.save_output(RecordType.assignment, label, label_pred,ps)
    print_estimate(label_true, label_pred, task.dataName, int(task.iterIndex), 0, time.time() - start, str(task.params))
    return {'record': record}


if __name__ == "__main__":
    path = '../data/small'
    taskList = []
    for file in os.listdir(path):
        data, label, K = data_processing(path + '/' + file)
        # low, height = 0, 0
        # if K >= 7:
        #     low = K - 7 if K - 7 > 1 else 1
        #     height = K + 7
        # else:
        #     low = 2
        #     height = 13
        # for k in range(low, height):
        for testIndex in range(5):
            taskList.append(Task(K, testIndex, file.split('.')[0], path + '/' + file))
    for task in taskList:
        run(task)
    print('done')
