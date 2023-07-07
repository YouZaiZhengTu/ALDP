import os
from time import time

import numpy as np

from Algorithm.construction import construction
from Algorithm.densityPeak import findDensityPeak
from Algorithm.findNNs import findNNs
from Algorithm.rebuild_sct import rebuild
from Entities.Node import Node
from ExperimentTool.Entity.Basis import Record, RecordType, Task
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing, print_estimate
from aldp.Entities.experiment import get_distribute


@ExpMonitor(expId='ALDP', algorithmName='ALDP', storgePath=os.pardir + '/Experiment')
def run(task: Task):
    # 数据提取和处理
    record = Record()
    data, label, K = data_processing(path + '/' + dataName)
    nodeList = {}
    for i in range(len(data)):
        nodeList[i] = Node(i, data[i], label[i], label[i])
    iteration = 0
    # 循环迭代
    start = time()
    while len(nodeList) > 3 and len(nodeList) > K:
        nns, snns = findNNs(nodeList=nodeList, k=3)
        roots = construction(nodeList=nodeList, nns=nns, iteration=iteration)
        rebuild(snns, roots, nodeList, iteration)
        nodeList = findDensityPeak(roots, task.params, iteration=iteration)
        label_true, label_pred = get_distribute(roots, len(label))
        end = time()
        print_estimate(label_true, label_pred, task.dataName, task.iterIndex, 0, end - start)

        record.save_time(end - start)
        record.save_output(RecordType.assignment, label_true, label_pred)
        iteration += 1
    return {'record': record}


if __name__ == '__main__':
    path = '../data/small'
    iteration = 5
    cut = 0.05
    for dataName in os.listdir(path):
        data, label, K = data_processing(path + '/' + dataName)
        for cut_off in range(1, 15):
            for j in range(50):
                task = Task(round(cut * cut_off, 2), j, dataName.split('.')[0], path + '/' + dataName)
                run(task=task)
    # k = 9
    # s = 0
    # while k < 18:
    #     s = pow(2, k)
    #     for i in range(1):
    #         data, label = make_classification(n_samples=s, n_features=10, n_informative=2, n_redundant=0,
    #                                           n_repeated=0,
    #                                           n_classes=3, n_clusters_per_class=1, weights=None, flip_y=0.01,
    #                                           class_sep=1.0,
    #                                           hypercube=True, shift=0.0, scale=100, shuffle=True, random_state=7)
    #         randomData(s, data, iteration=i, parameter=str(s),
    #                    dataName='DRSC-' + str(s), sonIteration=0)
    #     k = k + 1
