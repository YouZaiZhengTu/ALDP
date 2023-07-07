import os
from time import time

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

from ExperimentTool.Entity.Basis import Record, RecordType, Task
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing


def readFile(filePath):
    dataset = pd.read_csv(filePath, header=None).drop_duplicates().reset_index(drop=True)
    data = dataset.iloc[:, :-1]
    label = dataset.iloc[:, -1]
    scaler = MinMaxScaler()
    # 归一化之后，进行赋值
    data = scaler.fit_transform(data).tolist()
    return data, label


def construct_tree(clustering, label):
    tree = np.full(clustering.children_.shape[0] + clustering.n_leaves_, -1)
    leaves = dict(enumerate(clustering.children_, clustering.n_leaves_))
    for key in leaves:
        tree[leaves[key][0]] = key
        tree[leaves[key][1]] = key
    a1 = np.array(label)
    if tree.shape[0] > len(label):
        a2 = np.full(tree.shape[0] - len(label), -1)
    return tree, np.concatenate((a1, a2), axis=0)


@ExpMonitor(expId='HAC', algorithmName='HAC', storgePath=os.pardir+'/Experiment')
def run(task: Task,k, **kwargs):
    record = Record()
    data, label, K = data_processing(task.filePath)
    start = time()
    clustering = AgglomerativeClustering(n_clusters=k, linkage='average').fit(data)
    end = time()
    record.save_time(end - start)
    label_pred, label_true = construct_tree(clustering, label)
    record.save_output(RecordType.tree, label_true, label_pred)
    record.save_output(RecordType.assignment, label, clustering.labels_)
    return {'record': record}


if __name__ == '__main__':
    path = '../data/small'
    for dataName in os.listdir(path):
        data, label, K = data_processing(path + '/' + dataName)
        for k in range(K - 5 if K - 5 > 1 else 1, K + 5):
            for j in range(50):
                task = Task(k, j, dataName.split('.')[0], path + '/' + dataName)
                run(task=task, k=k)
