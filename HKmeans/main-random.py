import os
import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ExperimentTool.Entity.Basis import Task, Record, RecordType
from ExperimentTool.Entity.Running import ExpMonitor
from FilesHandler import data_processing
from kmeans import HKMeans, Dataset


def pretreatment(filePath, openScaler=True):
    # 文件读取并去重
    d = pd.read_csv(filePath, header=None)
    data = d.iloc[:, 0:-1]
    label = d.iloc[:, -1]
    # 归一化处理
    if openScaler:
        scaler = MinMaxScaler()
        # 归一化之后，进行赋值
        data = scaler.fit_transform(data)
    nodeList = []
    for i in range(data.shape[0]):
        nodeList.append(Dataset(data[i], name=label[i], label=label[i], comment=label[i]))
    return nodeList, label.drop_duplicates().shape[0]


# @ExpMonitor(expId='HKMeans', algorithmName='HKMeans', storgePath='G:/Experiment')
def run(task, k, **kwargs):
    record = Record()
    data, label, K = data_processing(task.filePath)
    nodeList = []
    for i in range(data.shape[0]):
        nodeList.append(Dataset(data[i], name=label[i], label=label[i], comment=label[i]))
    start = time.time()
    hkmeans = HKMeans(nodeList, k)
    hkmeans.generate_clusters()
    end = time.time()
    print('%s：%0.4f' % (task.dataName, end - start))
    points, label_true, label_pred = hkmeans.get_labels()
    record.save_output(RecordType.assignment, label_true, label_pred, 0)
    # print_estimate(label_true, label_pred, task.dataName, task.iterIndex, 0, time.time() - start)
    record.save_time(end - start)
    return {'record': record}


if __name__ == "__main__":
    path = '../../data/random'
    for file in os.listdir(path):
        data, label, k = data_processing(path + '/' + file)
        for testIndex in range(1):
            task = Task(k, testIndex, file.split('.')[0], path + '/' + file)
            run(task=task, k=k)
