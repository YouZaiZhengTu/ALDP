import os
import pathlib
import time

from sklearn.metrics import euclidean_distances
import pathlib as pl
from Affinity.single_affinty import single_affinty, result_to_predict_vec
from ExperimentTool.Entity.Basis import Task, Record, RecordType
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing, print_estimate

@ExpMonitor(expId='Affinity', algorithmName='Affinity', storgePath=os.pardir+'/Experiment')
def run(task,k, **kwargs):
    record = Record()
    data, label, K = data_processing(task.filePath, shuffle=False)
    start = time.time()
    # 构建欧式距离矩阵
    similarity_matrix = euclidean_distances(data)
    # 寻找每个点的最近距离来构建树：输入 similarity matrix,k(要聚类的个数) 输出 result 分类的结果
    result = single_affinty(similarity_matrix, k=k)
    # 将聚类的结果映射为一个向量
    assignment = result_to_predict_vec(result)
    end = time.time()
    # print_estimate(label, assignment, task.dataName, task.iterIndex, 0, end - start)
    record.save_output(RecordType.assignment, label, assignment, 0)
    record.save_time(end-start)
    return {'record': record}


if __name__ == '__main__':
    path = '../data/small'
    for dataName in os.listdir(path):
        data, label, K = data_processing(path + '/' + dataName)
        for k in range(K - 5 if K - 5 > 1 else 1, K + 5):
            for j in range(1):
                task = Task(k, j, dataName.split('.')[0], path + '/' + dataName)
                run(task=task, k=k)
