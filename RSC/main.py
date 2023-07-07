import copy
import os
import time

from ExperimentTool.Entity.Basis import Task, Record, RecordType
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing
from Algorithm.construction import construction
from Algorithm.findNNs import findNNs
from Algorithm.pruning import pruning
from RSC.Entities.utils import get_assignment
from RSC.Entities.objectTransform import data_to_node


@ExpMonitor(expId='RSC', algorithmName='RSC', storgePath=os.pardir + '/Experiment')
def run(task: Task):
    # 数据提取和处理
    record = Record()
    data, label,K = data_processing(task.filePath)
    start = time.time()
    originData = data_to_node(data, label)
    originList = originData['originList']
    nodeList = copy.deepcopy(originList)
    iterIndex = 0
    # 循环迭代
    while len(nodeList) > len(list(set(label))):
        # 运行rsc算法
        roots = construction(nodeList=nodeList, originList=originList,
                             nnsList=findNNs(nodeList), iteration=iterIndex)
        roots.extend(pruning(roots=roots, threshold=1.5, iteration=iterIndex))
        label_true, label_pred = get_assignment(roots, len(label))
        end = time.time()
        # print_estimate(label_true, label_pred, task.dataName, int(task.iterIndex), iterIndex, end - start)
        record.save_output(RecordType.assignment, label_true, label_pred, iter=iterIndex)
        nodeList = roots
        iterIndex += 1
    end = time.time()
    record.save_time(end - start)
    return {'record': record}


if __name__ == '__main__':
    path = '../data/small/'
    for file in os.listdir(path):
        for testIndex in range(50):
            task = Task('1.5', testIndex, file.split('.')[0], path + '/' + file)
            run(task=task)
