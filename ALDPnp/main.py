import os
from time import time

from ExperimentTool.Entity.Running import ExpMonitor
from algorithm import getKNNs, construction, rebulidSCT,findDensityPeak, get_assignment, pretreatment
from ExperimentTool.Entity.Basis import Record, RecordType, Task
from Utils.FilesHandler import print_estimate

@ExpMonitor(expId='ALDPnp', algorithmName='ALDPnp', storgePath=os.pardir + '/Experiment')
def run(task, **kwargs):
    record = Record()
    nodeList, K = pretreatment(task.filePath)
    num = len(nodeList)
    sonIteration = 0
    times = 0
    while len(nodeList) > 3 and len(nodeList) > K:
        start = time()
        nns, snns = getKNNs(nodeList)
        roots = construction(nodeList, nns, sonIteration)
        rebulidSCT(snns, roots, nodeList, sonIteration)
        nodeList = findDensityPeak(roots, iteration=sonIteration)
        end = time()
        label_true, label_pred, tree = get_assignment(nodeList, num)
        record.save_output(RecordType.tree, tree, label_true, sonIteration)
        record.save_output(RecordType.assignment, label_true, label_pred, sonIteration)
        times += end - start
        print_estimate(label_true, label_pred, task.dataName, task.iterIndex
                       , sonIteration, end - start)
        sonIteration += 1
    record.save_time(times)
    return {'record': record}


if __name__ == '__main__':
    path = '../data/small'
    for file in os.listdir(path):
        for testIndex in range(1):
            task = Task(str('NoParam'), testIndex, file.split('.')[0], path + '/' + file)
            run(task=task)
