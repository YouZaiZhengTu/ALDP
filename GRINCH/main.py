import multiprocessing
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd

from ExperimentTool.Entity.Basis import Task, Record, RecordType
from ExperimentTool.Entity.Running import ExpMonitor
from Utils.FilesHandler import data_processing, print_estimate
from GRINCH.clustering.grinch import Grinch
from GRINCH.gendataset.realworld_dataset import data_wrapper
from GRINCH.linkage.vector import cosine_similarity_for_binary_vector_data_point
from GRINCH.model.data_point import BinaryVectorDataPoint
from clustering.evaluation import dendrogram_purity
import os
from typing import Tuple

import pandas as pd

from ExperimentTool.Entity.Basis import Task
from Utils.FilesHandler import data_processing
from GRINCH.clustering.grinch import Grinch
from GRINCH.gendataset.realworld_dataset import data_wrapper
from GRINCH.linkage.vector import cosine_similarity_for_binary_vector_data_point
from clustering.evaluation import dendrogram_purity


@ExpMonitor(expId='GIRNCH', algorithmName='GIRNCH', storgePath=os.pardir + '/Experiment')
def run(task, **kwargs):
    # read dataset from ALOI
    # other shuffle functions are provided in gendataset/shuffle.py
    record = Record()
    data, label_true, K = data_processing(task.filePath)
    # label_true = [0] * len(true)
    # label_id = pd.DataFrame(true).drop_duplicates().values
    # if type(true[0]) == str:
    #     for i, id in enumerate(label_id):
    #         for j, label in enumerate(true):
    #             if label == id:
    #                 label_true[j] = i
    data = data.tolist()
    start = time.time()
    # data_stream, ground_truth = data_wrapper(true,data)
    data_stream = [BinaryVectorDataPoint(d, i, label_true[i]) for i,d in enumerate(data)]
    # settings of Grinch algorithm
    single_nn_search = True
    k_nn = 10

    single_elimination = True,

    capping = Tuple
    capping_height = 100

    navigable_small_world_graphs = False
    k_nsw = 50

    # the purity monitor to monitor the dendrogram purity during inserting data points
    # monitor = DpMonitor(n_data_points=len(data_stream), n_workers=8, ground_truth=ground_truth)
    # grinch approach to cluster
    # other linkage functions are provided in linkage/*.py
    # customized linkage function is also allowed as long as it takes same inputs and return same outputs.
    clustering = Grinch(cosine_similarity_for_binary_vector_data_point, debug=False,
                        single_nn_search=single_nn_search, k_nn=k_nn,
                        single_elimination=single_elimination,
                        capping=capping, capping_height=capping_height,
                        navigable_small_world_graphs=navigable_small_world_graphs, k_nsw=k_nsw)
    # process data stream
    for dp in data_stream:
        clustering.insert(dp)
    # wait for monitor to finish its tasks (because it's multiprocessing)
    # monitor.join()
    # monitor.show_purity_plot()
    # roots = [clustering.dendrogram]
    # while len(roots) < task.params:
    #     root = roots.pop(0)
    #     if root.lchild:
    #         roots.append(root.lchild)
    #     if root.rchild:
    #         roots.append(root.rchild)
    #     if root.lchild is None and root.rchild is None:
    #         roots.append(root)
    # distribution = [-1 for i in range(len(data))]
    # labels = [-1 for i in range(len(data))]
    # for i, root in enumerate(roots):
    #     for n in root.data_points:
    #         distribution[int(n.id)] = i
    #         labels[int(n.id)] = n.nodeId
    # print(-1 in distribution)
    patitions = []
    patitions.append(clustering.dendrogram)
    flag = 0
    while len(patitions) < K:
        bigest_c_size = 0
        bigest_c_index = 0
        for index in range(len(patitions)):
            if len(patitions[index].lvs) > bigest_c_size:
                bigest_c_size = len(patitions[index].data_points)
                bigest_c_index = index
        root = patitions.pop(bigest_c_index)
        if root.rchild == None and root.lchild == None:
            patitions.append(root)
            flag += 1
            if flag > 2:
                print('circle!')
        else:
            flag = 0
            if root.rchild != None:
                patitions.append(root.rchild)
            if root.lchild != None:
                patitions.append(root.lchild)
    grich_label_ = np.zeros(len(data))
    for i in range(len(patitions)):
        for node in patitions[i]:
            grich_label_[int(node.id)] = i
    grich_label_.tolist()
    # clustering.dendrogram.print()
    # grich_label_ = clustering.dendrogram
    # print("dendrogram purity:", dendrogram_purity(benchmark, clustering.dendrogram))
    # r = rand_index(benchmark, grich_label_)
    # RI += rand_index(benchmark, grich_label_)
    # RI += cluster.normalized_mutual_info_score(benchmark, grich_label_)
    # print(rand_index(benchmark, grich_label_))
    #     output.flush()
    #     print(data_name, RI/ts)
    # output.close()
    print_estimate(label_true, grich_label_, task.dataName, int(task.iterIndex), 0, time.time() - start)
    # print("dendrogram purity:", dendrogram_purity(ground_truth, clustering.dendrogram))
    # print('emd')
    record.save_time(time.time() - start)
    record.save_output(RecordType.assignment, label_true, grich_label_)
    return {'record': record}

if __name__ == '__main__':
    path = '../data/small'
    taskList = []
    for file in os.listdir(path):
        data, label, K = data_processing(path + '/' + file)
        # low,height = 0, 0
        # if K >= 7:
        #     low = K-7 if K-7 > 1 else 1
        #     height = K + 7
        # else:
        #     low = 2
        #     height = 13
        # for k in range(low, height):
        for testIndex in range(20):
            taskList.append(Task(K, testIndex, file.split('.')[0], path + '/' + file))
    for task in taskList:
        run(task)
    # pool = multiprocessing.Pool(processes=10)
    # for task in taskList:
    #     pool.apply_async(run, (task, ))
    # pool.close()
    # pool.join()
