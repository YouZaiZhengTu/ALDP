import os

from grinch_master.linkage.classic import *

import pandas as pd
from util.estimate import rand_index

from grinch_master.clustering.evaluation import dendrogram_purity
from grinch_master.clustering.grinch import Grinch
from grinch_master.gendataset.synthetic_dataset import generate_synthetic_dataset
from grinch_master.gendataset.shuffle import random_shuffle
from grinch_master.linkage.vector import cosine_similarity_for_binary_vector_data_point

def run(fileName:str,filePath:str,iteration: int):
    origin = pd.read_csv(filePath+'/'+fileName, header=None)
    data = origin.iloc[:, 0:-1]
    label = origin.iloc[:, -1].values.tolist()
    data = (data - data.mean()) / (data.std())
    data_value = data.values
    print(data)

    # the setting of generating synthetic dataset
    n_cluster = 4
    n_point_each_cluster = 25
    n_dim_datapoint = 2500
    # generate synthetic dataset
    # other shuffle functions are provided in gendataset/shuffle.py
    data_stream, ground_truth = generate_synthetic_dataset(n_cluster=n_cluster, n_point_each_cluster=n_point_each_cluster,
                                   n_dim_datapoint=n_dim_datapoint, shuffle=random_shuffle)

    # settings of Grinch algorithm
    single_nn_search = False
    k_nn = 25

    single_elimination = False,

    capping = False
    capping_height = 100

    navigable_small_world_graphs = False
    k_nsw = 50
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
    clustering.dendrogram.print()
    #print("dendrogram purity:", dendrogram_purity(ground_truth, clustering.dendrogram))

    clustering = Grinch(group_average_linkage)

    for d in data:
        # print("inserting", d)
        clustering.insert(d)
        # print dendrogram to debug
        # clustering.dendrogram.print()
    # Grinch algorithm can achieve a perfect dendrogram purity when clustering chain-shaped clusters.
    grich_label_ = clustering.dendrogram
    randIndex = rand_index(label, grich_label_)
    print('文件 %s ,迭代第 %d 次,rand_index %f'%(fileName.split(".")[0],iteration,randIndex))
    return randIndex
if __name__ == "__main__":
    path = 'E:/Project/Python/hierarchical-clustering/data/small'
    fileName = os.listdir(path)
    iteration = 5
    rand_index_ave = 0
    for name in fileName:
        for i in range(iteration):
            rand_index_ave += run(name,path,i)
