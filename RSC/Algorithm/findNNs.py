import numpy as np
from sklearn.neighbors import NearestNeighbors

from Entities.Node import Node


# 寻找数据的最近邻
# input: list[Node]
# output: list: Node's NN index
def findNNs(nodeList: list[Node]):
    dataList = extract_data_from_Node(nodeList)
    return kdTree(dataList)


def kdTree(dataList, k=2, return_dist=False):
    origin = np.array(dataList)
    neighbors = NearestNeighbors(n_neighbors=2).fit(origin)
    dist = neighbors.kneighbors(origin, return_distance=False)
    return [obj[1:] for obj in dist.tolist()]


# 将数据对象的数据提取用于kdTree计算
# input: list[Node]
# output: list: data
def extract_data_from_Node(nodeList: list[Node]):
    dataList = []
    for node in nodeList:
        dataList.append(node.data)
    return dataList
