import numpy as np
from sklearn.neighbors import NearestNeighbors

from Entities.Node import Node


# 将数据对象的数据提取用于kdTree计算
# input: list[Node]
# output: list: data
def extract_data_from_Node(nodeList: dict[int, Node]):
    dataList = []
    for key in nodeList:
        dataList.append(nodeList[key].data)
    return dataList


# 寻找数据的最近邻
# input: list[Node]
# output: list: Node's NN index
def findNNs(nodeList: list[Node], k=2):
    dataList = extract_data_from_Node(nodeList)
    return kdTree(dataList, nodeList, k)


def kdTree(dataList, nodeList: dict[int, Node], k, return_dist=False):
    origin = np.array(dataList)
    neighbors = NearestNeighbors(n_neighbors=3).fit(origin)
    dist = neighbors.kneighbors(origin, return_distance=False).tolist()
    nns = {}
    snns = {}
    i = 0
    pos = [key for key in nodeList]
    for key in nodeList:
        nns[nodeList[key].id] = pos[dist[i][1:][0]]
        snns[nodeList[key].id] = pos[dist[i][2:][0]]
        i += 1
    return nns, snns
