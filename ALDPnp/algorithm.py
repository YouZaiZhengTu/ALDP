import math
from random import randint

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler



class Node(object):
    def __init__(self, id, data, label):
        # 基础数据
        self.id = id  # 节点id
        self.data = data  # 节点数据
        self.label = label  # 节点标签
        # 迭代过程
        self.adjacentNode = {}  # 相邻节点
        self.degree = 0  # 度
        self.iteration = 0  # 迭代序数
        self.isVisited = False  # 访问标记
        self.node_num = 0
        # 结果
        self.label_pred = 0  # 预测标签

    def add_adjacent_node(self, node):
        self.adjacentNode[node.id] = node
        self.degree += 1

    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def set_node_num(self, node_num: int):
        self.node_num = node_num

    def __repr__(self):
        return str(self.id)


# 获取结点的前k个领居
def getKNNs(nodeList: list[Node], neighbors=3):
    data = np.array([nodeList[key].data for key in nodeList])
    neighbors = NearestNeighbors(n_neighbors=neighbors).fit(data)
    dist = neighbors.kneighbors(data, return_distance=False).tolist()
    nns, snns, i = {}, {}, 0
    pos = [key for key in nodeList]
    for key in nodeList:
        nns[nodeList[key].id] = pos[dist[i][1:][0]]
        snns[nodeList[key].id] = pos[dist[i][2:][0]]
        i += 1
    return nns, snns


# 构造SCT
def construction(nodeList: list[Node], nns: dict[int], iteration: int):
    roots = {}
    candidates = np.array([key for key in nodeList])
    while len(candidates) > 0:
        link = np.array([])
        # i 代表随机选取的数据点的数组索引
        i = randint(0, len(candidates) - 1)
        n = candidates[i]
        while True:
            link = np.append(link, n)
            nodeList[n].set_iteration(iteration)
            # j 代表随机选取的i的数据点的对应最近邻的数组索引
            j = nns[n]
            if j in link:
                roots[nodeList[n].id] = nodeList[n]
                candidates = np.setdiff1d(candidates, link)
                # print('link: %s ,root: %d' % (output + '->' + str(j), n))
                break
            elif j not in candidates:
                nodeList[n].add_adjacent_node(nodeList[j])
                nodeList[j].add_adjacent_node(nodeList[n])
                candidates = np.setdiff1d(candidates, link)
                # print(output + '->' + str(j))
                break
            nodeList[n].add_adjacent_node(nodeList[j])
            nodeList[j].add_adjacent_node(nodeList[n])
            n = j
    return roots


# 统计每个sct树的节点数量
def sct_travel(roots: dict[int, Node], iteration: int):
    rebuild_roots, SCTs = [], {}
    for key in roots:
        sct_node, sct_data, next_node = [roots[key]], [roots[key].data], [roots[key]]
        node_num, other_node, visited = 1, 0, {key}
        while next_node:
            r = next_node.pop()
            for n in r.adjacentNode:
                if r.adjacentNode[n].iteration == iteration and n not in visited:
                    next_node.append(r.adjacentNode[n])
                    sct_node.append(r.adjacentNode[n])
                    sct_data.append(r.adjacentNode[n].data)
                    visited.add(n)
                    other_node = n
                    node_num += 1
        SCTs[key] = dict(data=sct_data, node=sct_node)
        if node_num == 2:
            rebuild_roots.append((key, other_node))
        for node in sct_node:
            node.set_node_num(node_num)
    return rebuild_roots, SCTs


# 连接MSCT到其他SCT
def rebulidSCT(snns, roots, nodeList: list[Node], iteration):
    rebuild_roots, SCTs = sct_travel(roots, iteration)
    candidates = np.array(rebuild_roots).reshape(-1)
    for root in rebuild_roots:
        roots.pop(root[0])
        left_connect_node = nodeList[snns[root[0]]].node_num
        right_connect_node = nodeList[snns[root[1]]].node_num
        if left_connect_node <= right_connect_node:
            nodeList[snns[root[0]]].add_adjacent_node(nodeList[root[0]])
            nodeList[root[0]].add_adjacent_node(nodeList[snns[root[0]]])
            if snns[root[0]] in candidates:
                roots[snns[root[0]]] = nodeList[snns[root[0]]]
            # print('%d 0 link 0 %d' % (root[0], snns[root[0]]))
        else:
            nodeList[snns[root[1]]].add_adjacent_node(nodeList[root[1]])
            nodeList[root[1]].add_adjacent_node(nodeList[snns[root[1]]])
            if snns[root[1]] in candidates:
                roots[snns[root[1]]] = nodeList[snns[root[1]]]
            # print('%d 1 link 1 %d' % (root[1], snns[root[1]]))


# 提取距离矩阵中左下角的值并排序
def format_distance(distance):
    d = []
    for i in range(len(distance)):
        for j in range(i + 1, len(distance[i])):
            d.append(dict(x=i, y=j, dist=distance[i][j]))
    d.sort(key=lambda d: d['dist'])
    return d


# 根据长度生成对应搜索位置
def generatePos(length: int) -> list[int]:
    position = []
    mid = math.ceil(length / 2)
    position.append(mid)
    while mid > 1:
        diff = math.ceil(mid / 2)
        position.append(position[-1] + diff)
        position.insert(0, position[0] - diff)
        mid = diff
    position.pop(0)
    return position


# 如果存在相同的局部密度节点，挑选度大的
def selectDiffRoot(nodes: list):
    if nodes[0]['ld'] != nodes[1]['ld']:
        return nodes[0]['node']
    elif nodes[0]['node'].degree >= nodes[1]['node'].degree:
        return nodes[0]['node']
    else:
        return nodes[1]['node']


# 计算点对之间的局部密度
def compute_local_density(sct: list[Node], pairs_distance):
    nodes = [dict(node=node, ld=0) for node in sct]
    for pos in generatePos(len(pairs_distance)):
        cut_off_distance = pairs_distance[:pos]
        for nodeInCut in cut_off_distance:
            nodes[nodeInCut['x']]['ld'] += 1
            nodes[nodeInCut['y']]['ld'] += 1
        nodes.sort(key=lambda e: e['ld'], reverse=True)
    return selectDiffRoot(nodes)


# 为每个SCT寻找它的密度最大的互惠最近邻
def findDensityPeak(roots: dict[Node], iteration: int):
    rebuild_roots, SCTs = sct_travel(roots, iteration)
    rootList = {}
    for key in SCTs:
        sct = SCTs[key]['node']
        sctData = SCTs[key]['data']
        distances = pairwise_distances(sctData, metric="euclidean")
        pairs_distance = format_distance(distances)
        root = compute_local_density(sct, pairs_distance)
        rootList[root.id] = root
    return rootList


def get_assignment(roots: dict[int, Node], size: int, ):
    label_true = [-1 for i in range(size)]
    label_pred = [-1 for i in range(size)]
    tree = np.full(size, -1)
    i = 0
    for key in roots:
        next_node = [roots[key]]
        visited = {roots[key].id}
        while next_node:
            r = next_node.pop()
            label_pred[r.id] = i
            label_true[r.id] = r.label
            for n in r.adjacentNode:
                if n not in visited:
                    visited.add(n)
                    tree[n] = r.id
                    next_node.append(r.adjacentNode[n])
        i += 1
    return label_true, label_pred, tree

def pretreatment(filePath):
    data = pd.read_csv(filePath, header=None)
    true_k = data.iloc[:, -1].drop_duplicates().shape[0]
    data = data.drop_duplicates().reset_index(drop=True)
    data = data.sample(frac=1).reset_index(drop=True)

    data = np.array(data)
    nonzero_count = np.count_nonzero(data, axis=0)
    data = data[:, nonzero_count > 0.9 * data.shape[1]]
    data[:, :-1] = MinMaxScaler().fit_transform(data[:, :-1])
    nodeList = {}
    for i in range(data.shape[0]):
        nodeList[i] = Node(i, data[i,:-1], data[i,-1])
    return nodeList, true_k
