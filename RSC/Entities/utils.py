# 使用广度优先遍历获取树的所有遍历结点
from queue import Queue

import numpy as np
import pandas as pd
from Entities.Node import Node


def get_assignment(roots: dict[int, Node], size: int, ):
    label_true = [-1 for i in range(size)]
    label_pred = [-1 for i in range(size)]
    for i in range(len(roots)):
        Q = Queue()
        visited_vertices = list()
        Q.put(roots[i])
        visited_vertices.append(roots[i].id)
        while not Q.empty():
            vertex = Q.get()
            if vertex.id < size:
                label_true[vertex.id] = vertex.label
                label_pred[vertex.id] = i
            for key in vertex.adjoinList:
                if vertex.adjoinList[key].id not in visited_vertices:
                    Q.put(vertex.adjoinList[key])
                    visited_vertices.append(vertex.adjoinList[key].id)
    return label_true, label_pred


# 获取树形分配结果
def get_Tree(roots: dict[int, Node], size: int, originSize: int):
    label_true = np.zeros(size)
    tree = np.full(size, -1)
    for i in range(len(roots)):
        Q = Queue()
        visited_vertices = list()
        Q.put(roots[i])
        visited_vertices.append(roots[i].id)
        while not Q.empty():
            vertex = Q.get()
            if vertex.id < size:
                label_true[vertex.id] = vertex.label
            for key in vertex.adjoinList:
                if vertex.adjoinList[key].id not in visited_vertices:
                    Q.put(vertex.adjoinList[key])
                    tree[vertex.adjoinList[key].id] = vertex.id
                    visited_vertices.append(vertex.adjoinList[key].id)
    a1 = np.array(label_true)[:originSize]
    if tree.shape[0] > originSize:
        a2 = np.full(tree.shape[0] - originSize, 'newRoot')
    tree = pd.DataFrame(dict(parent=tree, real=np.concatenate((a1, a2), axis=0)))
    return tree
