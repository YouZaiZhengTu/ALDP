import random

import numpy as np

from Entities.Node import Node


def construction(nodeList: list[Node], originList: list[Node], nnsList: list[int], iteration: int):
    roots = []
    candidates = np.array(range(len(nodeList)))
    while len(candidates) > 0:
        link = np.array([])
        # i 代表随机选取的数据点的数组索引
        i = random.randint(0, len(candidates) - 1)
        n = candidates[i]
        while True:
            link = np.append(link, n)
            nodeList[n].setIteration(iteration)
            # j 代表随机选取的i的数据点的对应最近邻的数组索引
            j = nnsList[n][0]
            if j in link:
                root = createRoot(nodeList[n], nodeList[j], originList, iteration)
                roots.append(root)
                candidates = np.setdiff1d(candidates, link)
                # print("%d,节点：%d与节点：%d互为最近邻，链接终止，当前剩余%d个结点，新节点为%d" % (
                # num, nodeList[n].id, nodeList[j].id, len(list(candidates)), root.id))
                break
            elif j not in candidates:
                nodeList[n].addAdjoinNode(nodeList[j])
                nodeList[j].addAdjoinNode(nodeList[n])
                #print("%d,节点：%d与节点：%d连接，当前剩余%d个节点" % (num, nodeList[n].id, nodeList[j].id, len(list(candidates))))
                candidates = np.setdiff1d(candidates, link)
                break
            nodeList[n].addAdjoinNode(nodeList[j])
            nodeList[j].addAdjoinNode(nodeList[n])
            #print("%d,节点：%d与节点：%d连接，当前剩余%d个节点" % (num, nodeList[n].id, nodeList[j].id, len(list(candidates))))
            n = j
    return roots


def createRoot(point1: Node, point2: Node, originList: list[Node], iteration: int):
    data1 = point1.data
    data2 = point2.data
    newData = []
    for i in range(len(data1)):
        newData.append((float(data1[i]) + float(data2[i])) / 2)
    root = Node(len(originList), newData, str(len(originList)), str(len(originList)))
    originList.append(root)
    #print('1: %d,2: %d' % (point1.id, point2.id))
    point1.removeAdjoinNode(point2.id)
    point2.removeAdjoinNode(point1.id)
    root.addAdjoinNode(point1)
    root.addAdjoinNode(point2)
    point1.addAdjoinNode(root)
    point2.addAdjoinNode(root)
    return root
