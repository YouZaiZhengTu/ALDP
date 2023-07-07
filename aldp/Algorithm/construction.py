import random

import numpy as np

from Entities.Node import Node


def construction(nodeList: list[Node], nns: dict[int], iteration: int):
    roots = {}
    candidates = np.array([key for key in nodeList])
    while len(candidates) > 0:
        link = np.array([])
        # i 代表随机选取的数据点的数组索引
        i = random.randint(0, len(candidates) - 1)
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
