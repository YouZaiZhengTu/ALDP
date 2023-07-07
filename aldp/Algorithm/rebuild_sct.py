import numpy as np

from Entities.Node import Node


# 统计每个sct树的节点数量
def compute_sct_num(roots: dict[int, Node], iteration: int):
    rebuild_roots = []
    for key in roots:
        sct_node = [roots[key]]
        node_num = 1
        next_node = [roots[key]]
        other_node = 0
        visited = {key}
        while next_node:
            r = next_node.pop()
            for n in r.adjacentNode:
                if r.adjacentNode[n].iteration == iteration and n not in visited:
                    next_node.append(r.adjacentNode[n])
                    sct_node.append(r.adjacentNode[n])
                    visited.add(n)
                    other_node = n
                    node_num += 1
        if node_num == 2:
            rebuild_roots.append((key, other_node))
        for node in sct_node:
            node.set_node_num(node_num)
        # print('%d sct共有 %d 个节点' % (key, node_num))
    return rebuild_roots


def connect_roots(rebuild_roots, roots, snns, nodeList: list[Node]):
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


def rebuild(snns: dict[int], roots: list[Node], nodeList: list[Node], iteration: int):
    rebuild_roots = compute_sct_num(roots, iteration)
    connect_roots(rebuild_roots, roots, snns, nodeList)
