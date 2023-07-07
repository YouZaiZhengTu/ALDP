import copy
import math
from queue import Queue

from Entities.Node import Node


class Tree(object):
    root: Node  # 根节点
    maxDepthNode: int  # 深度最大的节点
    nodeNum: int  # 该树的节点数量
    maxDepth: int  # 该树的最大深度

    def __init__(self, root: Node):
        self.root = root
        self.maxDepthNode = -1
        self.nodeNum = 1
        self.maxDepth = 1


def findMaxDepthNode(treeList: list[Tree], iteration: int):
    for tree in treeList:
        currentStack = []
        for key in tree.root.adjoinList:
            currentStack.append(tree.root.adjoinList[key])
        depth = 0
        num = 0
        isVisit = []
        while currentStack:
            nextStack = []
            tree.maxDepthNode = currentStack
            for node in currentStack:
                if node.id not in isVisit and node.iteration == iteration:
                    isVisit.append(node.id)
                    num += 1
                    for key in node.adjoinList:
                        if node.adjoinList[key].id not in isVisit and node.adjoinList[key].iteration == iteration:
                            nextStack.append(node.adjoinList[key])
            currentStack = nextStack
            depth += 1
        tree.nodeNum = num + 1
        tree.maxDepth = copy.deepcopy(depth)


def BFS_Algorithm(source: Node, iteration: int):
    Q = Queue()
    visited_vertices = list()
    depth = 0
    source.depth = depth
    leaf_size = 1
    node_size = 1
    prun_node = None
    Q.put(source)
    visited_vertices.append(source.id)
    while not Q.empty():
        size = 0
        depth += 1
        for level in range(leaf_size):
            vertex = Q.get()
            # print('第%d层,节点%d' % (vertex.depth, vertex.id))
            for key in vertex.adjoinList:
                if vertex.adjoinList[key].id not in visited_vertices and vertex.adjoinList[key].iteration == iteration:
                    Q.put(vertex.adjoinList[key])
                    vertex.adjoinList[key].depth = depth
                    visited_vertices.append(vertex.adjoinList[key].id)
                    size += 1
                    node_size += 1
            # print("节点%d的邻接节点添加完毕，共添加%d个节点" % (vertex.id, size))
        prun_node = vertex
        leaf_size = size
    return depth - 1, node_size, prun_node


def pruning(roots: list[Node], threshold: float, iteration: int):
    newRoots = []
    for root in roots:
        # draw_tree_graph(root, str(root.id), 'pruning', str(root.id))
        depth, size, prun_node = BFS_Algorithm(root, iteration)
        c = math.ceil(math.log(size, threshold))
        if depth > c + 1:
            # print('%d节点被剪除' % (prun_node.id))
            prun_node.delEdge(iteration)
            newRoots.append(prun_node)
    return newRoots
    # newRoots = []+
    # for tree in treeList:
    #     c = math.ceil(math.log(tree.nodeNum, threshold))
    #     if tree.maxDepth >= c:
    #         for node in tree.maxDepthNode:
    #             node.delEdge()
    #             node.iteration = iteration + 1
    #             newRoots.append(node)
    # print("断开节点：%d和节点：%d,当前新节点%d个"%(node.id,father.id,len(newRoots)))
