from Entities.Node import Node


class Tree(object):
    root: Node   # 根节点
    maxDepthNode: int  # 深度最大的节点
    nodeNum: int  # 该树的节点数量
    maxDepth: int  # 该树的最大深度

    def __init__(self, root:Node):
        self.root = root
        self.maxDepthNode = -1
        self.nodeNum = 1
        self.maxDepth = 1

