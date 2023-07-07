from Entities import Node


class Node(object):
    # 基础数据
    id: int  # 节点id
    data: tuple  # 节点数据
    label: int  # 节点标签
    labelName: str  # 节点标签名称
    # 迭代过程
    adjacentNode: dict  # 相邻节点
    degree: int  # 度
    iteration: int  # 迭代序数
    isVisited: bool  # 访问标记
    # 结果
    label_pred: int  # 预测标签

    def __init__(self, id, data, label, labelName):
        # 基础数据
        self.id = id  # 节点id
        self.data = data  # 节点数据
        self.label = label  # 节点标签
        self.labelName = labelName  # 节点标签名称
        # 迭代过程
        self.adjacentNode = {}  # 相邻节点
        self.degree = 0  # 度
        self.iteration = 0  # 迭代序数
        self.isVisited = False  # 访问标记
        self.node_num = 0
        # 结果
        self.label_pred = 0  # 预测标签

    def add_adjacent_node(self, node: Node):
        self.adjacentNode[node.id] = node
        self.degree += 1

    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def set_node_num(self, node_num: int):
        self.node_num = node_num
