class Node(object):
    id: int  # 节点id
    data: tuple  # 节点数据
    label: int  # 节点标签
    labelName: str  # 节点标签名称
    adjoinList: dict  # 该节点的邻接节点
    iteration: int  # 迭代序数
    isVisited: bool  # 访问标记
    label_pred: int  # 预测标签
    depth: int  # sct树深度

    def __init__(self, id, data, label, labelName):
        self.id = id
        self.data = data
        self.label = label
        self.labelName = labelName
        self.adjoinList = {}
        self.iteration = 0
        self.isVisited = False
        self.label_pred = -1
        self.depth = -1

    def addAdjoinNode(self, node):
        self.adjoinList[node.id] = node

    def removeAdjoinNode(self, id):
        if id in self.adjoinList.keys():
            self.adjoinList.pop(id)
            return False
        else:
            return True

    def delEdge(self, iteration):
        for key in self.adjoinList:
            node = self.adjoinList[key]
            if node.iteration == iteration:
                node.removeAdjoinNode(self.id)
                self.removeAdjoinNode(node.id)
            break

    def setIteration(self, iteration):
        self.iteration = iteration

    def setLabel_pred(self, label: int):
        self.label_pred = label

    def __repr__(self):
        if self.id is not None:
            return str(self.id)
        else:
            return str(self)
