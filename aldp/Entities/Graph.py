from Entities import Node


class Graph(object):
    nodeList: list[Node]
    node_size: int

    def __init__(self):
        self.nodeList = []
        self.node_size = len(self.nodeList)

    def add_Node(self, node: Node):
        self.nodeList.append(node)
        self.node_size = len(self.nodeList)
