from collections import Callable
from typing import Dict

from grinch_master.dendrogram.node import Leaf, Node
from grinch_master.model.cluster import Cluster
from grinch_master.model.data_point import DataPoint
from .hac import HAC


class OnlineHAC(HAC):
    """
    The implementation of Online algorithm
    """

    def insert(self, data_point: DataPoint):
        leaf = Leaf(data_point)
        sibling = self.nearest_neighbour(leaf)
        self.make_sib(sibling, leaf)
