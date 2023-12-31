import sys
from typing import List, Union, Callable, Dict

from grinch_master.dendrogram.node import Node, Leaf
from grinch_master.dendrogram.tree import swap
from grinch_master.model.cluster import Cluster
from grinch_master.model.data_point import DataPoint
from .hac import HAC


class RotationHAC(HAC):
    """
    The implementation of Rotation Algorithm
    """

    def insert(self, data_point: DataPoint):
        leaf = Leaf(data_point)
        sibling = self.nearest_neighbour(leaf)
        self.make_sib(sibling, leaf)
        for v in self.dendrogram.descendants:
            while v.sibling is not None and v.aunt is not None and \
                    self.get_similarity(v, v.sibling) < self.get_similarity(v, v.aunt):
                swap(v.sibling, v.aunt)
