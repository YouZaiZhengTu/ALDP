import json
from typing import Union

import treelib

from .node import Node


class Tree(Node):
    """
    This class represents the dendrogram tree, (a head node)
    a tree can be represented by a head node which only have one right child which is the root of the dendrogram tree.
    """

    def __init__(self):
        super().__init__()

    @property
    def root(self) -> Node:
        return self.rchild

    @root.setter
    def root(self, _root):
        self.rchild = _root

    @property
    def lchild(self):
        return self._lc

    @lchild.setter
    def lchild(self, child):
        raise Exception("head node do not have left child")

    def print(self):
        """
        use treelib package to print dendrogram in a human-readable way
        :return:
        """
        tree = treelib.Tree()
        if self.root is not None:
            print("number of leaves:", len(self.root.lvs))

        def traverse_tree(root: Node, parent: Union[Node, None]):
            tree.create_node("node", root, parent=parent, data=root)
            if root.lchild is not None:
                traverse_tree(root.lchild, root)
            if root.rchild is not None:
                traverse_tree(root.rchild, root)

        traverse_tree(self.root, None)
        path = tree.paths_to_leaves()
        tree.show(data_property="string")

    def to_json(self) -> str:
        return json.dumps(self.root.to_dict())


def lca(n1: Union[Node, None], n2: Union[Node, None]) -> Union[Node, None]:
    """
    find the lowest common ancestors of n1 and n2
    """
    if n1 is None or n2 is None:
        return None
    n1_chain = n1.ancestors
    n1_chain.append(n1)
    tmp = n2
    while tmp not in n1_chain:
        tmp = tmp.parent
        if tmp is None:
            return None
    else:
        return tmp


def swap(s: Node, a: Node):
    """
    swap two node (together with the subtree rooted at these two nodes) in the dendrogram tree
    """
    s_par = s.parent
    a_par = a.parent
    if s_par.lchild == s:
        if a_par.lchild == a:
            s_par.lchild = a
            a_par.lchild = s
        elif a_par.rchild == a:
            s_par.lchild = a
            a_par.rchild = s
    elif s_par.rchild == s:
        if a_par.lchild == a:
            s_par.rchild = a
            a_par.lchild = s
        elif a_par.rchild == a:
            s_par.rchild = a
            a_par.rchild = s
