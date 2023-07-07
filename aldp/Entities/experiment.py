import numpy as np

from aldp.Entities import Node


def get_distribute(roots: dict[int, Node], size: int):
    label_true = [-1 for i in range(size)]
    label_pred = [-1 for i in range(size)]
    i = 0
    for key in roots:
        next_node = [roots[key]]
        visited = {roots[key].id}
        while next_node:
            r = next_node.pop()
            label_pred[r.id] = i
            label_true[r.id] = r.label
            for n in r.adjacentNode:
                if n not in visited:
                    visited.add(n)
                    next_node.append(r.adjacentNode[n])
        i += 1
    return label_true, label_pred