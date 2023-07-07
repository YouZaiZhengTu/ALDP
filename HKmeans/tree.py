from itertools import chain


class Tree(object):
    def __init__(self, content):
        self.content = content
        self.points = content.points
        self.left = None
        self.right = None

    def __len__(self):
        return sum(1 for _ in self)

    def __lt__(self, other):
        return self.content < other.content

    def __iter__(self):
        if self.left != None and self.right != None:
            for content in chain(*map(iter, [self.left, self.right])):
                yield content
        else:
            yield self


if __name__ == '__main__':
    a = Tree("a")
    b = Tree("b")
    c = Tree("c")
    d = Tree("d")
    e = Tree("e")
    a.left = b
    a.right = c
    b.left = d
    b.right = e
    for node in a:
        print(node.content)
    print(len(a))
