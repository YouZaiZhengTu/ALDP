import numpy as np
from scipy.stats import pearsonr
import random
from tree import Tree
import sklearn.metrics as metrics


class ClusteringAlgorithm(object):
    def __init__(self, points):
        self.points = points
        self.idmaker = IdMaker()
        self.clusters = None

    def __str__(self):
        string = ""
        count = 1
        for cluster in self.clusters:
            string += "cluster%s:\n" % (count)
            count += 1
            tmp = [str(point) for point in cluster.points]
            tmp.sort()
            for point in tmp:
                string += "\t%s\n" % (point)
        return string

    def get_labels(self):
        points = []
        labels_true = []
        labels_pred = []
        count = 0
        for cluster in self.clusters:
            for point in cluster.points:
                points.append(point)
                label = self.idmaker.get_id(point.label)
                labels_true.append(label)
                labels_pred.append(count)
            count += 1
        return points, labels_true, labels_pred

    def eval_clusters(self):
        """calculates the adjusted rand index of the clustering
        based on the label of the points
        """
        _, labels_true, labels_pred = self.get_labels()
        ari = metrics.adjusted_rand_score(labels_true, labels_pred)
        hom = metrics.homogeneity_score(labels_true, labels_pred)
        comp = metrics.completeness_score(labels_true, labels_pred)
        return ari, hom, comp


class KMeans(ClusteringAlgorithm):
    def __init__(self, points, k):
        super(KMeans, self).__init__(points)
        self.k = k
        self.clusters = []

    def generate_clusters(self):
        # initialise centers
        centers = random.sample(self.points, self.k)
        self.clusters = [Cluster([], centers[i]) for i in range(self.k)]
        self.clusters[0].points = self.points
        # repeat until no change
        moves = [None]
        while moves:
            moves = []
            # add points to clusters
            for cluster in self.clusters:
                for point in cluster.points:
                    dists = [clust.distance(point) for clust in self.clusters]
                    index_min = np.argmin(dists)
                    if cluster is not self.clusters[index_min]:
                        cluster.points.remove(point)
                        self.clusters[index_min].points.append(point)
                        moves.append(point)
            # calculate new centers
            for cluster in self.clusters:
                try:
                    cluster.new_center()
                except IndexError:
                    cluster.center = random.choice(self.points)


class HKMeans(ClusteringAlgorithm):
    def __init__(self, points, k):
        super(HKMeans, self).__init__(points)
        self.k = k
        self.clusters = Tree(Cluster(points))

    def split_node(self, node):
        cluster = node.content
        kmeans = KMeans(cluster.points, k=2)
        kmeans.generate_clusters()
        node.left, node.right = [Tree(cluster) for cluster in kmeans.clusters]

    def node_to_split(self):
        return max(self.clusters)

    def generate_clusters(self):
        Tree(Cluster(self.points))
        while len(self.clusters) < self.k:
            self.split_node(self.node_to_split())


class Cluster(object):
    def __init__(self, points, center=None):
        self.points = points
        self.center = center

    def average_distance(self):
        dists = [self.center.distance(point) for point in self.points]
        return np.mean(dists)

    def distance(self, point):
        return self.center.distance(point)

    def new_center(self):
        if self.points:
            new_data = []
            for arrays in zip(*[point.data for point in self.points]):
                new_array = np.mean(np.dstack(arrays), axis=2)
                new_data.append(*new_array)
            self.center = Dataset(new_data)
        else:
            raise IndexError

    def __lt__(self, other):
        return self.average_distance() < other.average_distance()

    def __gt__(self, other):
        return self.average_distance() > other.average_distance()


class Dataset(object):
    def __init__(self, data, name="", label="", comment=""):
        self.data = np.array([np.array(array) for array in data])
        self.name = name
        self.label = label
        self.comment = comment

    def __str__(self):
        return "{0} - {1}".format(self.label, self.comment)

    def distance(self, dataset):
        return 1 - pearsonr(self.data.flatten(), dataset.data.flatten())[0]


class IdMaker(object):
    """returns a unique id for any object
    """

    def __init__(self):
        self.items = {}
        self.next_id = 0

    def __len__(self):
        return len(self.items)

    def get_next_id(self):
        """returns the next id
        """
        self.next_id += 1
        return self.next_id

    def get_id(self, item):
        """looks if object already assigned an id
        otherwise creates a new id
        """
        index = self.items.get(item)
        if index == None:
            self.items[item] = self.get_next_id()
        return self.items[item]

    def get_items(self):
        """returns the list of all items
        """
        return self.items


def read_input():
    """test function which reads points from test_data.txt(iris3),
    clusters it and outputs the clusters
    """
    points = []
    with open("test_data.txt") as test_file:
        for line in test_file:
            line = line.split()
            values = [[float(x) for x in line[1:4]]]
            point = Dataset(values, label=line[5])
            points.append(point)
    return points


if __name__ == '__main__':
    POINTS = read_input()
    # KMEANS = KMeans(POINTS, 3)
    # KMEANS.generate_clusters()
    # print KMEANS
    HKMEANS = HKMeans(POINTS, 3)
    HKMEANS.generate_clusters()
    print(HKMEANS)
