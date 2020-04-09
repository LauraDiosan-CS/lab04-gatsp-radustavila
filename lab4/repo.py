
import numpy as np

class Repo(object):
    def __init__(self):
        self._length = 0
        self._graph = []
        self._berlin = np.zeros((52,52))

        self._source = -1
        self._destination = -1
        self.load_from_file()

    def load_from_file(self):
        f = open("input.txt", "r")
        lines = f.readlines()

        self._length = int(lines[0])
        for i in range(1, self._length + 1):
            self._graph.append([int(j.rstrip()) for j in lines[i].split(',')])

        self._source = int(lines[self._length + 1])
        self._destination = int(lines[self._length + 2])

    def load_from_file_berlin(self):
        f = open("berlin52.txt", "r")
        lines = f.readlines()

        self._length = int(lines[0])

        for i in range(1, self._length + 1):
            self._graph.append([float(i) for i in lines[i].split(' ')])

        for i in range(0, self._graph.__len__()):
            for j in range(i + 1, self._graph.__len__()):
                plot1 = [self._graph[i][1], self._graph[i][2]]
                plot2 = [self._graph[j][1], self._graph[j][2]]
                euclidean_distance = np.sqrt((plot1[0] - plot2[0]) ** 2 + (plot1[1] - plot2[1]) ** 2)
                self._berlin[i][j] = euclidean_distance
                self._berlin[j][i] = euclidean_distance



    def get_berlin(self):
        return self._berlin

    def get_graph(self):
        return self._graph

    def get_length(self):
        return self._length

    def get_source(self):
        return self._source

    def get_destination(self):
        return self._destination
