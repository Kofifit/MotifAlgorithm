from collections import defaultdict

class Graph:

    def __init__(self):

        self.graph = defaultdict(list)
        self.fullyConnected = False
        self.nodes = defaultdict(bool)

    def addEdge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.nodes[u] = False
        self.nodes[v] = False

    def DFSUtil(self, v, visited):

        visited.add(v)
        self.nodes[v] = True

        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    def DFS(self, v):

        visited = set()

        self.DFSUtil(v, visited)

        for val in self.nodes.values():
            if val == False:
                return False
        return True
