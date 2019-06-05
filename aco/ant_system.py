import numpy as np

class AntSystem:
    def __init__(self, alpha, beta, evaporation_rate, Q, initial_pheromone=1):
        self.nodes = None
        self.adjacencymatrix = None
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.initial_pheromone = initial_pheromone
        self.Q = Q
        self.iteration = -1

    def isInitialized(self):
        return self.iteration >= 0

    def initialize(self, nodes):
        self.iteration = 0
        self.nodes = nodes
        self.numNodes = len(nodes)
        self.adjacencymatrix = np.zeros(2*[len(nodes)]+[2])
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i > j:
                    # 0 = pheromone
                    # 1 = distance 
                    self.adjacencymatrix[i, j] = [self.initial_pheromone, nodes[i].distanceTo(nodes[j])]
        self.bestSolution = None, np.inf

    def nodeProbabilities(self, current_node, possible_nodes):
        node_factors = []
        for i in possible_nodes:
            pheromone, distance = self.adjacencymatrix[(current_node, i) if current_node > i else (i, current_node)]
            attractivity = 1/(distance + 1E-5)
            node_factors.append((pheromone**self.alpha)*(attractivity**self.beta))
        node_factors = np.array(node_factors) + 1E-5
        return node_factors/node_factors.sum()

    def chooseNode(self, current_path):
        current_node = current_path[-1]
        possible_nodes = [i for i in range(self.numNodes) if i not in current_path]
        probabilities = self.nodeProbabilities(current_node, possible_nodes)
        return np.random.choice(possible_nodes, p=probabilities)

    def pathLength(self, path):
        length = 0
        for i in range(len(path)):
            current_node = path[i]
            next_node = path[(i+1)%len(path)]
            index = (current_node, next_node) if current_node > next_node else (next_node, current_node)
            length += self.adjacencymatrix[index][1]
        return length

    def step(self):
        assert self.isInitialized()
        paths = [[i] for i in range(self.numNodes)]
        for path in paths:
            for _ in range(self.numNodes-1):
                path.append(self.chooseNode(path))
        
        lengths = [self.pathLength(path) for path in paths]
        best_path_index = np.argmin(lengths)
        if lengths[best_path_index] < self.bestSolution[1]:
            self.bestSolution = paths[best_path_index], lengths[best_path_index]
        
        delta_pheromone = np.zeros(2*[self.numNodes])
        for path, length in zip(paths, lengths):
            for i in range(len(path)):
                node1, node2 = path[i], path[(i+1)%len(path)]
                index = (node1, node2) if node1 > node2 else (node2, node1)
                delta_pheromone[index] += self.Q/length
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                index = (i, j) if i > j else (j, i)
                pheromone = self.adjacencymatrix[index][0]
                self.adjacencymatrix[index][0] = (1-self.evaporation_rate)*pheromone + delta_pheromone[index]
        

class Node:
    def distanceTo(self, node: 'Node') -> float:
        raise NotImplementedError()

class Point(Node):
    def __init__(self, position, name=None):
        self.name = name
        self.position = np.array(position)

    def distanceTo(self, node: Node) -> float:
        return np.linalg.norm(self.position - node.position)

# def pathLength(nodes: list, cycle: bool) -> float:
#     length = 0
#     for i in range(len(nodes) - 1):
#         length += nodes[i].distanceTo(nodes[i+1])
#     if cycle:
#         length += nodes[-1].distanceTo(nodes[0])
#     return length