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

# α = 1;
# β = 5;
# ⍴ = 0.5;
# Q = 100;
# Iterações: 2500
    
    def isInitialized(self):
        return self.iteration >= 0

    def initialize(self, nodes):
        self.iteration = 0
        self.nodes = nodes
        self.numNodes = len(nodes)
        self.adjacencymatrix = np.zeros(2*[len(nodes)])
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    self.adjacencymatrix[i, j] = {
                        'pheromone': self.initial_pheromone,
                        'attractivity': 1/nodes[i].distanceTo(nodes[j])
                    }
        self.bestSolution = None, np.inf

    def nodeProbabilities(self, current_node, possible_nodes):
        node_factors = []
        for i in possible_nodes:
            pheromone = self.adjacencymatrix[current_node, i].get('pheromone')
            attractivity = self.adjacencymatrix[current_node, i].get('attractivity')
            node_factors.append((pheromone**self.alpha)*(attractivity**self.beta))
        node_factors = np.array(node_factors)
        return node_factors/node_factors.sum()

    def chooseNode(self, current_path):
        current_node = current_path[-1]
        possible_nodes = [i for i in range(self.numNodes) if i not in current_path]
        probabilities = self.nodeProbabilities(current_node, possible_nodes)
        return np.random.choice(possible_nodes, p=probabilities)

    def step(self):
        assert self.isInitialized()
        paths = [[i] for i in range(self.numNodes)]


class Node:
    def distanceTo(self, node: Node) -> float:
        raise NotImplementedError()

class Point(Node):
    def __init__(self, position, name=None):
        self.name = name
        self.position = np.array(position)

    def distanceTo(self, node: Node) -> float:
        return np.linalg.norm(self.position - node.position)

def pathLength(nodes: list, cycle: bool) -> float:
    length = 0
    for i in range(len(nodes) - 1):
        length += nodes[i].distanceTo(nodes[i+1])
    if cycle:
        length += nodes[-1].distanceTo(nodes[0])
    return length