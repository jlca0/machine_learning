from queue import Queue, LifoQueue, PriorityQueue

class Graph:

    def __init__(
            self,
            nodes: list,
            edges:list
    ) -> None:
        """Initializes a Graph object.
        
        Attributes:
            nodes: (list) The nodes of the graph.
            edges: (list) The edges of the graph, a list of tuples.
            
        """
        self.nodes = nodes
        self.edges = edges
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)
        self.adjacency_list = {node: [] for node in nodes}
        for edge in edges:
            self.adjacency_list[edge[0]].append(edge[1])
            self.adjacency_list[edge[1]].append(edge[0])
        
    def is_adjacent(
            self, 
            x: float, 
            y: float
    ) -> bool:
        """Checks whether there exists an edge of the form (x,y) or (y,x).
        
        Parameters:
            x: (float) A node.
            y: (float) Another node.
        
        Returns:
            (bool) True if x and y are adjacent.
        """
        return y in self.adjacency_list[x]
        
    def _degree(
            self, 
            x: float
    ) -> int:
        """Given a node of the graph it counts edges of the form (x,y) or (y,x).

        Parameters:
            x: (float) A node.
        
            
        Returns:
            (int) Number of nodes adjacent to x.
        """
        return len(self.adjacency_list[x])
    
    def _get_path(
            self,
            node: float,
            parent: dict
    ) -> list:
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        return path[::-1]
    
    def bfs(
            self, 
            start: float, 
            end: float
    ) -> list:
        visited = set()
        queue = Queue()
        queue.put(start)
        parent = {start: None}
        while not queue.empty():
            node = queue.get()
            if node not in visited:
                visited.add(node)
                if node == end:
                    return self._get_path(node, parent)
                for neighbor in self.adjacency_list[node]:
                    if neighbor not in visited:
                        queue.put(neighbor)
                        parent[neighbor] = parent.get(neighbor, node)
        return []
    
    def dfs(
            self, 
            start: float, 
            end: float
    ) -> list:
        visited = set()
        queue = LifoQueue()
        queue.put(start)
        parent = {start: None}
        while not queue.empty():
            node = queue.get()
            if node not in visited:
                visited.add(node)
                if node == end:
                    return self._get_path(node, parent)
                for neighbor in self.adjacency_list[node]:
                    if neighbor not in visited:
                        queue.put(neighbor)
                        parent[neighbor] = node
        return []
    
class DirectedGraph(Graph):

    def __init__(
            self,
            nodes: list,
            edges:list
    ) -> None:
        """Initializes a Graph object.
        
        Attributes:
            nodes: (list) The nodes of the graph.
            edges: (list) The edges of the graph, a list of tuples.
            
        """
        self.nodes = nodes
        self.edges = edges
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)
        self.adjacency_list = {node: [] for node in nodes}
        for edge in edges:
            self.adjacency_list[edge[0]].append(edge[1])

    def is_adjacent(
            self, 
            x: float, 
            y: float
        ) -> bool:
        """Checks whether there exists an edge of the form (x,y).
        
        Parameters:
            x: (float) A node.
            y: (float) Another node.
        
        Returns:
            (bool) True if x and y are adjacent.
        """
        return y in self.adjacency_list[x]
    
    def _out_degree(
            self, 
            x: float
    ) -> int:
        """Given a node of the graph it counts edges of the form (x,y).

        Parameters:
            x: (float) A node.
        
            
        Returns:
            (int) Number of nodes adjacent to x.
        """
        return len(self.adjacency_list[x])
    
    def _in_degree(
            self, 
            x: float
    ) -> int:
        """Given a node of the graph it counts edges of the form (y,x).

        Parameters:
            x: (float) A node.
        
            
        Returns:
            (int) Number of nodes which x is adjacent to.
        """
        return sum(1 for edge in self.edges if x == edge[1])


class WeightedGraph(Graph):
    """Initializes a WeightedGraph object.
    
    Attributes:
        nodes: (list) The nodes of the graph.
        edges: (list) The edges and weights of the graph, a list of tuples.
        
    """
    def __init__(
            self, 
            nodes: list, 
            edges: list
    ) -> None:
        assert all(len(edge) == 3 for edge in edges), "Edges must be of the form (x, y, weight)"
        self.nodes = nodes
        self.edges = edges
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)
        self.adjacency_list = {node: [] for node in nodes}
        for edge in edges:
            self.adjacency_list[edge[0]].append((edge[1], edge[2]))
            self.adjacency_list[edge[1]].append((edge[0], edge[2]))


    def weight(
            self, 
            x: float, 
            y: float
        ) -> float:
        """Given two nodes x, y it returns the weight of their edge (x, y, weight).
        
        Parameters:
            x: (float) A node.
            y: (float) Another node.

        Returns:
            (float) The weight of the edge (x,y).
            (None) If the edge (x,y) does not exist.
        """
        for edge in self.edges:
            if edge[0] == x and edge[1] == y:
                return edge[2]
        return None

class DirectedWeightedGraph(DirectedGraph, WeightedGraph):
    """Initializes a WeightedGraph object.
    
    Attributes:
        nodes: (list) The nodes of the graph.
        edges: (list) The edges and weights of the graph, a list of tuples.
        
    """
    def __init__(
            self, 
            nodes: list, 
            edges: list
    ) -> None:
        assert all(len(edge) == 3 for edge in edges), "Edges must be of the form (x, y, weight)"
        self.nodes = nodes
        self.edges = edges
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)
        self.adjacency_list = {node: [] for node in nodes}
        for edge in edges:
            self.adjacency_list[edge[0]].append((edge[1], edge[2]))