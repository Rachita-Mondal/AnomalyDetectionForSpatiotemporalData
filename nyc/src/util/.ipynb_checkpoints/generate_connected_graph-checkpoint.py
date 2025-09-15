import networkx as nx

def generate_connected_graph(size, type, seed=0, maxit=200, **kwargs):
    """Generate a random connected graph 

    Args:
        size (int): number of vertices in the graph. If the graph type is grid, takes in
            2-tuple as argument the size of the grid. Ex: (5,10)
        type (str): Type of the random graph. The choices are 'geometric' for random geometric,
            'grid' for a grid of vertices with cartesian product structure, 'er' erdos renyi, 
            'ba' barabasi-albert random graph.
        param (_type_): _description_
        seed (int): random generation 
        maxit (int, optional): _description_. Defaults to 200.
        radius (float,optional): 

    Raises:
        ValueError: _description_

    Returns:
        G, sd (2-tuple): Returns the graph and the seed
    """
    it = 0
    while True:
        sd = seed+it
        if type == 'geometric':
            G = nx.random_geometric_graph(size, kwargs.get('radius',0.25), seed=sd)
        elif type == 'grid':
            if not isinstance(size,tuple):
                raise TypeError("Specified size must be a tuple for the grid option.")
            G = nx.grid_2d_graph(size[0],size[1])
        elif type == 'er':
            G = nx.erdos_renyi_graph(size, kwargs.get('p', 0.2), seed=sd)
        elif type == 'ba':
            G = nx.barabasi_albert_graph(size, kwargs.get('m', 1), seed=sd)
        if nx.is_connected(G):
            print("Graph is connected.")
            break
        it +=1
    if it == maxit:
        raise ValueError("Couldn't construct a connected graph")
    return G, sd

