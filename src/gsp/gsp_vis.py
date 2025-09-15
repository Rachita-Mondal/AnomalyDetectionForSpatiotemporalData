import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

def grid_positions(G, flip_x=False, flip_y=False):
    """Generates a grid layout for the graph G.

    Args:
        G (_type_): _description_

    returns:
        pos (np.ndarray): An array of positions of the nodes in the graph.
    """
    n = len(G)
    pos = np.zeros((n,2))
    t = 0
    for i,j in G.nodes():
        pos[t,:] = np.array([i,j])
        t+=1
    if flip_x:
        pos[:,0] = np.max(pos[:,0])-pos[:,0]
    if flip_y:
        pos[:,1] = np.max(pos[:,1])-pos[:,1]
    return pos

def draw_graph_signal(G, x, save_path=None, **kwargs):
    """Draws the graph signal on the graph structure provided. 

    Args:
        G (nx.classes.graph.Graph): Graph structure to visualize the signal on.
        x (np.ndarray): A single graph signal as a vector of length equal to number
            of graph vertices.
        pos (np.ndarray, optional): Positions of the graph nodes. Defaults to kamada_kawai_layout.
    """
    figsize = kwargs.get('figsize', (10,10))
    pos = kwargs.get('pos', None)
    node_size = kwargs.get('node_size', 400)
    node_color = kwargs.get('node_color', '0.15')
    edge_width = kwargs.get('edge_width', 1.5)
    variation = kwargs.get('variation', None)
    node_width = kwargs.get('node_width', 1)
    node_labels = kwargs.get('node_labels', None)
    edge_labels = kwargs.get('edge_labels', None)
    anomaly_color = kwargs.get('anomaly_color', 'C3')
    colormap = kwargs.get('colormap', 'viridis')
    c = kwargs.get('cmap', 'spring')
    edge_font_size = kwargs.get('edge_font_size', 12)
    node_font_size = kwargs.get('node_font_size', 12)
    suptitle = kwargs.get('suptitle', 'Graph Signal Visualization')
    norm = kwargs.get('norm', cm.colors.Normalize(vmax=np.max(x), vmin=np.min(x)))
    anomaly_labels = kwargs.get('anomaly_labels', np.zeros(x.shape, dtype=bool))
    save_as = kwargs.get('save_as', 'png')
    layout = kwargs.get('layout', 'spring')
    legend_on = kwargs.get('legend_on', True)

    if isinstance(pos,type(None)):
        if layout=='spring':
            pos = nx.kamada_kawai_layout(G)
        elif layout=='kamada_kawai':
            pos = nx.spring_layout(G)
        else:
            raise TypeError('Layout not recognized. Use either spring or kamada_kawai')
    elif isinstance(pos, np.ndarray):
        if pos.shape[1]!=2:
            raise ValueError('The position array provided should be of shape (n,2)')
        else:
            pos = {list(G)[i]: pos[i,:] for i in range(len(G))}
        
    pos_array = np.zeros((len(G),2))
    for i in range(len(G)):
        pos_array[i,:] = pos[list(G)[i]]
    
    if not isinstance(x, np.ndarray):
        raise TypeError('x provided is not a numpy.ndarray')
    if not isinstance(G, nx.classes.graph.Graph):
        raise TypeError('G provided is not an nx.classes.graph.Graph')
    if x.size!= len(G):
        raise ValueError('The dimensions of the signal x and the number of vertices'+
                         ' do not match.')

    cmap = mpl.colormaps[colormap]
    idxs = np.arange(x.size).reshape((x.shape[0],1))
    

    fig, axe = plt.subplots(figsize=figsize)
    fig.tight_layout()
    # Draw the skeleton graph structure
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size,node_color='none',
                           edgecolors=node_color,  linewidths=node_width, ax=axe)
    nx.draw_networkx_edges(G, pos=pos, node_size=node_size,width=edge_width, ax=axe)
    if edge_labels is not None:
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, ax=axe, font_size=edge_font_size)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, ax=axe, font_size=node_font_size)
    if variation is not None:
        nx.draw_networkx_edges(G, pos=pos, node_size=node_size,width=edge_width*2, ax=axe,
                            edge_color=variation, alpha=0.5, edge_cmap=mpl.colormaps['plasma'])
    # Paint the signals and anomalies
    scat_signal = axe.scatter(pos_array[:,0],pos_array[:,1], s=node_size, c=x, cmap=c)
    if legend_on:
        scat_anomaly = axe.scatter(pos_array[idxs[anomaly_labels],0], pos_array[idxs[anomaly_labels],1],
                                s=node_size*0.7, facecolors='none', edgecolors=anomaly_color,
                                lw=node_width*2, label='Anomaly')
    else:
        scat_anomaly = axe.scatter(pos_array[idxs[anomaly_labels],0], pos_array[idxs[anomaly_labels],1],
                                s=node_size*0.7, facecolors='none', edgecolors=anomaly_color,
                                lw=node_width*2)
    # Set the colorbar and name the figure
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=axe, orientation='horizontal',
             label='Signal strength', pad=0.01,fraction=0.05, aspect=80, location='bottom',
            extendrect=False, extend='both')
    if legend_on:
        axe.legend()
    fig.suptitle(suptitle, fontsize=figsize[0]*2)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format=save_as)
    else:
        plt.show()
    return fig, axe



class GraphSignalAnimation:
    def __init__(self, G, X, **kwargs):
        self.G = G
        self.X = X
        self.figsize = kwargs.get('figsize', (10,10))
        self.pos = kwargs.get('pos', None)
        self.node_size = kwargs.get('node_size', 400)
        self.node_color = kwargs.get('node_color', '0.15')
        self.edge_width = kwargs.get('edge_width', 1.5)
        self.node_width = kwargs.get('node_width', 1)
        self.node_labels = kwargs.get('node_labels', None)
        self.anomaly_color = kwargs.get('anomaly_color', 'C3')
        self.colormap = kwargs.get('colormap', 'viridis')
        self.suptitle = kwargs.get('suptitle', 'Sequential Graph Signal Visualization')
        self.norm = kwargs.get('norm', cm.colors.Normalize(vmax=np.max(X), vmin=np.min(X)))
        self.interval = kwargs.get('interval', 200)
        self.anomaly_labels = kwargs.get('anomaly_labels', np.zeros(X.shape, dtype=bool))
        self.layout = kwargs.get('layout', 'spring')

        if isinstance(self.pos,type(None)):
            if self.layout=='spring':
                self.pos = nx.kamada_kawai_layout(G)
            elif self.layout=='kamada_kawai':
                self.pos = nx.spring_layout(G)
            else:
                raise('Layout not recognized. Use either spring or kamada_kawai')
        elif isinstance(self.pos, np.ndarray):
            if self.pos.shape[1]!=2:
                raise ValueError('The position array provided should be of shape (n,2)')
            else:
                self.pos = {list(G)[i]: self.pos[i,:] for i in range(len(G))}
            
        if not isinstance(X, np.ndarray):
            raise TypeError('x provided is not a numpy.ndarray')
        if not isinstance(G, nx.classes.graph.Graph):
            raise TypeError('G provided is not an nx.classes.graph.Graph')
        if X.shape[0]!= len(G):
            raise ValueError('The dimensions of the signal x and the number of vertices'+
                            ' do not match.')

        self.cmap = mpl.colormaps[self.colormap]
        self.idxs = np.arange(len(G)).reshape((len(G),1))
        self.pos_array = np.zeros((len(G),2))
        for i in range(len(G)):
            self.pos_array[i,:] = self.pos[list(G)[i]]
        # fig = plt.figure(figsize=figsize)
        self.fig, self.axe = plt.subplots(figsize=self.figsize)
        # Draw the skeleton graph structure
        nx.draw_networkx_nodes(self.G, pos=self.pos, node_size=self.node_size,node_color='none',
                        edgecolors=self.node_color,  linewidths=self.node_width, ax=self.axe)
        nx.draw_networkx_edges(self.G, pos=self.pos, node_size=self.node_size,width=self.edge_width, ax=self.axe)
        nx.draw_networkx_labels(self.G, pos=self.pos, labels=self.node_labels, ax=self.axe)
        self.fig.suptitle(self.suptitle, fontsize=self.figsize[0]*2)
        self.fig.tight_layout()

        x = X[:,0]
        anomaly_idxs = self.idxs[self.anomaly_labels[:,0].reshape((len(G),1))]
        self.scat_signal = self.axe.scatter(self.pos_array[:,0],self.pos_array[:,1], s=self.node_size, c=x,
                                cmap=self.colormap, norm=self.norm)
        self.scat_anomaly = self.axe.scatter(self.pos_array[anomaly_idxs,0], self.pos_array[anomaly_idxs,1],
                                    s=self.node_size*0.7, facecolors='none', edgecolors=self.anomaly_color,
                                    lw=self.node_width*2, label='Anomaly')

        self.fig.colorbar(mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
                ax=self.axe, orientation='horizontal',
                label='Signal strength', pad=0.01,fraction=0.05, aspect=80, location='bottom',
                extendrect=False, extend='both')
        self.fig.suptitle(self.suptitle, fontsize=self.figsize[0]*2)
        self.fig.tight_layout()

    def __call__(self, i):
        x = self.X[:,i]
        anomaly_idxs = self.idxs[self.anomaly_labels[:,i].reshape((len(self.G),1))]
        self.scat_signal.remove()
        del self.scat_signal
        self.scat_anomaly.remove()
        del self.scat_anomaly
        self.scat_signal = self.axe.scatter(self.pos_array[:,0],self.pos_array[:,1], s=self.node_size, c=x,
                                cmap=self.colormap, norm=self.norm)
        self.scat_anomaly = self.axe.scatter(self.pos_array[anomaly_idxs,0], self.pos_array[anomaly_idxs,1],
                                   s=self.node_size*0.7, facecolors='none', edgecolors=self.anomaly_color,
                                   lw=self.node_width*2, label='Anomaly')
        self.axe.legend()
        return self.scat_signal, self.scat_anomaly

    def animate(self):
        ani = mpl.animation.FuncAnimation(self.fig, self, frames=self.X.shape[1], interval=self.interval)
        html = HTML(ani.to_jshtml())
        return html


    def save_as_gif(self, save_path=None):
        if save_path is None:
            save_path = self.sup_title.replace(' ', '_')+'.gif'
        ani = mpl.animation.FuncAnimation(self.fig, self, frames=self.X.shape[1], interval=self.interval)
        ani.save(save_path, writer='imagemagick')
        return ani
