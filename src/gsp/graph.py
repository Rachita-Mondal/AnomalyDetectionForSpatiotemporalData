"""'graph.py'.

========
Is a module is intended as a wrapper module for Graph Signal Processing tools with
the emphasis on Product Graphs, and Graph Signal Generation. Along with helper utilities
it's three main classes are Graph, ProductGraph and GraphProcess
Classes:
    Graph: is mainly used for random Graph generation. Acts as a wrapper to networkx graphs
    ProductGraph: is mainly used for computing product graphs of other graphs and initializing
    random graphs.
    GraphProcess: Represents a random process defined on a graph. It is used to generate random
  graph signals. 
"""
import numpy as np
import networkx as nx
from scipy import linalg
from src.multilinear_ops.m2t import m2t
from src.multilinear_ops.t2m import t2m
from numpy.linalg import norm
import numpy.linalg as LA
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as auprc
import copy, os


class Graph(object):
    """Wrapper class for networkx Graph with custom functionalities.
    
    Initializes a graph object by specifying one of the following 5,
            1. Parameters of a random graph
            2. A graph laplacian matrix
            3. Vectorized upper triangular part of a graph laplacian
            4. A graph adjacency matrix
            5. A networkx Graph object
    Args:
        rand_graph_params (dictionary, optional): Specifies the parameters to initialize a random graph. 
            Defaults to None. Must have the following keywords,
                'dim' (int),  The number of nodes in the graph,  
                'type' (str), Type of the random graph, must be of the following three
                'er'/'ba'/'rbf' for erdos-renyi/barabasi-albert/gaussian-rbf
                'g_params' (float/int/tuple) p/m/(std,thr)
        L (np.array, optional): Initialize with laplacian. Defaults to None.
        l (np.array, optional): Initialize with the vectorized upper part of the the laplacian. Defaults to None.
        A (np.array, optional): Initialize with an adjecancy matrix
        G (nx.Graph,optional): Initialize with a networkx graph
        connected(boolean, optional): Ensures the random graph to be connected.
        max_iter(integer,optional): If the graph is required to be connected, 
            max_iter is the maximum number of trials to construct a random connected graph with given parameters.
    Raises:
        ValueError: _description_
    Returns:
        Graph object.
    """

    def __init__(self, rand_graph_params=None, L=None, l=None, A=None, G=None, connected=False, max_iter=100):
        self.graph_type = None
        self.graph_params = None
        if rand_graph_params!=None:
            rng = np.random.default_rng()
            P = rand_graph_params
            n = P['dim']
            self.graph_type=P['type']
            self.graph_params=P['g_params']
            seed = P.get('seed',int(rng.uniform(0,9999)))
            iter=0
            if P['type']=='er':
                msg = "Cannot create a connected graph. Please increase edge probability."
            elif P['type']=='ba':
                msg = "Cannot create a connected graph. Please increase number of edges to be attached."
            elif P['type']=='rbf':
                    msg = "Cannot create a connected graph. Please decrease threshold or increase standard deviation."
            while True:
                if P['type']=='er':
                    G = nx.erdos_renyi_graph(n, P['g_params'],seed)
                    seed +=1
                elif P['type']=='ba':
                    G = nx.barabasi_albert_graph(n, P['g_params'],seed)
                    seed +=1
                elif P['type']=='rbf':
                    std,threshold = P['g_params']
                    while True:
                        points = rng.uniform(-1, 1, size=(n, 2))            
                        A = rbf_kernel(points, gamma=1/(std**2))
                        A[A<threshold] = 0
                        A[np.diag_indices_from(A)] = 0
                        G = nx.from_numpy_array(A)
                        if nx.is_connected(G): # break if the graph is not required to be connected
                            self.G=G
                            break
                        elif iter==max_iter:
                            raise ValueError(msg)
                        iter +=1
                    self.G = G
                    self.A = nx.to_numpy_array(self.G)
                    self.L = self.A_to_L(self.A).copy()
        elif type(G)!=type(None):
            self.G = copy.deepcopy(G)
            self.A = nx.to_numpy_array(G)
            self.L = self.A_to_L(self.A)
        elif type(L)==type(np.zeros(1)):
            self.L = L.copy()
            self.A = self.L_to_A(L).copy()
            self.G = nx.from_numpy_array(self.A)
        elif type(l)==type(np.zeros(1)):
            self.L = self.l_to_L(l).copy()
            self.A = self.l_to_A(l).copy()
            self.G = nx.from_numpy_array(self.A)
        elif type(A)==type(np.zeros(1)):
            self.L = self.A_to_L(A).copy()
            self.G = nx.from_numpy_array(A)
            self.A = A.copy()
        else:
            raise ValueError("No initialization parameter was given to Graph object")
        self.E = self.edges_in_L(self.L).copy()
        self.lda, self.V = linalg.eigh(self.L) 
        self.n = self.L.shape[0]


    # def update(self, G=None, L=None, l=None, A=None):
    #     self.__init__(G=G, L=L,l=l, A=A)

    def l_to_L(self, l):
        """Convert the vectorized upper triangular part of a laplacian matrix into laplacian matrix.
        
        Args:
            l (np.array): vectorized upper triangular part of the laplacian.

        Returns:
            L (np.array): Square Laplacian matrix
        """
        n = int(np.floor(np.sqrt(2*len(l))))+1
        L= np.zeros((n,n))
        L[np.triu_indices(n,1)]=l.ravel()
        L= L+L.T
        for i in range(n):
            L[i,i] = -np.sum(L[i,...])
        return L
    
    def l_to_A(self,l):
        """Convert the vectorized upper triangular part of a laplacian matrix into adjacency matrix.
        
        Args:
            l (np.array): vectorized upper triangular part of the laplacian.

        Returns:
            A (np.array): Square Adjacency matrix
        """
        n = int(np.floor(np.sqrt(2*len(l))))+1
        A= np.zeros((n,n))
        A[np.triu_indices(n,1)]=-l.ravel()
        A= A+A.T
        return A

    def L_to_l(self, L):
        """Vectorize the upper triangular part of given square matrix and returns it.

        Indexing is done row wise.

        Args:
            L (np.array): Square matrix matrix

        Returns:
            l (np.array): vectorized upper triangular part of the square matrix. 
        """
        n = L.shape[0]
        np.zeros((n*(n-1)//2,1))
        l = L[np.triu_indices(n,1)] # burada indexing hatasi verebilir, (len,1) to (len,)
        return l
    
    def L_to_A(self,L):
        """Convert given laplacian matrix into adjacency.

        Args:
            L (np.array): Laplacian matrix

        Returns:
            A (np.array): Adjacency matrix
        """
        A = -L
        A[np.diag_indices_from(A)]=0
        return A

    def A_to_L(self,A):
        """Convert given adjacency matrix into laplacian.

        Args:
            A (np.array): Adjacency matrix

        Returns:
            L (np.array): Laplacian matrix
        """
        L = np.zeros(A.shape)
        L = -A
        for i in range(A.shape[0]):
            L[i,i] = np.sum(A[i,...])
        return L

    def edges_in_L(self,L, thr=1e-4): # NORMALIZE L KISMINA BIR DAHA BAK 
        """Tresholds the edges of the laplacian.

        Args:
            L (_type_): _description_
            thr (_type_, optional): _description_. Defaults to 1e-4.

        Returns:
            _type_: _description_
        """
        #L = self.normalize_L(L)
        E =np.zeros(L.shape)
        E[L<-thr]=1
        idx = np.diag_indices_from(E)
        E[idx] = 0
        return E

    def Fmeasure(self, b=1, Egt=None, E=None):
        """Compute the f measure of the graph wrt another one.
        
        Compares the graph edges with a ground truth graph structure or
        treats the graph as a ground truth graph structure and computes Presicion,
        Recall and Fb measures

        Args:
            b (int, optional): F_b measure. Defaults to 1.
            Egt (_type_, optional): Ground truth edge matrix. Defaults to None.
            E (_type_, optional): Edge matrix. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            Fb: F_b score
            P: Precision score
            R: Recall score
        """
        # GT disaridan verilebilir, GT iceride olabilir
        if type(E)==type(np.zeros(1)) and type(Egt)!=type(np.zeros(1)): # Ground truth edges is the self graph edges
            Egt=self.E
        elif type(E)!=type(np.zeros(1)) and type(Egt)==type(np.zeros(1)): # Ground truth edges are given to evaluate self graph
            E=self.E
        else:
            raise ValueError("Graph.Fmeasure() Ground truth can either be given from outside or can be the self.")
        p_ind = Egt==1
        n_ind = Egt==0
        tp = sum(E[p_ind])
        fp = sum(sum(p_ind))-tp
        fn = sum(E[n_ind])
        tn = sum(sum(n_ind))-fn
        if tp ==0:
            P = 0; R = 0
            Fb = 0
        else :
            P = tp/(tp+fp); R = tp/(tp+fn)
            Fb = ((1+b**2)*P*R)/((b**2)*P+R )
        return (Fb, P, R)
    
    def auprc(self, **kwargs):
        """Return the auprc score of the graph by comparing it with a given ground truth or vice versa.

        Arguments:
            'A' (np.ndarray): If adjacency matrix is given then self.E is the ground truth
            'Egt'(np.ndarray): If ground truth edges is given then self.A is the estimate.

        Returns: Area Under Precision Recall Curve
        """
        # if A is given, self.E is the ground truth
        # if Egt is given, self.A is the estimate
        if type(kwargs.get('Egt',False)) == type(np.zeros(1)):
            Egt = kwargs['Egt']
            A = self.A
        elif type(kwargs.get('A',False)) == type(np.zeros(1)):
            Egt = self.E
            A = kwargs['A']
        else:
            raise ValueError("Graph.auprc() is given incompatible values")
        return auprc(Egt,A)

    def density(self):
        """Compute the density of the graph structure.

        Returns:
            d (float): The density of the graph
        """
        n = self.E.shape[0]
        d = np.sum(self.E) / (n**2-n)
        return d


    def image_L(self, title=None, **kwargs):
        """Graph the laplacian matrix of the graph object into a 2d image.

        Args:
            title (str, optional): Given title to the figure.
            Defaults to 'Graph Laplacian'

        Returns:
            (fig, ax, pc): Returns the matplotlib figure.
        """
        if title==None:
            title = "Graph Laplacian"
        fs = kwargs.get('figsize',(4,4))
        fig, axe = plt.subplots(1,1,figsize=fs)
        fig.suptitle(title)
        pc = axe.matshow(self.L)
        fig.colorbar(pc, ax=axe)
        if self.graph_type!=None:
            title = self.graph_type.swapcase() + f" - {self.graph_params}"
            axe.set_title(title)
        return (fig,axe,pc)

    def image_A(self, title=None, **kwargs):
        """Graph the adjacency matrix of the graph object into a 2d image.

        Args:
            title (str, optional): Given title to the figure.
            Defaults to 'Graph Adjacency'

        Returns:
            (fig, ax, pc): Returns the matplotlib figure.
        """
        if title==None:
            title = "Graph Adjacency"
        fs = kwargs.get('figsize',(4,4))
        fig, axe = plt.subplots(1,1,figsize=fs)
        fig.suptitle(title)
        pc = axe.matshow(self.A)
        fig.colorbar(pc, ax=axe)
        if self.graph_type!=None:
            title = self.graph_type.swapcase() + f" - {self.graph_params}"
            axe.set_title(title)
        return (fig,axe,pc)

    def image_E(self, title=None,**kwargs):
        """Graph the edges of the graph object into an image.

        Args:
            title (str, optional): Given title to the figure.
            Defaults to 'Graph Edges'

        Returns:
            (fig, ax, pc): Returns the matplotlib figure.
        """
        if title==None:
            title = "Graph Edges"
        fs = kwargs.get('figsize',(4,4))
        fig, axe = plt.subplots(1,1,figsize=fs, layout='constrained')
        fig.suptitle(title)
        pc = axe.matshow(self.E)
        if self.graph_type!=None:
            subtitle = self.graph_type.swapcase() + f" - {self.graph_params}"
            axe.set_title(subtitle)
        return (fig,axe,pc)


    def plot_L(self, title=None,**kwargs):
        """Plot the eigenvalues and vectors of the graph laplacian."""
        if title==None:
            if self.graph_type!=None:
                title = self.graph_type.swapcase() + f" - {self.graph_params} Eigendecomposition"
            else:
                title = "Graph Laplacian Eigendecomposition"
        fs = kwargs.get('figsize',(4,4))
        fig, axs = plt.subplots(1,2,figsize=fs, layout='constrained')
        fig.suptitle(title)
        pc = axs[0].plot(self.lda)
        axs[0].set_title("Eigenvalues")
        pc = axs[1].plot(self.V)
        axs[1].set_title("Eigenvectors")
        return (fig, axs, pc)
    
    def normalize_L(self,L):
        """Normalize the given Laplacian with its spectral norm."""
        if norm(L)!=0:
            return L/norm(L)
        else:
            return L

class ProductGraph(Graph):
    """Product graph object.

    ProductGraph.PG
        2 ways to initialize:
            1. With factor graphs
            2. With random product graph parameters
        ###
        1. Either factor laplacians, upper(laplacian), or graphs are given
        kwargs = {  
            'n': dimensions (Optional, defaults to the dimensions of factors in the same order) 
            'ls': [] List of vectorized upper triangual part of the laplacians (Optional)
            'Ls': [] List of laplacians (Optional)
            'Gs': [] List of Graphs (Optional, Initialized user defined graphs and not networkx graphs)
            'modes': [m1,m2,m3, ..., m_len(Gs)] maps the factor graphs to the modes of the data tensor (Optional, defaults to [1,2,3,...])
            'product_type': 'kron'/'cart'/'strong' (Optional, defaults to 'cart')
        }
        ###
        2. Random product graph parameters are given
        kwargs = {
            'n':
            'modes':
            'factor_graph_params': [List of rand_graph_params(dict)] (description is in Graph class)
            'product_type': 'kron'/'cart'/'strong' (Optional, defaults to 'cart')
        }
    """

    def __init__(self, **kwargs):
        self.product_type = kwargs.get('product_type','cart') # defaults to 'cart'
        self.n = kwargs.get('n')
        self.modes = kwargs.get('modes')
        if kwargs.get('ls'):
            ls = kwargs.get('ls').copy()
        else:
            ls = None
        
        if type(kwargs.get('Ls',None))!=type(None):
            Ls = kwargs.get('Ls').copy()
        else:
            Ls = None
        if kwargs.get('Gs'):
            Gs= copy.deepcopy(kwargs.get('Gs'))
        else:
            Gs = None
        fg_params = kwargs.get('factor_graph_params')
        if fg_params !=None:
            self.FactorGraphs = [Graph(rand_graph_params=fgp) for fgp in fg_params]
        elif Gs!=None:
            if len(Gs)==1: raise ValueError("ProductGraph.__init__(Gs=G): Only one graph is given to the product graph")
            self.FactorGraphs = copy.deepcopy(Gs)
        elif type(Ls)!=type(None):
            self.FactorGraphs = [Graph(L=Li) for Li in Ls]
        elif ls!=None:
            self.FactorGraphs = [Graph(l=li) for li in ls]
        else:
            raise ValueError("ProductGraph.__init__(): Neither random graph parameters nor factor graphs are provided.")
        
        if self.n==None:
            self.n = [fg.n for fg in self.FactorGraphs]
        if self.modes==None:
            self.modes = [i+1 for i in range(len(self.FactorGraphs))]
        
        if self.product_type == 'cart':
            self._cart_prod()
        elif self.product_type == 'kron':
            self._kron_prod()
        elif self.product_type == 'strong':
            self._strong_prod()
        else:
            raise ValueError(f"ProductGraph.__init__(): Product type is unknown ("+self.product_type+")")
        self.V = self.PG.V.copy()
        self.L = self.PG.L.copy()
        self.E = self.PG.E.copy()



    def _cart_prod(self): # Eigenvaluelari normalize ettirmedim. Bir sorun cikarsa aklinda olsun
        """Compute the cartesian product of the Factor Graphs.
        
        Factor Graphs are stored within self.FactorGraphs list
        and initializes a new Graph object in self.PG.
        It's called when the Product Graph object is initialized.
        """
        I = [np.ones((1,ni)) for ni in self.n]
        II = [np.eye(ni) for ni in self.n]
        V = []; lda=[]; j=0
        for i in range(len(self.n)):
            if self.modes.count(i+1)==1:
                if i==0:
                    a = [self.FactorGraphs[j].lda] + I[i+1:]
                elif i==len(self.n)-1:
                    a = I[:i]+[self.FactorGraphs[j].lda]
                else:
                    a = I[:i]+[self.FactorGraphs[j].lda] + I[i+1:]
                
                lda.append( list_kronecker(a))
                V.append(self.FactorGraphs[j].V)
                j+=1
            else:
                V.append(II[i])
        et = sum(lda)
        #et[et < 1e-8] = 0
        #et /= np.max(et)
        pg_lda =et.reshape((1,et.size))
        pg_V = list_kronecker(V)
        pg_L = pg_V@ np.diag(pg_lda.ravel())@pg_V.T
        #self.E = self.edges_in_L(self.L, 1e-6)
        #self.A = self.L_to_A(self.L)
        self.PG = Graph(L=pg_L)#nx.from_numpy_array(self.A)

    def _kron_prod(self): # MIGHT BE WRONG DIDNT CHECK PROPERLY
        """Compute the kronecker product of the Factor Graphs.
        
        Factor Graphs are stored within self.FactorGraphs list
        and this initializes a new Graph object in self.PG.
        It's called when the Product Graph object is initialized.
        """
        I = [np.ones((1,ni)) for ni in self.n]
        II = [np.eye(ni) for ni in self.n]
        V = []; lda=[]; j=0
        for i in range(len(self.n)):
            if self.modes.count(i+1)==1:
                lda.append(self.FactorGraphs[j].lda)
                V.append(self.FactorGraphs[j].V)
                j+=1
            else:
                lda.append(I[i])
                V.append(II[i])
        et = list_kronecker(lda)
        #et[et < 1e-8] = 0
        #et /= np.max(et)
        pg_lda =et.reshape((1,et.size))
        pg_V = list_kronecker(V)
        pg_L = pg_V@ np.diag(pg_lda.ravel())@pg_V.T
        #self.E = self.edges_in_L(self.L, 1e-6)
        #self.A = self.L_to_A(self.L)
        self.PG = Graph(L=pg_L)#nx.from_numpy_array(self.A)

    def _strong_prod():
        """Compute the strong product of the Factor Graphs.
        
        It's called when the Product Graph object is initialized.
        """
        pass


    def PG_Fmeasure(self, b=1, Egt=None, E=None):
        """Compute the product graph F measure."""
        pg_fscore= self.Fmeasure(b=1,Egt=Egt,E=E)
        # factor_fscores = [   for fg in self.FactorGraphs]
        Fscore = {'b':b,
                  'Fscore':pg_fscore,
                  'factor_fscores':None,
                  'modes':self.modes}
        return Fscore
        


class GraphProcess(object):
    """Stochastic Graph Process used to generate random signals defined on a graph.
    
        Initialized by initializing a graph and a filter type. The graph initialization arguments are
    the same as Graph or ProductGraph class instances. The graph process filter is initialized by specifying
    the filter_type among the following options,
        filter_type (str): Optional, defaults to 'Gaussian'  
        - 'Gaussian' 
        - 'Tikhonov'
        - 'Heat'
        filter_parameters (float): Optional, defaults to 10 for the above options.
        filter_type (str):
        - 'Markov Random Field' 
        filter_parameters (tuple): (sigma, delta)
        filter_type (str): 
        - 'Polynomial'
        h (np.array): Graph polynomial coefficients.    

        After the GraphProcess is initialized, a random graph signal can be generated via
        self.gen_signal() method.
    """

    def __init__(self, **kwargs):

        if kwargs.get('rand_graph_params',False):
            self.Graph = Graph(rand_graph_params=kwargs.get('rand_graph_params'))
        elif kwargs.get('factor_graph_params',False):# Initialize with random product graph arguments
            self.Graph = ProductGraph(**kwargs)   
        elif kwargs.get('Graph',False):             # Initialize with the Mert defined graph instance
            self.Graph = copy.deepcopy(kwargs.get('Graph'))    
        elif kwargs.get('L',False):                 # Initialize with Laplacian matrix
            self.Graph = copy.deepcopy(Graph(L=kwargs.get('L')))
        elif kwargs.get('l',False):                 # Initialize with vectorized upper triangilar part of a laplacian matrix
            self.Graph = copy.deepcopy(Graph(l=kwargs.get('l')))
        elif kwargs.get('G',False):                 # Initialize with networkx graph
            self.Graph = copy.deepcopy(Graph(G=kwargs.get('G'))) 
        elif kwargs.get('A',False):                 # Initialize with adjecancy matrix
            self.Graph = Graph(A=kwargs.get('A')) 
        else:
            raise ValueError("Graph process initialization keywords are wrong")
        if kwargs.get('seed',False):
            rng = np.random.default_rng(kwargs['seed'])
        else:
            rng = np.random.default_rng()
        self.filter_type = kwargs.get('filter_type','Gaussian')
        self.filter_parameters = kwargs.get('filter_parameters',10)
        self.GSO=kwargs.get('GSO','L')

        if self.filter_type == "Markov Random Field":
            self.filter_parameters = kwargs.get('filter_parameters',(1,np.random.default_rng().uniform(0.1,2)))
        if type(self.Graph) == Graph:
            if self.GSO=='L':
                lda = self.Graph.lda
                L = self.Graph.L
                V = self.Graph.V
                lda[lda<1e-8]=0
                lda /=np.max(lda)
            elif self.GSO=='A':
                lda, V = LA.eigh(self.Graph.A)
                L = self.Graph.L
                lda[lda<1e-8]=0
                lda /=np.max(lda)
        elif type(self.Graph) == ProductGraph:
            if self.GSO=='L':
                lda = self.Graph.PG.lda
                lda[lda<1e-8]=0
                L = self.Graph.PG.L
                V = self.Graph.PG.V
                lda /=np.max(lda)
            elif self.GSO=='A':
                lda, V = LA.eigh(self.Graph.PG.A)
                L = self.Graph.PG.L
                lda[lda<1e-8]=0
                lda /=np.max(lda)
        else:
            raise ValueError("Somehow you managed to initialize the product graph withouth any graph. Well done!")
        
        # Initialize graph filters or covariances
        if self.filter_type== "Gaussian":                   ## Smooth and stationary graph processes
            self.h = np.ones(lda.size) # I made it ones, We may need to change this
            alpha = self.filter_parameters
            self.h[lda > 0] = 1/np.sqrt(alpha*lda[lda>0])
            self.process_type= "Smooth and stationary"
            self.C = V @ np.diag(self.h*self.h) @ V.T
        elif self.filter_type== "Tikhonov":
            alpha = self.filter_parameters
            self.h = 1/(1+alpha*lda)
            self.process_type= "Smooth and stationary"
            self.C = V @ np.diag(self.h*self.h) @ V.T
        elif self.filter_type== "Heat":
            alpha = self.filter_parameters
            self.h = np.exp(-alpha*lda)
            self.process_type= "Smooth and stationary"
            self.C = V @ np.diag(self.h*self.h) @ V.T
        elif self.filter_type== "Polynomial":               ## Only stationary graph processes
            self.process_type= "Stationary"
            if type(kwargs.get('h',None))!=type(None): # If filter coefficients were given
                self.h= kwargs.get('h').ravel()
                self.filter_length = len(self.h)
            else: # If the length is not given it is assumed to be 5
                self.filter_length=  kwargs.get('filter_length',4)
                self.h = rng.uniform(1e-8,1, self.filter_length)
                if norm(self.h) ==0:
                    print("Stationary graph process somehow has 0 filter coefficients")
                    pass
                else: # Normalize h
                   self.h = self.h/(norm(self.h)) # *self.filter_length
            self.H = np.zeros(L.shape)
            #lda = lda/np.max(lda) # Normalizing eigenvalues to be 1 at maximum H grows very rapidly otherwise
            lda_tmp = np.zeros(L.shape[0]) 
            # lda_powered = np.ones(L.shape[0])
            # for i in range(self.filter_length):
            #     lda_tmp += self.h[i]*lda_powered # Sum h_i * Lambda^i
            #     lda_powered = lda_powered*lda
            
            for i in range(self.filter_length):
                lda_tmp += (lda**i)*self.h[i]

            self.H = V @ np.diag(lda_tmp) @ V.T
            self.C = self.H@ self.H.T
            self.GSO_coefficients = self.h
            self.h = lda_tmp
        elif self.filter_type == "Markov Random Field":
            self.process_type= "Stationary"
            try:
                sigma, delta = self.filter_parameters
                self.C = LA.inv(sigma*np.eye(L.shape[0])+delta*L)
            except:
                raise ValueError("Stationary Markov Random Field Graph process had incompatible parameters (sigma, delta) given.")
        elif self.filter_type == "Smooth non-stationary":   ## Only smooth graph processes
            self.process_type = "Smooth non-stationary"
            self.C = V @ LA.pinv(np.diag(lda),hermitian=True) @ V.T
        else:
            raise ValueError("Filter/Process type could not be parsed.")
  



    def gen_signal(self, NoS,noise_amount=0.1,sigma=0, noise_type="AGWN", masked=True):
        """Generate a graph signal with possibly unobserved masked entries on the graph structure.

        Args:
            NoS (integer): Number of samples
            sigma (float, optional): Percentage of the corrupted elements. Defaults to 0.
            noise_type (str, optional): The noise type. Defaults to "AGWN"masked=True.
            noise_amount (float, optional): If the noise type is AGWN, it represents 
                the ratio of the noise to signal power. Defaults to 0.1. If the noise type is 
                Gross, it should be (abs_max_peak,cardinality)
        Returns:
            result (dict): Dictionary of ground truth, noisy and masked signal with signal
            generation parameters. They keys are as follows,
                'X'-> Ground truth data.
                'Xn'-> Noisy data.
                'mask'-> Mask of unobserved entries.
                'noise'-> Noise ratio
                'sigma'-> Percentage of corrupted elements.
        """
        obs_ratio = 1-sigma
        if obs_ratio <=0 or obs_ratio >1:
            raise ValueError("Corruption ratio cannot be more than or equal to one or smaller than zero")    
        if self.process_type == "Smooth and stationary":
            X = self.__gen_smooth_and_stationary_signal(NoS)
        elif self.filter_type== "Polynomial":
            X = self.__gen_polynomial_gso_filtered_signal(NoS)
        elif self.filter_type== "Markov Random Field":
            X = self.__gen_MRF_signal(NoS)
        elif self.filter_type== "Smooth non-stationary":
            X = self.__gen_smooth_nonstationary_signal(NoS)
        else:
            raise ValueError("Idk what's wrong")

        Xn = add_noise(X, noise_type, noise_amount)
        Xn, mask = corrupt_elements(Xn,obs_ratio, masked=masked)
        return {'X':X,'Xn':Xn,'mask':mask, 'NoS':NoS, 'noise':noise_amount, 'sigma':sigma} 
        

    
    
        

    def __gen_smooth_and_stationary_signal(self, NoS):
        """Generate smooth and stationary data."""
        if type(self.Graph) == Graph:
            V = self.Graph.V
        elif type(self.Graph) == ProductGraph:
            V = self.Graph.PG.V
        n = self.Graph.n
         
        rng = np.random.default_rng()
        X0 = rng.multivariate_normal(np.zeros(np.prod(n))//1, np.eye(np.prod(n))//1, int(NoS)).T
             # White noise

        X0_hat = V.T@X0 # GFT of X0
        X_ = (V@np.diag(self.h.ravel())@X0_hat).T # X_ is the (len(n)+1)'th mode matricization of the data
        try:
            n_new = [ni for ni in n]+[NoS]
        except TypeError:
            n_new = [n]+[NoS]
        X = m2t(X_,n_new,len(n_new))
        return X    

    def __gen_polynomial_gso_filtered_signal(self, NoS): # Polynomial of Graph Shift Operator as filter. X= H.w (w:white noise, H:filter)
        rng = np.random.default_rng()
        n = self.Graph.n
        X0 = rng.multivariate_normal(np.zeros(np.prod(n)), np.eye(np.prod(n)), NoS).T # Transpose is intentional
        X0_hat = (self.H@X0).T                  # The signals matricization in the sample mode (last mode)
        n = self.Graph.n
        try:
            n_new = [ni for ni in n]+[NoS]
        except TypeError:
            n_new = [n, NoS]
        X = m2t(X0_hat,n_new,len(n_new))
        return X

    def __gen_MRF_signal(self, NoS):
        n = self.Graph.n
        rng = np.random.default_rng()
        X0 = rng.multivariate_normal(np.zeros(np.prod(n)), self.C, NoS).T
        n = self.Graph.n
        n_new = [ni for ni in n]+[NoS]
        X = m2t(X0,n_new,len(n_new))
        return X

    def __gen_smooth_nonstationary_signal(self, NoS):
        rng = np.random.default_rng()
        if type(self.Graph) == Graph:
            lda = self.Graph.lda
            V = self.Graph.V    
        elif type(self.Graph) == ProductGraph:
            lda = self.Graph.PG.lda
            V = self.Graph.PG.V
        cov = LA.pinv(np.diag(lda),hermitian=True)
        
        X0 = rng.multivariate_normal(np.zeros(np.prod(n)), cov, NoS).T
        
        n_new = [ni for ni in n]+[NoS]
        X = m2t( (V@X0).T ,n_new,len(n_new))
        return X


    
def add_noise(X, noise_type='AGWN', noise_amount=0.1):
    """Apply additional noise to the given signal.

    Args:
        X (np.ndarray): Signal to add noise to
        noise_type (str, optional): _description_. Defaults to 'AGWN'.
        noise_amount (float, optional): If the noise type is AGWN, it represents the ratio of the noise to signal power. Defaults to 0.1
                                        If the noise type is Sparse, it should be (abs_max_peak,cardinality)
    Raises:
        ValueError: If noise_type is invalid, throws ValueError

    Returns:
        Xn (np.ndarray): Returns the noisy signal.
    """
    rng = np.random.default_rng()
    # Add noise
    if noise_type=="AGWN":
        X_norm = np.linalg.norm(X)
        if noise_amount==0:
            E = np.zeros(X.shape)
            Xn = X.copy()
        elif noise_amount>0:
            E = rng.normal(0, 1, X.shape)
            E_norm = np.linalg.norm(E)
            Xn =X+ E*(noise_amount*X_norm/E_norm)
        else:
            raise ValueError("Invalid AWG Noise amount given.")
    elif noise_type =="Sparse":
        assert type(noise_amount)==tuple
        assert len(noise_amount)==2
        (peak_ratio,card) =noise_amount
        assert peak_ratio>0
        assert card>=0 and card<=1
        peak = peak_ratio*(np.max(X)-np.min(X))
        perm = rng.permutation(X.size) 
        obs_cardlty = np.floor(card*X.size).astype(int)  
        clean_idx = perm[obs_cardlty:]
        E = peak*np.ones(X.shape)#rng.uniform(-peak,peak,X.shape)
        E.flat[clean_idx]=0
        Xn = X+E
    return Xn

def corrupt_elements(X,obs_ratio, masked=False):
    """Select random elements from the tensor X and assigns them and mask those entries.

    Args:
        X (np.ndarray): Data to be corrupted
        obs_ratio (float): Ratio of the entries left unmasked.  
        masked (bool, optional): If set to True, X is turned converted to 
        masked array before returned. Defaults to False.

    Returns:
        Xn (np.ndarray): masked data
        mask (np.ndarray): Tensor with the same dimensions as X with masked
        entries set to 1.
    """
    rng = np.random.default_rng()
    if obs_ratio !=1:
        perm = rng.permutation(X.size) 
        obs_cardlty = np.floor(obs_ratio*X.size).astype(int)  
        obs_idx = perm[:obs_cardlty]                    # The indexes of the observed elements
        vec_mask = np.ones(X.size, dtype='bool')        # Start with all masked vector mask
        vec_mask.flat[obs_idx]=False                    # Set the observed indices False
    else:
        vec_mask = np.zeros(X.size, dtype='bool')
    Xn = X.copy()
    mask = vec_mask.reshape(X.shape)
    if masked:
        Xn = np.ma.array(X, mask=mask)
    return Xn, mask 

    
def list_kronecker(L):
    """Return the kronecker product of the arrays in the List.

    Args:
        I (list of np.arrays): _description_
    Returns:
        np.array: kronecker product of the arrays
    """
    a=L[0]
    k=1
    while k<len(L):
        a= np.kron(a,L[k])
        k+=1
    return a


def how_diagonal(C,V):
    D = V.T@C@V
    return norm(np.diag(D))/norm(C)

def init_M(n): # Works!
    """Compute the transformation matrix M that transforms the vectorized
    upper triangular part of the laplacian L. In other words, M satisfies the
    following relation for l=upper(L)
        M @ l = L

    Args:
        n (int): size of the square matrix L

    Returns:
        M (np.array): Transformation matrix M
    """
    m = lambda i,j: (i-1)*n+j # (i,j)'th element of L's place in vecL
    M = np.zeros((n*n, n*(n-1)//2))
    for j in range(1,n+1):
        for i in range(1,n+1):
            if i==j:
                # Fill left hand side of the diag loop
                current_idx=i-1
                step_len=n-2
                for _ in range(i-1):
                    M[m(i,j)-1,current_idx-1]=-1
                    current_idx += step_len
                    step_len-=1
                # Fill the right hand side of the diag loop
                r_rpt = n-i
                start_idx = n*(i-1)-i*(i-1)//2
                end_idx = start_idx+r_rpt
                M[(i-1)*n+j-1,start_idx:end_idx]=-1
            else:
                p= find_Lij_in_l(i,j,n)
                M[m(i,j)-1,p-1]=1 
    return M

def find_Lij_in_l(i,j,n): 
    """
    Maps (i,j)th index of a laplacian matrix to the index of the
    vectorization of upper triangular part of the laplacian. 
    upper(L)=l --> L_ij = l_p
    Args:
        i: row index
        j: column index
        n: size of the square matrix
    Returns:
        p: index of the (i,j)'th element in upper(L)=l
    """
    if i>n or j >n:
        raise IndexError('find_Lij_in_l: i or j > n')
    else:
        if i<j:
            return (n*(i-1)-i*(i-1)//2 + j -i)
        elif i>j:
            return (n*(j-1) - j*(j-1)//2 + i-j)
        else:
            print(f"i,j = ({i},{j})")
            raise IndexError('find_Lij_in_l: i=j')


def account_for_PSD_eigval_fperror(C,r=10):
    """ Accounts for the floating point precision errors of PSD matrices.
    Ensures that the matrix C is Positive semi definite.
    Args:
        C (square matrix): A square matrix that should have been a PSD
    Returns:
        Cnew (square matrix): A square matrix that is very close to the original
        but without floating point errors.
    """
    assert len(C.shape) == 2
    assert C.shape[0] == C.shape[1]
    lda, _ = np.linalg.eig(C)
    Cnew = C.copy()
    fp_error= min(lda)
    flag = False
    if fp_error<0:
        print(f"floating point error on smallest eigenvalue was {fp_error}")
        flag = True
    while np.min(lda)<0:
        Cnew = Cnew + np.eye(C.shape[0])*r*np.abs(fp_error)
        lda, _ = np.linalg.eig(Cnew)
    if flag:
        print(f"New smallest eigval = {np.min(lda)}")
    return Cnew

def estimate_covariance(X):
    pass



def load_GP(self, dataname): ### BUNLAR DUZENLENECEK 
    cwd= os.getcwd()
    fname = os.path.join(cwd, 'results','stationary_PGL',dataname+'.mat')
    D = loadmat(fname)
    PG = ProductGraph(Ls= D['FGLs'])
    GP = GraphProcess(Graph= PG, filter_type=D['filter_type'], filter_parameter=D['filter_parameters'], h=D['GSO_coefficients'])
    return GP, D

def save_GP(self, GP, search_logs, save_name, PGmetric=False):
    """ Saves a graph process into memory

    Args:
        GP (GraphProcess): _description_
        search_logs (_type_): _description_
        save_name (_type_): _description_
        PGmetric (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_
    """
    cwd= os.getcwd()
    fname = os.path.join(cwd, 'results','stationary_PGL', save_name+'.mat')
    if type(GP.Graph) == Graph:
        PGL = GP.Graph.L
        FGLs = [GP.Graph.L]
        FGs_types = [GP.Graph.graph_type]
        FGs_params = [GP.Graph.graph_params]
        product_type = 'None'
        gt_density = [GP.Graph.density()]
    elif type(GP.Graph) == ProductGraph:
        PGL = GP.Graph.PG.L
        product_type = GP.Graph.product_type
        FGLs = [ G.L for G in GP.Graph.FactorGraphs]
        FGs_types = [G.graph_type for G in GP.Graph.FactorGraphs]
        FGs_params = [G.graph_params for G in GP.Graph.FactorGraphs]
        gt_density = [ G.density() for G in GP.Graph.FactorGraphs]
    if GP.process_type == "Smooth and stationary":
        filter_h = GP.h
        GSO_coefficients = 'None'
    elif GP.process_type == "Stationary":
        filter_h = GP.h 
        GSO_coefficients = GP.GSO_coefficients
        
    else:
        raise ValueError("GP is not smooth or stationary!")

    
    D = {'PGL' : PGL, 'FGLs':FGLs, 'FGs_types':FGs_types, 'FGs_params':FGs_params,
        'product_type':product_type, 'filter_type': GP.filter_type, 'filter_parameters':GP.filter_parameters,
        'filter_h':filter_h, 'GSO_coefficients': GSO_coefficients, 
        'mode1_search': search_logs[0], 'mode2_search':search_logs[1],
        'gt_densities':gt_density }
    if PGmetric:
        D['PG_search'] = search_logs[2]
    D =_dict_to_matfile(D)
    savemat(fname, D)
    print("GP and its hyperparameters for Algo4 has been saved as\n" +fname, flush=True)

def _dict_to_matfile(self, D):
    """Saves a dictionary with to mat file.

    Args:
        D (_type_): _description_

    Returns:
        _type_: _description_
    """
    for k in D.keys():
        if type(D[k])==list:
            D[k]=np.array(D[k])
        elif  type(D[k])==dict:
            for k2 in D[k].keys():
                if type(D[k][k2])==list:
                    D[k][k2] = np.array(D[k][k2])
    return D