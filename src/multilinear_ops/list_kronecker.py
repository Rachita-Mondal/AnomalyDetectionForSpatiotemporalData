import numpy as np


def list_kronecker(L):
    """Compute the kronecker product of the arrays in the List.

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


# def graph_cart_prod(n, modes, L): # Eigenvaluelari normalize ettirmedim. Bir sorun cikarsa aklinda olsun
#     ###
#     I = [np.ones((1,ni)) for ni in n]
#     II = [np.eye(ni) for ni in n]
#     V = []; lda=[]; j=0
#     for i in range(len(n)):
#         if modes.count(i+1)==1:
#             if i==0:
#                 a = [FactorGraphs[j].lda] + I[i+1:]
#             elif i==len(n)-1:
#                 a = I[:i]+[FactorGraphs[j].lda]
#             else:
#                 a = I[:i]+[self.FactorGraphs[j].lda] + I[i+1:]
            
#             lda.append( list_kronecker(a))
#             V.append(self.FactorGraphs[j].V)
#             j+=1
#         else:
#             V.append(II[i])
#     et = sum(lda)
#     #et[et < 1e-8] = 0
#     #et /= np.max(et)
#     pg_lda =et.reshape((1,et.size))
#     pg_V = list_kronecker(V)
#     pg_L = pg_V@ np.diag(pg_lda.ravel())@pg_V.T
#     #self.E = self.edges_in_L(self.L, 1e-6)
#     #self.A = self.L_to_A(self.L)
#     self.PG = Graph(L=pg_L)#nx.from_numpy_array(self.A)