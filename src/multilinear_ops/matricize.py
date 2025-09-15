import numpy as np
import torch

def matricize(X, rows, cols=None):
    """WRITE ELABORATE DESCRIPTION

    Args:
        X (_type_): _description_
        rows (_type_): _description_
        cols (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    N = len(X.shape)
    dims = [i for i in range(1,N+1)]
    # Check to see if indicated rows are valid 
    if isinstance(rows, tuple) or isinstance(rows, list):
        if not set(rows).issubset(set(dims)):
            raise ValueError("Indicated rows are not valid!")
    
    # If the columns are not specified, map the dimensions other than rows to columns
    if isinstance(cols, type(None)):
        cols = tuple(set(dims).difference(set(rows)))
    elif isinstance(cols, tuple) or isinstance(cols, list):
        # If the columns are specified check for their validity
        if not set(cols).isdisjoint(set(rows)):
            raise ValueError("Indicated rows and columns intersect!")
        elif set(cols).union(set(rows)) != set(dims):
            raise ValueError("Rows and columns do not cover all dimensions")
    else:
        raise ValueError("Indicated columns are not valid!")
    
    d_r = [X.shape[i-1] for i in rows]
    d_c = [X.shape[i-1] for i in cols]
    dims_ = [i-1 for i in dims]
    rows_ = [i-1 for i in rows]
    cols_ = [i-1 for i in cols]
    if isinstance(X, np.ndarray):
        X = np.moveaxis(X, rows_+cols_, dims_)
    elif isinstance(X, torch.Tensor):
        X = torch.moveaxis(X, rows_+cols_, dims_)
    X = X.ravel().reshape((np.prod(d_r, dtype=int),np.prod(d_c, dtype=int)))
    return X


def unfolded_index(indices, shape, rows, cols=None):
    """Maps the tensor indices to row and column index of its matricization

    Args:
        indices (tuple): Indices specifying the tensor element
        shape (tuple): Shape of the original tensor
        row (tuple): Indicator list of the dimensions mapped to the rows. Elements must be integers.
        col (tuple): Indicator list of the dimensions mapped to the columns. Elements must be integers.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        r,c (tuple): row and column indices of the entry specified by `indices`
    """
    N = len(shape)
    dims = [i for i in range(1,N+1)]
    # Check to see if indicated rows are valid 
    if isinstance(rows, tuple) or isinstance(rows, list):
        if not set(rows).issubset(set(dims)):
            raise ValueError("Indicated rows are not valid!")

    # If the columns are not specified, map the dimensions other than rows to columns
    if isinstance(cols, type(None)):
        cols = tuple(set(dims).difference(set(rows)))
    elif isinstance(cols, tuple) or isinstance(cols, list):
        # If the columns are specified check for their validity
        if not set(cols).isdisjoint(set(rows)):
            raise ValueError("Indicated rows and columns intersect!")
        elif set(cols).union(set(rows)) != set(dims):
            raise ValueError("Rows and columns do not cover all dimensions")
    else:
        raise ValueError("Indicated columns are not valid!")
    M = len(rows)
    dims_r = [dims[i-1] for i in rows]
    dims_c = [dims[i-1] for i in cols]
    i_r = [indices[i-1] for i in rows]
    i_c = [indices[i-1] for i in cols]
    
    r = 0
    for k in range(M):
        r += i_r[k]*np.prod(d_r[k+1:], dtype=int)
    c = 0
    for k in range(N-M):
        c += i_c[k]*np.prod(d_c[k+1:], dtype=int)
    return (r,c)
