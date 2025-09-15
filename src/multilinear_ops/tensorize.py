import numpy as np
import torch

def tensorize(X, og_shape, rows, cols=None):
    """WRITE ELABORATE DESCRIPTION

    Args:
        X (np.array): Matrix to be tensorized
        og_shape (tuple): Original shape of the tensor before matricized by matricize function
        rows (list of int): The indices corresponding to rows used by matricize function 
        cols (_type_, optional): Optional. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    N = len(og_shape)
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
    d_r = [og_shape[i-1] for i in rows]
    d_c = [og_shape[i-1] for i in cols]
    dims_ = [i-1 for i in dims]
    rows_ = [i-1 for i in rows]
    cols_ = [i-1 for i in cols]
    X = X.ravel().reshape(d_r+d_c)
    if isinstance(X, np.ndarray):
        X = np.moveaxis(X, dims_, rows_+cols_)
    elif isinstance(X, torch.Tensor):
        X = torch.moveaxis(X, dims_, rows_+cols_)
    return X