import numpy as np

def merge_tensors(T1, T2, modes_1, modes_2=None):
    """
    Merges a tensor with tensor factors or another tensor on specified modes.
    
    Parameters:
    T1 : numpy.ndarray
        First tensor.
    T2 : numpy.ndarray or list of numpy.ndarrays
        Second tensor or a list of tensors.
    modes_1 : list of int
        Modes of the first tensor along which to merge.
    modes_2 : list of int, optional
        Modes of the second tensor along which to merge. If not provided, assumes T2 is a list of tensors.
        
    Returns:
    numpy.ndarray
        Merged tensor.
    """
    s1 = np.array(T1.shape)  # Get the size of the first tensor
    if max(modes_1) > len(s1):
        s1 = np.concatenate([s1, np.ones(max(modes_1) - len(s1), dtype=int)])
    
    N = len(s1)  # Number of modes of T1

    if modes_2 is None:  # Merging T1 with a list of factors
        if not isinstance(T2, list):
            raise ValueError('Missing merging modes for the second tensor.')
        
        l = len(modes_1)
        TO = merge_tensors(T1, T2[0], [modes_1[0]], [2])
        for i in range(1, l):
            mode = modes_1[i] - sum(modes_1[:i] < modes_1[i])
            TO = merge_tensors(TO, T2[i], [N - i + 1, mode], [1, 2])
        return TO

    # Merging two tensors
    md1 = [i for i in range(1, N + 1) if i not in modes_1]
    s2 = np.array(T2.shape)
    if max(modes_2) > len(s2):
        s2 = np.concatenate([s2, np.ones(max(modes_2) - len(s2), dtype=int)])
    
    N2 = len(s2)
    md2 = [i for i in range(1, N2 + 1) if i not in modes_2]

    if np.any(s1[modes_1] != s2[modes_2]):
        raise ValueError('Sizes of merged modes do not match!')

    T2 = np.reshape(np.transpose(T2, axes=modes_2 + md2), (np.prod(s2[modes_2]), -1))
    T1 = np.reshape(np.transpose(T1, axes=md1 + modes_1), (-1, np.prod(s1[modes_1])))

    if md1 and md2:
        TO = np.reshape(np.dot(T1, T2), (s1[md1], s2[md2]))
    elif md1:
        TO = np.reshape(np.dot(T1, T2), s1[md1])
    elif md2:
        TO = np.reshape(np.dot(T1, T2), s2[md2])
    else:
        TO = np.dot(T1, T2)

    if len(s1) == 2 and s1[-1] == 1:  # Remove excess size if input is a column vector
        if modes_1 == [1]:
            TO = np.transpose(TO, axes=list(range(1, TO.ndim)) + [0])
    
    return TO