import numpy as np


def contaminate_signal(X, noise_rate=10, noise_type='AWGN', M=1, obs_ratio=1):
    ''' Contaminates data with noise and chooses random elements as missing.
    Parameters:
        X: np.array(), double
            Original data tensor.
        noise_rate: float
            For 'AWGN', target (-)SNR (ratio of noise/signal). 
            For others, it is the ratio of corrupted elements cardinality to tensor size.
        noise_type: string
            Type of noise to be added to data. Default: 'AWGN'
                Possible types:
                    'AWGN' - Additive White Gaussian Noise applied
                    to all elements.
                    'unif' - Additive IID uniform noise with in the
                    range (-M,M) applied to the sparse elements.
                    'bernoulli' - Additive IID bernoulli distribution
                    noise with possible values {-M,M}. 
        obs_ratio: float,
            Ratio of the cardinality of the observed elements in the tensor.
            Should be in (0,1].
        M: float,
            Magnitude prameter for sparse noise types.
        
    Outputs:
        Y: Masked Array. np.ma()
            Noisy tensor with missing elements.
    '''

    # Generate noise
    sizes = X.shape
    rng = np.random.default_rng()
    Y = X.copy()
    if noise_type == 'AWGN':
        signal_power = np.linalg.norm(X)**2/X.size
        signal_dB = 10 * np.log10(signal_power)
        noise_db = signal_dB + noise_rate 
        noise_power = 10 ** (noise_db / 10)
        noise = np.sqrt(noise_power)*np.random.standard_normal(sizes)
        Y = Y + noise
    else:
        noise_cardinality= np.floor(noise_rate*X.size).astype(int) 
        if noise_rate <1: # If the noise is not applied to all elements
            # Determine the random indices to of the corrupt elements
            vec_ind = rng.permutation(X.size)[:noise_cardinality]
    
        if noise_type == 'gross': # Construct the noise vector
            noise = rng.uniform(low=X.min(), high=X.max(), size=noise_cardinality)
        elif noise_type == 'unif':
            noise = rng.uniform(low=-M, high=M, size=noise_cardinality)
        elif noise_type == 'bernoulli':
            noise = (rng.binomial(1,0.5, size=noise_cardinality)-0.5)*2*M
        
        if noise_rate != 1: # Add the noise vector
            Y.flat[vec_ind] = Y.flat[vec_ind] + noise
        else:
            Y.flat[:] = Y.flat[:] + noise

        
    # Create mask for unobserved elements.
    if obs_ratio !=1:
        perm = rng.permutation(X.size) 
        obs_cardlty = np.floor(obs_ratio*X.size).astype(int)  
        obs_idx = perm[:obs_cardlty]                    # The indexes of the observed elements
        vec_mask = np.ones(X.size, dtype='bool')        # Start with all masked vector mask
        vec_mask.flat[obs_idx]=False                    # Set the observed indices False
           
    else:
        vec_mask = np.zeros(X.size, dtype='bool')

    mask = vec_mask.reshape(sizes)
    return np.ma.array(Y, mask=mask)