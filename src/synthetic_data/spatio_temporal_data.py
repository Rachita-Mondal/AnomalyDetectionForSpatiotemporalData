import numpy as np
from src.util.t2m import t2m
from src.util.m2t import m2t

class STData(object):
    """Wrapper object for Tensorized Spatio-temporal data with metadata labels and methods.

    """
    def __init__(self, M, resolution):
        pass

    def gen_similar_st_data(self, std, mode='week'):
        """Generate synthetic spatio-temporal data similar to self.

        Generate random data that is similar the given real spatio-temporal data.
        This is done by calculating the mean values of the temporal data for each week
        and multiplying it with gaussian noise with low standard deviation for each location.
        
        Args:
            std (_type_): _description_
            mode (str, optional): _description_. Defaults to 'week'.
        """
        # return synth_st_data
        pass