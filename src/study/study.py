from pprint import pprint
import os, sys

from copy import deepcopy
from collections import defaultdict
from abc import ABC, abstractmethod

import wandb
# import ray
from dask.distributed import Client, as_completed
import numpy as np
import pandas as pd
from tqdm import tqdm


class StudyBaseClass(ABC):
    def __init__(self, project_name, api_key, exp_config, exp_prefix, group_name, tags, **kwargs):
        self.project_name = project_name
        self.exp_config = exp_config
        self.group_name = group_name
        self.tags = tags
        self.all_results = defaultdict(list)
        wandb.login(key=api_key, verify=True)
        self.exp_prefix = exp_prefix
        self.run_kwargs = kwargs
        self.num_active_remote = 0
        self.client = kwargs.get('client', Client(n_workers=kwargs.get('n_workers', 5)))
    
    def save_results(self, path):
        result = pd.DataFrame(self.all_results)
        result.to_csv(path)
        return result


    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def model_runner(self):
        pass

    @abstractmethod
    def calculate_metrics(self):
        pass

    def __del__(self):
        self.client.shutdown()