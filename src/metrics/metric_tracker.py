import torch
import matplotlib.pyplot as plt

class MetricTracker:
    """Metric tracker for tracking algorithm progress.

    Example:
    >>> def auc_roc(obj, **kwargs):
    >>>     labels = kwargs['labels']
    >>>     calculate_roc_auc(obj.S.ravel(), labels.ravel())
    >>>     return roc_auc
    >>>
    >>> def cardinality(obj):
    >>>     return torch.sum(obj.S != 0)
    >>>
    >>> def sparsity(obj):
    >>>     return torch.sum(obj.S == 0)/obj.S.numel()
    >>>
    >>> metric_functions = [auc_roc, cardinality]
    >>> external_inputs = {'auc_roc': {'labels': labels}}
    """
    def __init__(self, metric_functions, backend='torch', **kwargs):
        """Initializes the MetricTracker object.

        Args:
            metric_functions (list of functions): Pure functionals that take the algorithm object as input and return a scalar.
                Functions must be designed with the algorithm object in mind.
            external_inputs (dict): Dictionary of external inputs for each metric function. Keys must match the function names.
            backend (str, optional): _description_. Defaults to 'torch'.
        """
        self.backend = backend
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.metric_functions = metric_functions
        if backend == 'torch':
            self.metrics = {func.__name__: torch.tensor([], device=self.device) for func in metric_functions}
        else:
            self.metrics = {func.__name__: [] for func in metric_functions}
        self.external_inputs = kwargs.get('external_inputs', {})
        for metric_function in self.metric_functions:
            if metric_function.__name__ not in self.external_inputs:
                self.external_inputs[metric_function.__name__] = {}
        self.tracker_frequency = kwargs.get('tracker_frequency', 1)
        self.verbose = kwargs.get('verbose', 1)
        self.tb_writer = kwargs.get('tb_writer', None) # TensorBoard writer
    
    def track(self, obj):
        for func in self.metric_functions:
            stat = func(obj, **self.external_inputs[func.__name__])
            if self.verbose > 0:
                print(f'{func.__name__}: {stat:.4e}')
            
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(func.__name__, stat, obj.it)
            else:
                if self.backend == 'torch':
                    self.metrics[func.__name__] = torch.cat((self.metrics[func.__name__], stat.unsqueeze(0)))
                else:
                    self.metrics[func.__name__].append(stat)
    
    def plot(self, **kwargs):
        if self.tb_writer is not None:
            error_msg = 'The metrics are not kept in the memory and written to disk via TensorBoard writer.'
            error_msg += f'\nThe metrics can be accessed via TensorBoard at {self.tb_writer.get_logdir()}'
            raise ValueError(error_msg)
        figsize = kwargs.get('figsize', (4*len(self.metric_functions), 4))
        fig, axs = plt.subplots(1, len(self.metric_functions), figsize=figsize)
        for i, func in enumerate(self.metric_functions):
            if self.backend == 'torch':
                axs[i].plot(self.metrics[func.__name__].cpu().numpy())
            else:
                axs[i].plot(self.metrics[func.__name__])
            axs[i].set_title(func.__name__)
            axs[i].grid()
        return fig, axs