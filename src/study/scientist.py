# Need to do type hinting for the classes and methods

from abc import ABC, abstractmethod

# from matplotlib import pyplot as plt
# import pandas as pd
import optuna
# from tqdm import tqdm


class ExperimentBase(ABC):
    def __init__(self,
                 exp_name,
                 ExperimentRunner,
                 variables_of_interest=None,
                 optuna_storage='sqlite:///optuna.db'):
        """Initialize the experiment for the optuna study.

        Parameters:
        ----------
        exp_name (str): Experiment name
        ExperimentRunner (ExperimentTrialRunnerBase): Experiment runner class that will be used to run
            the experiment trials.
        directions (str): Direction of the dependent variables. 'maximize' or 'minimize' 
        optuna_storage (str, optional): _description_. Defaults to 'sqlite:///optuna.db'.
        """
        self.exp_name = exp_name
        self.experiment_runner = ExperimentRunner
        self.optuna_storage = optuna_storage
        self._study_name = ExperimentRunner.study_name_parser(exp_name, variables_of_interest)
        
        if len(ExperimentRunner.dependent_vars) > 1:
            self.study = optuna.create_study(directions=ExperimentRunner.directions,
                                          storage=optuna_storage,
                                          study_name=self.study_name,
                                          load_if_exists=True)
        else:
            self.study = optuna.create_study(direction=ExperimentRunner.directions[0],
                                          storage=optuna_storage,
                                          study_name=self.study_name,
                                          load_if_exists=True)
        
        self.study.set_metric_names(ExperimentRunner.dependent_vars)
        self.study.set_user_attr('Experiment name', exp_name)
        self.study.set_user_attr('Independent Variables', ExperimentRunner.independent_vars)
        self.study.set_user_attr('Dependent Variables', ExperimentRunner.dependent_vars)
        self.study.set_user_attr('Control Variables', ExperimentRunner.control_vars)
        for model_name, model_runner in ExperimentRunner.model_runners.items():
            self.study.set_user_attr(f'{model_name} Independent Variables', model_runner.model_independent_vars)
            self.study.set_user_attr(f'{model_name} Control Variables', model_runner.model_control_vars)

    @abstractmethod
    def execute(self, n_repeat=1, devices=['cuda:{i}' for i in range(4)]):
        """Execute the experiment"""
        pass
    

    def delete_study(self):
        """Delete the experiment records from the optuna storage."""
        optuna.study.delete_study(study_name=self.study_name, storage=self.optuna_storage)
    
    @property
    def study_name(self):
        return self._study_name

    @property
    @abstractmethod
    def directions(self):
        pass


class ExperimentTrialRunnerBase(ABC):
    def __init__(self,
                 independent_vars,
                 control_vars,
                 dependent_vars,
                 runtime_misc,
                 model_runners,
                 directions=None,
                 ):
        """Initialize the experiment trial runner for optuna study to call
        
        Parameters:
        ----------
        independent_vars (dict of dict): dict of dict containing
            the name, type and range of the common independent variables.
            If the independent variable is a model in comparison,
            the type should be 'categorical' and the choices should
            be specified.
        control_vars (dict): The common control control variables for
            the experiment. The control variables should be specified
            as a dict with the key as the variable name and the value
            as the variable value.
        dependent_vars (list of str): Names of the dependent variables
            that will be recorded. Example: ['accuracy', 'loss']
        runtime_misc (list of str): Names of the runtime variables
            that must be passed to the enque_trial method of the study
            as user attributes. Example: ['device']
        model_runners (dict): Dictionary containing the model wrappers
            that will be used to run the experiment. The keys should be
            the model names and the values should be the model wrapper
            classes.

        Notes:
        -----
        1. If an independent variable is used by a model, it should be
            specified in the `model_independent_vars` and not in the
            `independent_vars`
        """
        self.independent_vars = independent_vars
        self.control_vars = control_vars
        self.dependent_vars = dependent_vars
        self.runtime_misc = runtime_misc
        self.model_runners = model_runners
        self._directions = directions
        

    def __call__(self, trial):
        """Run the experiment trial for the optuna study

        Will be called by the optuna study to run the experiment trial
        Parameters:
        ----------
        trial (optuna.trial.Trial): The trial object that will be
            used to get the independent variables and miscallenous runtime 
            parameters to run the experiment and record the dependent
            variables.

        Returns:
        -------
        objectives: Returns the dependent variables as a tuple
        """
        variables = {}
        for name, var_dict in self.independent_vars.items():
            var_type = var_dict['type']
            if name == 'model':
                model_name = trial.suggest_categorical(name, var_dict['choices'])
            else:
                if var_type == 'int':
                    variables[name] = trial.suggest_int(name, var_dict['low'], var_dict['high'],
                                                        step=var_dict.get('step', 1),
                                                        log=var_dict.get('log', False))
                elif var_type == 'float':
                    variables[name] = trial.suggest_float(name, var_dict['low'], var_dict['high'],
                                                          step=var_dict.get('step', 1),
                                                          log=var_dict.get('log', False))
                elif var_type == 'categorical':
                    variables[name] = trial.suggest_categorical(name, var_dict['choices'])
                else:
                    raise ValueError('Invalid independent variable type')

        for var in self.runtime_misc:
            if var in variables.keys():
                raise ValueError(f'Runtime variable {var} is already present in the independent variables')
            variables[var] = trial.user_attrs[var]

        if model_name is None:
            try:
                model_name = self.control_vars['model']
            except KeyError:
                raise ValueError('Model name not provided in the control variables or as an independent variable')
        
        variables = {**variables, **self.control_vars}

        data = self._get_data(variables)

        results = self.model_runners[model_name].run(trial, data)

        metrics = self._calculate_metrics(data, results)
        objectives = tuple([metrics.pop(dep_var) for i, dep_var in enumerate(self.dependent_vars)])
        
        for key, v in metrics.items():
            trial.set_user_attr(key, v) # Set the remaining metrics as user attributes
        return tuple(objectives)

    @abstractmethod
    def _get_data(self, variables):
        """Get the data for the experiment based on the variables"""
        pass

    @abstractmethod
    def _calculate_metrics(self, data, result):
        """Calculate the metrics based on data and the result"""
        pass

    def study_name_parser(self, exp_name, variables_of_interest=None):
        """Construct a meaningful study name from the experiment name and control variables to save the study
        
        Parameters:
        ----------
        exp_name (str): The experiment name
        variables_of_interest (list of str, optional): List of variables that should be included in the study name.
            Defaults to None.
        """
        study_name = f'{exp_name}_'
        if variables_of_interest is None:
            variables_of_interest =[]
        
        for key in variables_of_interest:
            value = None
            if key in self.control_vars.keys():
                value = self.control_vars[key]
            elif key in self.independent_vars.keys():
                value = self.independent_vars[key]
            else:
                for model_name, model_runner in self.model_runners.items():
                    if key in model_runner.model_independent_vars.keys():
                        value = model_runner.model_independent_vars[key]
                        break
            if value is not None:
                study_name += f'{key}_{value}_'
        study_name = study_name[:-1]
        return study_name
    
    @property
    def directions(self):
        """Get or make up the directions of the dependent variables

        Returns:
        -------
            (list of str): The directions of the dependent variables
        """
        if self._directions is None:
            directions = []
            for dep_var in self.dependent_vars:
                if dep_var.startswith('loss'):
                    directions.append('minimize')
                elif dep_var.startswith('accuracy') or dep_var.startswith('f1'):
                    directions.append('maximize')
                elif dep_var.startswith('precision') or dep_var.startswith('recall'):
                    directions.append('maximize')
                elif dep_var.startswith('time'):
                    directions.append('minimize')
                elif dep_var.startswith('AUC') or dep_var.startswith('ROC'):
                    directions.append('maximize')
                elif dep_var.endswith('error'):
                    directions.append('minimize')
                elif dep_var.endswith('score'):
                    directions.append('maximize')
                elif dep_var.endswith('distance'):
                    directions.append('minimize')
                elif dep_var.endswith('err'):
                    directions.append('minimize')
                elif dep_var.startswith('MSE') or dep_var.startswith('RMSE') or dep_var.startswith('MAE'):
                    directions.append('minimize')
                else: # Default is to maximize.
                    directions.append('maximize')
            self._directions = directions
        else:
            directions = self._directions
            if len(directions) != len(self.dependent_vars):
                raise ValueError('Number of directions must be equal to the number of dependent variables')
        return directions


class ModelRunnerBase(ABC):
    def __init__(self,
                 model_control_vars,
                 model_independent_vars,
                 model_runtime_misc):
        """Initialize the model wrapper with the variables
        
        ModelRunner classes are typically defined for each experiment 
        Parameters:
        ----------
        model_independent_vars (dict of dict): dict of dict containing
            the name, type and range of the independent variables
            for model(s) to run in the experiment. The key should be
            the model name and the value should be the dict of dict
            containing the name, type and range of the independent
            variables.
        model_control_vars (dict of dict): Model specific control
            variables for the experiment. The key should be the model
            name and the value should be the dict of dict containing
            the control variables for the model.
        model_runtime_misc (dict of list of str): Model specific runtime
            variables for the experiment. The key should be the model
            name and the value should be the list of runtime variables
            that must be passed to the enque_trial method of the study
            which will not be recorded.
        """
        self.model_control_vars = model_control_vars
        self.model_independent_vars = model_independent_vars
        self.model_runtime_misc = model_runtime_misc

    
    def run(self, trial, data):
        """Run the model with the data and return the result
        
        `run` method is called by the ExperimentTrialRunnerBase class and
        the run method can be customized for the experiment in design.
        Parameters:
        ----------
        trial (optuna.trial.Trial): The trial object that will be
            used to get the independent variables and miscallenous runtime 
            parameters to run the experiment and record the data dependent
            variables.
        data (dict): The data that will be used to run the model on

        Returns:
        -------
        result (dict): The results of the model run
        """
        variables = {}
        for name, var_dict in self.model_independent_vars.items():
            var_type = var_dict['type']
            if var_type == 'int':
                variables[name] = trial.suggest_int(name, var_dict['low'], var_dict['high'],
                                                    step=var_dict.get('step', 1),
                                                    log=var_dict.get('log', False))
            elif var_type == 'float':
                variables[name] = trial.suggest_float(name, var_dict['low'], var_dict['high'],
                                                      step=var_dict.get('step', 1),
                                                      log=var_dict.get('log', False))
            elif var_type == 'categorical':
                variables[name] = trial.suggest_categorical(name, var_dict['choices'])
            else:
                raise ValueError('Invalid independent variable type')

        for var in self.model_runtime_misc:
            if var in variables.keys():
                raise ValueError(f'Runtime variable {var} is already present in the independent variables')
            variables[var] = trial.user_attrs[var]

        for key in self.model_control_vars.keys():
            if key in variables.keys():
                raise ValueError(f'Control variable {key} is already present in the independent variables')

        variables = {**variables, **self.model_control_vars}
        result = self._run(data, variables)
        return result

    
    @abstractmethod
    def _run(self, data, variables):
        """Run the model with the data and return the result
        
        `_run` method is called by the `run` method or the `_run` method
        and must be customized for the model in consideration.
        Parameters:
        ----------
        data (dict): The data that will be used to run the model on
        variables (dict): The variables that will be used to run the model on

        Returns:
        -------
        result (dict): The results of the model run
        """
        pass
    
    @property
    @abstractmethod
    def model_variable_names(self):
        """Return the model variables"""
        pass

    @property
    @abstractmethod
    def name(self):
        """Return the model name"""
        pass
