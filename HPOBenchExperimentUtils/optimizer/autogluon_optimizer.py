import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Union
import shutil
import time

try:
    import autogluon.core as ag
    from autogluon.core.scheduler.hyperband import _get_rung_levels
    import autogluon.core.version as v
    assert v.__version__ == "0.2.0"
except:
    raise ValueError("""Autogluon is not installed or the wrong version is installed, please run:\n
                        python3 -m pip install --upgrade pip
                        python3 -m pip install --upgrade setuptools
                        python3 -m pip install --upgrade "mxnet<2.0.0"
                        python3 -m pip install autogluon==0.2.0""")

from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    UniformIntegerHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper

_log = logging.getLogger(__name__)


def nothing(self):
    print("Do not stop container")


def decorate_func(f):
    def _decorate_func(*args, **kwargs):
        return f(*args, **kwargs)
    return _decorate_func


def _obj_fct(args, reporter):
    run_id = SingleFidelityOptimizer._id_generator()

    # Build configuration
    config = dict()
    for h in args.cs.get_hyperparameters():
        if isinstance(h, UniformIntegerHyperparameter):
            config[h.name] = int(np.rint(args[h.name]))
        elif isinstance(h, OrdinalHyperparameter):
            config[h.name] = h.sequence[int(args[h.name])]
        else:
            config[h.name] = args[h.name]

    # Iterate only over fidelities that are interesting to the optimizer
    for epoch in args.valid_budgets:
        fidelity = {args.main_fidelity.name: epoch}
        res = args.benchmark.objective_function(configuration=config, configuration_id=run_id, fidelity=fidelity,
                                                **args.settings_for_sending)
        # Autogluon maximizes, HPOBench returns something to be minimized
        acc = -res['function_value']
        eval_time = res['cost']
        reporter(epoch=epoch,
                 performance=acc,
                 eval_time=eval_time,
                 time_step=time.time(), **config)


class AutogluonOptimizer(SingleFidelityOptimizer):
    """
    This class implements the HNAS optimization algorithm implemented in autogluon as described here
    https://github.com/awslabs/autogluon/pull/797
    """
    def __init__(self, benchmark: Bookkeeper,
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):
        super().__init__(benchmark, settings, output_dir, rng)
        # Setup can be done here or in run()
        _log.info('Successfully initialized')

        # Set up search space and rung_levels
        self.ag_space = self.get_ag_space(self.cs)
        if isinstance(self.main_fidelity, UniformFloatHyperparameter):
            # This would only be the case for dataset fraction as a fidelity and we don't run this
            # optimizer on these
            raise ValueError("This optimizer doesn't handle float fidelities")

        # Get time limit, so we know how long we should run the optimization
        self.time_limit = self.benchmark.resource_manager.limits.time_limit_in_s
        if self.benchmark.is_surrogate:
            # We limit this and cut the runhistory afterwards in case there are too many evaluations
            self.time_limit = 342000  # 95h; was 60*60*24*4=96h before

        # Get default settings as set in the PR
        self.reduction_factor = self.settings["reduction_factor"]  # 3 by default

        self.searcher = self.settings.get("searcher", "bayesopt")
        assert self.searcher in ("bayesopt", "random"), self.searcher

        self.scheduler = self.settings.get("scheduler", "hyperband_promotion")
        assert self.scheduler in ["hyperband_promotion", "hyperband_stopping"]

        self.hyperband_type = None
        if self.scheduler == "hyperband_stopping":
            self.hyperband_type = "stopping"
        elif self.scheduler == "hyperband_promotion":  # default
            self.hyperband_type = "promotion"

        # Some fixed settings we don't touch
        # setting the number of brackets to 1 means that we run effectively successive halving
        self.brackets = int(self.settings.get("brackets", 1))
        self.first_is_default = False
        self.num_gpus = 0
        self.num_cpus = 1

        self.rung_levels = self._get_rung_levels()

    def setup(self):
        pass

    def _get_rung_levels(self):
        # According to line 820 in hyperband.py
        rung_levels = _get_rung_levels(rung_levels=None, grace_period=self.min_budget,
                                       reduction_factor=self.reduction_factor,
                                       max_t=self.max_budget)
        rung_levels = rung_levels + [self.max_budget]
        return rung_levels

    @staticmethod
    def get_ag_space(cs):
        d = dict()
        for h in cs.get_hyperparameters():
            if isinstance(h, UniformFloatHyperparameter):
                d[h.name] = ag.space.Real(lower=h.lower, upper=h.upper, log=h.log)
            elif isinstance(h, UniformIntegerHyperparameter):
                if h.log:
                    # autogluon does not support int with log=True, make this a Real and round
                    d[h.name] = ag.space.Real(lower=h.lower, upper=h.upper, log=True)
                else:
                    d[h.name] = ag.space.Int(lower=h.lower, upper=h.upper)
            elif isinstance(h, CategoricalHyperparameter):
                d[h.name] = ag.space.Categorical(*h.choices)
            elif isinstance(h, OrdinalHyperparameter):
                # autogluon does not support ordinals, make this an Int
                d[h.name] = ag.space.Int(lower=0, upper=len(h.sequence)-1)
            else:
                raise ValueError("Cannot handle %s" % h.name)
        return d

    def make_benchmark(self):
        return ag.args(**self.ag_space,
                       benchmark=self.benchmark,
                       settings_for_sending=self.settings_for_sending,
                       epochs=self.max_budget,
                       valid_budgets=self.rung_levels,
                       cs=self.cs,
                       main_fidelity=self.main_fidelity)(_obj_fct)

    def run(self):
        """ Execute the optimization run. Return the path where the results are stored. """
        callback = None
        scheduler = ag.scheduler.HyperbandScheduler(self.make_benchmark(),
                                                    resource={'num_cpus': self.num_cpus,
                                                              'num_gpus': self.num_gpus},
                                                    # Autogluon runs until it either reaches num_trials or time_out
                                                    # This is None for most benchmarks
                                                    num_trials=self.benchmark.resource_manager.limits.tae_limit,
                                                    time_out=self.time_limit,
                                                    reward_attr='performance',
                                                    time_attr='epoch',
                                                    brackets=self.brackets,
                                                    checkpoint=None,
                                                    training_history_callback=callback,
                                                    training_history_callback_delta_secs=1,
                                                    searcher=self.searcher,
                                                    # Defines searcher for new configurations
                                                    # training_history_callback_delta_secs=args.store_results_period,
                                                    reduction_factor=self.reduction_factor,
                                                    type=self.hyperband_type,
                                                    # TODO Maybe we want to manually set the rung levels to avoid
                                                    # having too many levels. We limit these to 5 for other optimizers
                                                    search_options={'random_seed': self.rng,
                                                                    'first_is_default': self.first_is_default},
                                                    grace_period=self.min_budget)
        # Just to make sure that our assumption on the run levels is correct
        assert scheduler.terminator.rung_levels == self.rung_levels[:-1], \
            (scheduler.terminator.rung_levels, self.rung_levels[:-1])
        scheduler.run()
        scheduler.join_jobs()
