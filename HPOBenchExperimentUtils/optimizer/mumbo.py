import logging
from pathlib import Path
from typing import Union, Dict
import numpy as np

from HPOBenchExperimentUtils.optimizer.base_optimizer import SingleFidelityOptimizer
from HPOBenchExperimentUtils.core.bookkeeper import Bookkeeper
import HPOBenchExperimentUtils.utils.emukit_utils as emukit_utils
from HPOBenchExperimentUtils.utils.utils import get_mandatory_optimizer_setting, standard_rng_init
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.container.client_abstract_benchmark import AbstractBenchmarkClient

import ConfigSpace as cs
from emukit.core import ParameterSpace, InformationSourceParameter
from emukit.core.initial_designs import RandomDesign
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.optimization import MultiSourceAcquisitionOptimizer, GradientAcquisitionOptimizer
from emukit.bayesian_optimization.acquisitions.max_value_entropy_search import MUMBO
from emukit.core.acquisition import Acquisition
from emukit.multi_fidelity.models.linear_model import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.model_wrappers import GPyMultiOutputWrapper
from GPy.kern import RBF, Matern52

_log = logging.getLogger(__name__)

initial_designs = {
    "random": RandomDesign,
    "latin": LatinDesign
}

# Refer MUMBO paper: https://arxiv.org/pdf/2006.12093.pdf
# noinspection PyPep8Naming
class MultiTaskMUMBO(SingleFidelityOptimizer):
    def __init__(self, benchmark: Union[Bookkeeper, AbstractBenchmark, AbstractBenchmarkClient],
                 settings: Dict, output_dir: Path, rng: Union[int, None] = 0):

        super().__init__(benchmark, settings, output_dir, rng)

        # The benchmark defined its configuration space as a ConfigSpace.ConfigurationSpace object. This must be parsed
        # into emukit's version, an emukit.ParameterSpace object. Additionally, we need mappings between configurations
        # defined in either of the two conventions.
        self.original_space = self.benchmark.get_configuration_space()
        self.emukit_space, self.to_emu, self.to_cs = emukit_utils.generate_space_mappings(self.original_space)

        # The MUMBO acquisition has been implemented for discrete fidelity values only. Therefore, the fidelity
        # parameter needs to be appropriately handled. The paper does not say much about the modelled query costs, but
        # section B.1 of the appendix does mention very specific values for each synthetic function's costs. Thus,
        # the linear interpolation of costs for UniformFloatHyperparameter and UniformIntegerHyperparameter are simply
        # assumptions made in order to keep the cost estimates comparable to those made our dragonfly code. It should
        # be noted that MultiTaskMUMBO was used only for the synthetic objective functions in the paper. Other
        # experiments were performed by combining MUMBO with a different surrogate model, such as FABOLAS for MNIST.
        if isinstance(self.main_fidelity, cs.UniformFloatHyperparameter):
            num_fidelity_values = get_mandatory_optimizer_setting(
                settings, "num_fidelity_values",
                err_msg="When using a continuous fidelity parameter, number of discrete fidelity levels must be "
                        "specified in the parameter 'num_fidelity_values'")
            _log.debug("Discretizing the main fidelity %s for use with MUMBO into %d fidelity levels." %
                       (self.main_fidelity.name, num_fidelity_values))
            self.info_sources = np.linspace(self.min_budget, self.max_budget, num_fidelity_values)

        elif isinstance(self.main_fidelity, cs.OrdinalHyperparameter):
            self.info_sources = np.asarray(self.main_fidelity.sequence)
        elif isinstance(self.main_fidelity, cs.CategoricalHyperparameter):
            self.info_sources = np.asarray(self.main_fidelity.choices)
        elif isinstance(self.main_fidelity, cs.UniformIntegerHyperparameter):
            num_fidelity_values = get_mandatory_optimizer_setting(
                settings, "num_fidelity_values",
                err_msg="When using a continuous fidelity parameter, number of discrete fidelity levels must be "
                        "specified in the parameter 'num_fidelity_values'")
            self.info_sources = np.floor(np.linspace(self.min_budget, self.max_budget, num_fidelity_values,
                                                     endpoint=True)).astype(int)

        else:
            raise NotImplementedError(f"Handling fidelity parameters of type {type(self.main_fidelity)} has not yet "
                                      f"been implemented for MUMBO.")

        # InformationSourceParameter is a sub-class of a DiscreteParameter and must be used on account of internal
        # checks in the MUMBO code for its type.
        self.emukit_fidelity = InformationSourceParameter(self.info_sources.shape[0])
        self.fidelity_emukit_to_cs = lambda s: {self.main_fidelity.name: self.info_sources[s]}

        def wrapper(x: np.ndarray):
            """
            Remember that for MUMBO the search space is augmented with a discrete parameter for the fidelity index.
            This wrapper simply parses the configuration and fidelity and sets them up for calling the underlying
            benchmark objective function. Emukit requires this function to accept 2D inputs, with individual
            configurations aligned along axis 0 and the various components of each configuration along axis 1. If a
            batch_size of greater than 1 is specified, the BO loop will start calling the wrapper with more than 1
            input configuration per iteration.
            """

            _log.debug("Benchmark wrapper received input %s." % str(x))
            if x.ndim == 1:
                x = np.expand_dims(x, axis=0)
            results = []
            for i in range(x.shape[0]):
                _log.debug("Extracted configuration: %s" % str(x[i, :-1]))
                _log.debug("Extracted fidelity value: %s" % str(self.info_sources[int(x[i, -1])]))
                fidelity = self.fidelity_emukit_to_cs(int(x[i, -1]))
                config = cs.Configuration(self.original_space, values=self.to_cs(x[i, :-1]))
                res = benchmark.objective_function(config, fidelity=fidelity)
                _log.debug("Benchmark evaluation results: %s" % str(res))
                results.append([res["function_value"]])
            results = np.asarray(results)
            # Assume that the "function_value" results are always scalars, therefore it makes sense to place all
            # individual values along the 0th axis in case the final result is not 2D for some reason.
            return results if results.ndim == 2 else np.expand_dims(results, axis=1)

        self.benchmark_caller = wrapper

        self.init_samples_per_dim = get_mandatory_optimizer_setting(settings, "init_samples_per_dim")
        self.initial_design_type = initial_designs[str(settings.get("initial_design", "random")).lower()]
        self.gp_settings = {
            "n_optimization_restarts": get_mandatory_optimizer_setting(settings, "n_optimization_restarts"),
            "update_interval": get_mandatory_optimizer_setting(settings, "update_interval"),
            "batch_size": get_mandatory_optimizer_setting(settings, "batch_size"),
            "kernel": str(get_mandatory_optimizer_setting(settings, "kernel")).lower()
        }
        self.mumbo_settings = {
            "num_mc_samples": get_mandatory_optimizer_setting(settings, "num_mc_samples"),
            "grid_size": get_mandatory_optimizer_setting(settings, "grid_size")
        }

        _log.info("Finished reading all settings for Multi-Task GP with MUMBO acquisition.")

    def _setup_model(self):
        """
        This is mostly boilerplate code required to setup a BO loop that uses a GP as a surrogate and a MUMBO
        acquisition function. Ref. MUMBO example code:
        https://github.com/EmuKit/emukit/blob/master/notebooks/Emukit-tutorial-multi-fidelity-MUMBO-Example.ipynb

        Where no references to the paper are made, it is to be assumed that the code was adapted directly from the
        above example code but no specific references were found in the paper.
        """

        # Generate warm-start samples. Same as the MUMBO example code. RandomDesign, as mentioned in B.1 of the
        # appendix of the MUMBO paper.
        augmented_space = ParameterSpace([*self.emukit_space.parameters, self.emukit_fidelity])
        initial_design = self.initial_design_type(self.emukit_space)

        # For n samples per dim, input space dimensionality D, we generate nxD samples.
        # As per Section B.1 of the appendix of the paper, we need to evaluate each of these samples on every available
        # fidelity value, but that is wasteful for non-synthetic benchmarks, so we opt for uniformly distributing the
        # fidelity values across the sampled initial configurations. It is assumed that the number of unique fidelity
        # values is less than the number of sampled configurations.
        n_init = self.init_samples_per_dim * self.emukit_space.dimensionality
        X_init = initial_design.get_samples(n_init)
        n_reps = n_init // self.info_sources.shape[0] + 1
        fmin, fmax = self.emukit_fidelity.bounds[0]
        sample_fidelities = np.tile(np.arange(fmin, fmax + 1), n_reps)[:n_init].reshape(-1, 1)
        X_init = np.concatenate((X_init, sample_fidelities), axis=1)
        Y_init = self.benchmark_caller(X_init)
        _log.debug("Generated %d warm-start samples, each evaluated on %d fidelity values, for a total of %d initial "
                   "evaluations using initial design %s." % (n_init, (fmax - fmin + 1), Y_init.shape[0],
                                                             self.initial_design_type.__name__))

        # Setup kernels for the GP. No specific references found in the paper except a brief mention in Eq. 3, section
        # 2.3. Using the code provided in the example notebook.
        n_fidelity_vals = self.info_sources.shape[0]
        fidelity_kernels = []
        recognized_kernels = {
            "rbf": RBF,
            "matern52": Matern52
        }
        try:
            kernel_type = recognized_kernels[self.gp_settings["kernel"]]
        except AttributeError:
            raise ValueError("Unrecognized value %s for 'kernel' in optimizer settings. Must be one of "
                             "['rbf', 'matern52']." % self.gp_settings["kernel"])
        for _ in range(n_fidelity_vals):
            kernel = kernel_type(self.emukit_space.dimensionality)
            kernel.lengthscale.constrain_bounded(0.01, 0.5)
            fidelity_kernels.append(kernel)

        multi_fidelity_kernel = LinearMultiFidelityKernel(fidelity_kernels)
        _log.debug("Mixture of %d %s kernels initialized." % (len(fidelity_kernels), kernel_type.__name__))

        # Initialize the GP for MTBO. Same as the MUMBO example code.
        gpy_model = GPyLinearMultiFidelityModel(X=X_init, Y=Y_init, kernel=multi_fidelity_kernel,
                                                n_fidelities=n_fidelity_vals)

        gpy_model.likelihood.Gaussian_noise.fix(0.1)
        for i in range(1, len(self.emukit_fidelity.bounds)):
            getattr(gpy_model.likelihood, f"Gaussian_noise_{i}").fix(0.1)

        # Emukit wrapper for GPy.core.GP. Same as the MUMBO example code.
        model = GPyMultiOutputWrapper(gpy_model, n_outputs=2,
                                      n_optimization_restarts=self.gp_settings["n_optimization_restarts"],
                                      verbose_optimization=False)
        model.optimize()
        _log.debug("GP initialized.")

        # Setup the acquisition function. Same as the MUMBO example code.
        cost_acquisition = Cost(np.linspace(start=1. / n_fidelity_vals, stop=1.0, num=n_fidelity_vals))
        mumbo_acquisition = MUMBO(model, augmented_space, num_samples=self.mumbo_settings["num_mc_samples"],
                                  grid_size=self.mumbo_settings["grid_size"]) / cost_acquisition
        acquisition_optimizer = MultiSourceAcquisitionOptimizer(GradientAcquisitionOptimizer(augmented_space),
                                                                space=augmented_space)

        # Setup the BO loop. Same as the MUMBO example code.
        self.optimizer = BayesianOptimizationLoop(space=augmented_space, model=model, acquisition=mumbo_acquisition,
                                                  update_interval=self.gp_settings["update_interval"],
                                                  batch_size=self.gp_settings["batch_size"],
                                                  acquisition_optimizer=acquisition_optimizer)

        # These are hooks that help us record the trajectory for an information theoretic acquisition function, which
        # cannot be handled otherwise by the Bookkeeper.
        self.optimizer.loop_start_event.append(emukit_utils.get_init_trajectory_hook(self.output_dir))
        self.optimizer.iteration_end_event.append(emukit_utils.get_trajectory_hook(self.output_dir, self.to_cs))
        _log.debug("Multi-Task GP with MUMBO acquisition ready to run.")

    def setup(self):
        pass

    def run(self) -> Path:
        _log.info("Starting GP optimizer with MUMBO acquisition.")

        # emukit does not expose any interface for setting a random seed any other way, so we reset the global seed here
        # Generating a new random number from the seed ensures that, for compatible versions of the numpy.random module,
        # the seeds remain predictable while still handling seed=None in a consistent manner.
        np.random.seed(standard_rng_init(self.rng).randint(0, 1_000_000))
        self._setup_model()
        self.optimizer.run_loop(user_function=self.benchmark_caller,
                                stopping_condition=emukit_utils.InfiniteStoppingCondition())
        _log.info("GP optimizer with MUMBO acquisition finished.")
        return self.output_dir


# Define cost of different fidelities as acquisition function, taken directly from the MUMBO example code
# Source: https://github.com/EmuKit/emukit/blob/master/notebooks/Emukit-tutorial-multi-fidelity-MUMBO-Example.ipynb
class Cost(Acquisition):
    def __init__(self, costs):
        self.costs = costs

    def evaluate(self, x):
        fidelity_index = x[:, -1].astype(int)
        x_cost = np.array([self.costs[i] for i in fidelity_index])
        return x_cost[:, None]

    @property
    def has_gradients(self):
        return True

    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)
