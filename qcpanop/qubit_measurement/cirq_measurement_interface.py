"""
Interface that will use the appriopriate method to simulate PauliStrinngs
given the stopping conditions

Sampler==Simulator and repetitions = inf
Use simulator
Sampler==Simulator and variance = 0
Use simulator
Sampler==Simulator and repetitions = finite
Use  sample
Sampler==Simulator and variance  = finite
use sample
"""
from typing import Iterable, List, Optional, Union

import numpy as np

import cirq
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
    BitstringAccumulator,
    ObservableMeasuredResult,
    flatten_grouped_results,
)
from cirq.work.observable_measurement import (StoppingCriteria,
                                              CheckpointFileOptions,
                                              RepetitionsStoppingCriteria,
                                              VarianceStoppingCriteria)


class InputError(Exception):
    """Configuration error for DataCollectorFactory"""
    pass


def _sim_measure_observables(circuit: 'cirq.AbstractCircuit',
                             observables: Iterable['cirq.PauliString'],
                             sampler: Union['cirq.Simulator', 'cirq.Sampler'],
                             stopping_criteria: StoppingCriteria,
                             *,
                             readout_symmetrization: bool = False,
                             circuit_sweep: Optional['cirq.Sweepable'] = None,
                             grouper: Union[
                                 str, GROUPER_T] = group_settings_greedy,
                             readout_calibrations: Optional[
                                 BitstringAccumulator] = None,
                             checkpoint: CheckpointFileOptions = CheckpointFileOptions(),
                             ) -> List[ObservableMeasuredResult]:
    """Measure a collection of PauliString observables for a state prepared by a Circuit.

    If you need more control over the process, please see `measure_grouped_settings` for a
    lower-level API. If you would like your results returned as a pandas DataFrame,
    please see `measure_observables_df`.

    Args:
        circuit: The circuit used to prepare the state to measure. This can contain parameters,
            in which case you should also specify `circuit_sweep`.
        observables: A collection of PauliString observables to measure. These will be grouped
            into simultaneously-measurable groups, see `grouper` argument.
        sampler: The sampler.
        stopping_criteria: A StoppingCriteria object to indicate how precisely to sample
            measurements for estimating observables.
        readout_symmetrization: IGNORED INPUT. If set to True, each run will be split into two: one normal and
            one where a bit flip is incorporated prior to measurement. In the latter case, the
            measured bit will be flipped back classically and accumulated together. This causes
            readout error to appear symmetric, p(0|0) = p(1|1).
        circuit_sweep: IGNORED INPUT. Additional parameter sweeps for parameters contained in `circuit`. The
            total sweep is the product of the circuit sweep with parameter settings for the
            single-qubit basis-change rotations.
        grouper: IGNORED INPUT. Either "greedy" or a function that groups lists of `InitObsSetting`. See the
            documentation for the `grouped_settings` argument of `measure_grouped_settings` for
            full details.
        readout_calibrations: IGNORED INPUT. The result of `calibrate_readout_error`.
        checkpoint: IGNORED INPUT. Options to set up optional checkpointing of intermediate data for each
            iteration of the sampling loop. See the documentation for `CheckpointFileOptions` for
            more. Load in these results with `cirq.read_json`.

    Returns:
        A list of ObservableMeasuredResult; one for each input PauliString.
    """
    if isinstance(stopping_criteria, RepetitionsStoppingCriteria):
        if stopping_criteria.total_repetitions != np.inf:
            raise TypeError("repetitions is {}. use cirq measure obs".format(
                stopping_criteria.total_repetitions))
    elif isinstance(stopping_criteria, VarianceStoppingCriteria):
        if not np.isclose(stopping_criteria.variance_bound, 0, atol=1.0E-14):
            raise TypeError("Variance bound is {}. use cirq measure obs".format(
                stopping_criteria.variance_bound))

    sim_results = sampler.simulate(circuit)
    results = []
    for term in observables:
        termval = term.expectation_from_state_vector(
            sim_results.final_state_vector, qubit_map=sim_results.qubit_map,
            check_preconditions=False)

        assert np.isclose(termval.imag, 0, atol=1e-4), termval.imag
        termval = termval.real

        results.append(ObservableMeasuredResult(
            mean=termval,
            variance=0,
            repetitions=np.inf,
            circuit_params=dict(),
            setting=None
        ))
    return results


class DataCollectorFactory:

    def __init__(self, repetitions=None, variance=None, sampler=None):
        """
        DataCollector object. Takes parameters of sampling or simulating from
        the user and then dispatches to the appropriate configuration.

        Sampler==Simulator and repetitions = inf
        Use simulator
        Sampler==Simulator and variance = 0
        Use simulator
        Sampler==Simulator and repetitions = finite
        Use  sample
        Sampler==Simulator and variance  = finite
        use sample

        """
        self.repetitions = repetitions
        self.variance = variance
        self.sampler = sampler

    def get_measure_observables(self):  # add type annotiation
        if self.repetitions is None and self.variance is None and isinstance(self.sampler, cirq.Simulator):
            # this should be the exact calculation
            self.stopping_criteria = RepetitionsStoppingCriteria(total_repetitions=np.inf)
            return _sim_measure_observables
        elif self.repetitions is not None and self.variance is not None:
            raise InputError("repetitions or variance must be set not both")
        elif self.repetitions is not None and self.variance is None:
            # use sampler to sample results. Use cirq measure_observable routine
            if np.isclose(self.repetitions, np.inf):
                # use simulator to exactly calculate expecation
                self.stopping_criteria = RepetitionsStoppingCriteria(
                    total_repetitions=np.inf)
                return _sim_measure_observables
            else:
                # use sampler to sample expectation
                self.stopping_criteria = RepetitionsStoppingCriteria(
                    total_repetitions=self.repetitions)
                return cirq.work.observable_measurement.measure_observables

        elif self.variance is not None and self.repetitions is None:
            if np.isclose(self.variance, 0, atol=1.0E-14):
                # use simulator to exactly calculate expectation.
                # zero variance
                self.stopping_criteria = VarianceStoppingCriteria(variance_bound=0)
                return _sim_measure_observables
            else:
                # use sampler to sample and measure until variance is stopped
                self.stopping_criteria = VarianceStoppingCriteria(variance_bound=self.variance)
                return cirq.work.observable_measurement.measure_observables
        else:
            raise InputError("Repetitions: {} \t Variance: {} is invalid config".format(self.repetitions, self.variance))





