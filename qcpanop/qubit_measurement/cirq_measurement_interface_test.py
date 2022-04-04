import pytest

import cirq

import numpy as np

from qcpanop.qubit_measurement.qubit_marginal_ops import (
    get_qubit_marginal_expectations,)
from qcpanop.qubit_measurement.cirq_measurement_interface import (
    InputError,
    DataCollectorFactory,
    RepetitionsStoppingCriteria,
    VarianceStoppingCriteria,
    ObservableMeasuredResult)


def test_datacollector_init():
    # set up simulation
    # test bad config 1
    with pytest.raises(InputError):
        sampler_factory = DataCollectorFactory()
        sampler_factory.get_measure_observables()
    # test bad config 2
    with pytest.raises(InputError):
        sampler_factory = DataCollectorFactory(repetitions=5,
                                               variance=0)
        sampler_factory.get_measure_observables()

def test_simulator_path():
    # testing the simulator
    n_qubits = 4
    n_ancilla = n_qubits
    qubits = cirq.LineQubit.range(n_qubits + n_ancilla)
    qubit_map = dict(zip(qubits, range(n_qubits + n_ancilla)))
    ancilla_map = dict(zip(qubits[:n_qubits], qubits[n_qubits:]))

    # get a random state as a circuit.
    hadmard_circuit = cirq.Circuit([cirq.H.on(xx) for xx in qubits[:n_qubits]])
    random_circuit = cirq.testing.random_circuit(qubits[:n_qubits], n_moments=10, op_density=1,
                                                 gate_domain={cirq.rx(np.pi/3): 1,
                                                              cirq.CZ: 2,
                                                              cirq.rz(np.pi/4): 1,
                                                              cirq.ry(np.pi/7): 1})
    random_circuit = hadmard_circuit + random_circuit
    print(random_circuit)
    u1 = cirq.unitary(random_circuit)
    random_circuit = cirq.transformers.merge_single_qubit_gates_to_phxz(circuit=random_circuit)
    print(random_circuit)
    u2 = cirq.unitary(random_circuit)
    print(abs(np.trace(u2.conj().T @ u1)))
    random_circuit = cirq.transformers.align_left(random_circuit)
    random_circuit = cirq.transformers.merge_single_qubit_gates_to_phxz(circuit=random_circuit)
    random_circuit = cirq.transformers.align_left(random_circuit)
    print(random_circuit)
    u3 = cirq.unitary(random_circuit)
    print(abs(np.trace(u3.conj().T @ u2)))

    marginals_exp = get_qubit_marginal_expectations(random_circuit.final_state_vector(qubit_order=qubits[:n_qubits]),
        qubits[:n_qubits], 3)



    # test np.inf repetition
    sampler_factory = DataCollectorFactory(repetitions=np.inf,
                                           sampler=cirq.Simulator(
                                               dtype=np.complex128))
    measure_observables = sampler_factory.get_measure_observables()
    for key, val in marginals_exp.items():
        observables_to_measure = list(val.keys())
        obs_results = measure_observables(observables=observables_to_measure,
                                          circuit=random_circuit,
                                          sampler=sampler_factory.sampler,
                                          stopping_criteria=RepetitionsStoppingCriteria(total_repetitions=np.inf)
                                          )
        for obs_results_obj, obs in zip(obs_results, observables_to_measure):
            assert np.isclose(obs_results_obj.variance, 0)
            assert np.isclose(obs_results_obj.mean, val[obs])

    # test 0 variance
    sampler_factory = DataCollectorFactory(variance=0,
                                           sampler=cirq.Simulator(
                                               dtype=np.complex128))
    measure_observables = sampler_factory.get_measure_observables()
    for key, val in marginals_exp.items():
        observables_to_measure = list(val.keys())
        obs_results = measure_observables(observables=observables_to_measure,
                                          circuit=random_circuit,
                                          sampler=sampler_factory.sampler,
                                          stopping_criteria=RepetitionsStoppingCriteria(total_repetitions=np.inf)
                                          )
        for obs_results_obj, obs in zip(obs_results, observables_to_measure):
            assert np.isclose(obs_results_obj.variance, 0)
            assert np.isclose(obs_results_obj.mean, val[obs])

def test_sampler_path():
    # testing the simulator
    n_qubits = 4
    n_ancilla = n_qubits
    qubits = cirq.LineQubit.range(n_qubits + n_ancilla)
    qubit_map = dict(zip(qubits, range(n_qubits + n_ancilla)))
    ancilla_map = dict(zip(qubits[:n_qubits], qubits[n_qubits:]))

    # get a random state as a circuit.
    hadmard_circuit = cirq.Circuit([cirq.H.on(xx) for xx in qubits[:n_qubits]])
    random_circuit = cirq.testing.random_circuit(qubits[:n_qubits], n_moments=10, op_density=1,
                                                 gate_domain={cirq.rx(np.pi/3): 1,
                                                              cirq.CZ: 2,
                                                              cirq.rz(np.pi/4): 1,
                                                              cirq.ry(np.pi/7): 1})
    random_circuit = hadmard_circuit + random_circuit
    print(random_circuit)
    u1 = cirq.unitary(random_circuit)
    random_circuit = cirq.transformers.merge_single_qubit_gates_to_phxz(circuit=random_circuit)
    print(random_circuit)
    u2 = cirq.unitary(random_circuit)
    print(abs(np.trace(u2.conj().T @ u1)))
    random_circuit = cirq.transformers.align_left(random_circuit)
    random_circuit = cirq.transformers.merge_single_qubit_gates_to_phxz(circuit=random_circuit)
    random_circuit = cirq.transformers.align_left(random_circuit)
    print(random_circuit)
    u3 = cirq.unitary(random_circuit)
    print(abs(np.trace(u3.conj().T @ u2)))

    marginals_exp = get_qubit_marginal_expectations(random_circuit.final_state_vector(qubit_order=qubits[:n_qubits]),
        qubits[:n_qubits], 3)



    # test np.inf repetition
    sampler_factory = DataCollectorFactory(repetitions=np.inf,
                                           sampler=cirq.Simulator(
                                               dtype=np.complex128))
    measure_observables = sampler_factory.get_measure_observables()
    for key, val in marginals_exp.items():
        observables_to_measure = list(val.keys())
        obs_results = measure_observables(observables=observables_to_measure,
                                          circuit=random_circuit,
                                          sampler=sampler_factory.sampler,
                                          stopping_criteria=RepetitionsStoppingCriteria(total_repetitions=np.inf)
                                          )
        for obs_results_obj, obs in zip(obs_results, observables_to_measure):
            assert np.isclose(obs_results_obj.variance, 0)
            assert np.isclose(obs_results_obj.mean, val[obs])

    # test 0 variance
    sampler_factory = DataCollectorFactory(variance=0,
                                           sampler=cirq.Simulator(
                                               dtype=np.complex128))
    measure_observables = sampler_factory.get_measure_observables()
    for key, val in marginals_exp.items():
        observables_to_measure = list(val.keys())
        obs_results = measure_observables(observables=observables_to_measure,
                                          circuit=random_circuit,
                                          sampler=sampler_factory.sampler,
                                          stopping_criteria=RepetitionsStoppingCriteria(total_repetitions=np.inf)
                                          )
        for obs_results_obj, obs in zip(obs_results, observables_to_measure):
            assert np.isclose(obs_results_obj.variance, 0)
            assert np.isclose(obs_results_obj.mean, val[obs])

    # test finite repetitions
    sampler_factory = DataCollectorFactory(repetitions=500_000,
                                           sampler=cirq.Simulator(
                                               dtype=np.complex128))
    measure_observables = sampler_factory.get_measure_observables()
    sim = cirq.Simulator(dtype=np.complex128)
    sim_res = sim.simulate(random_circuit)
    for key, val in marginals_exp.items():
        observables_to_measure = [cirq.PauliString({key[0]: cirq.Z}),
                                  cirq.PauliString({key[1]: cirq.Z}),
                                  cirq.PauliString({key[2]: cirq.Z}),
                                  cirq.PauliString({key[0]: cirq.Z,
                                                    key[1]: cirq.Z}),
                                  cirq.PauliString({key[0]: cirq.Z,
                                                    key[2]: cirq.Z}),
                                  cirq.PauliString({key[1]: cirq.Z,
                                                    key[2]: cirq.Z}),
                                  cirq.PauliString({key[0]: cirq.Z,
                                                    key[1]: cirq.Z,
                                                    key[2]: cirq.Z}),
                                  ]

        obs_results = measure_observables(observables=observables_to_measure,
                                          circuit=random_circuit,
                                          sampler=sampler_factory.sampler,
                                          stopping_criteria=RepetitionsStoppingCriteria(total_repetitions=sampler_factory.repetitions)
                                          )
        assert isinstance(obs_results[0], ObservableMeasuredResult)
        for reso, obs in zip(obs_results, observables_to_measure):
            tval = obs.expectation_from_state_vector(
                    sim_res.final_state_vector, qubit_map=sim_res.qubit_map,
                    check_preconditions=False).real
            assert np.isclose(reso.mean, tval, atol=1.0E-2)
            assert np.isclose(reso.mean, val[obs], atol=1.0E-2)
            print("{}: {} +- {}, {}, {}".format(repr(obs), reso.mean, np.sqrt(reso.variance), val[obs], tval))
            print()