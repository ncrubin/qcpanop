from collections import defaultdict

import cirq
from cirq.work.observable_measurement_data import (
    BitstringAccumulator,
    ObservableMeasuredResult,
    flatten_grouped_results,
)

import numpy as np

from qcpanop.qubit_measurement.bell_basis_marginals import (
    build_all_z_paulistrings, get_qubit_marginals,
    get_qubit_marginal_expectations, build_paulistring, two_q_bell_measurement,
    two_q_standard_measurement, build_twoq_measurement_circuit,
    BellBasisKMarginalSampler,
    qubit_marginal_op_basis)

from qcpanop.qubit_measurement.circuits import (zeta_circuit, bell_measurement, create_measurement_circuit,
                                                )

from qcpanop.qubit_measurement.cirq_measurement_interface import DataCollectorFactory


def bell_measure_finite_samples():
    print()
    from qcpanop.qubit_measurement.circuits import create_measurement_circuit
    np.random.seed(15)
    n_qubits = 6
    n_ancilla = n_qubits
    qubits = cirq.LineQubit.range(n_qubits + n_ancilla)
    sys_qubits = qubits[:n_qubits]
    sys_qubit_map = dict(zip(sys_qubits, range(len(sys_qubits))))
    ancilla_qubits = qubits[n_qubits:]
    qubit_map = dict(zip(qubits, range(n_qubits + n_ancilla)))
    ancilla_map = dict(zip(qubits[:n_qubits], qubits[n_qubits:]))
    marginal_rank = 4


    standard_basis_terms_to_bell_basis_terms = build_all_z_paulistrings(
        qubit_marginal_rank=marginal_rank,
        system_qubits=sys_qubits,
        system_to_ancilla_qubit_map=ancilla_map)
    bell_basis_terms_to_standard_basis = dict(
        zip(standard_basis_terms_to_bell_basis_terms.values(),
            standard_basis_terms_to_bell_basis_terms.keys()))

    sb_terms_to_measure = list(set(standard_basis_terms_to_bell_basis_terms.keys()))
    bb_terms_to_measure = list(set(standard_basis_terms_to_bell_basis_terms.values()))


    # get a random state as a circuit.
    hadmard_circuit = cirq.Circuit([cirq.H.on(xx) for xx in qubits[:n_qubits]])
    random_circuit = cirq.testing.random_circuit(qubits[:n_qubits], n_moments=10,
                                                 op_density=1,
                                                 gate_domain={cirq.rx(np.pi / 3): 1,
                                                              cirq.CZ: 2,
                                                              cirq.rz(np.pi / 4): 1,
                                                              cirq.ry(
                                                                  np.pi / 7): 1})
    random_circuit = hadmard_circuit + random_circuit

    inst_bbkms = BellBasisKMarginalSampler(circuit=random_circuit,
                                           system_qubits=sys_qubits,
                                           system_to_ancilla_map=ancilla_map,
                                           sampler=cirq.Simulator(dtype=np.complex128),
                                           marginal_rank=marginal_rank,
                                           repetitions=100_000,
                                           )
    test_marginal_dict = inst_bbkms.measure_marginals()
    exit()


    # get the margials. This is ground truth
    marginals = get_qubit_marginals(
        random_circuit.final_state_vector(qubit_order=qubits[:n_qubits]),
        qubits[:n_qubits], marginal_rank)
    marginals_exp = get_qubit_marginal_expectations(
        random_circuit.final_state_vector(qubit_order=qubits[:n_qubits]),
        qubits[:n_qubits], marginal_rank)

    # build bell basis measurement circuit
    measurement_circuit = cirq.Circuit()
    for sys_q, anc_q in zip(qubits[:n_qubits], qubits[n_qubits:]):
        measurement_circuit += zeta_circuit(anc_q)

    bell_meas_circuit =create_measurement_circuit(system_qubits=sys_qubits,
                                                sys_to_ancilla_map=ancilla_map)
    final_state_bell = (random_circuit + bell_meas_circuit).final_state_vector(qubit_order=qubits)
    # final_state_bell = (random_circuit + measurement_bell).final_state_vector(
    #     qubit_order=qubits)
    final_state = (random_circuit + measurement_circuit).final_state_vector(
        qubit_order=qubits)
    final_state_no_ancilla = (random_circuit).final_state_vector(
        qubit_order=qubits[:n_qubits])


    dcf = DataCollectorFactory(repetitions=np.inf, sampler=cirq.Simulator(dtype=np.complex128))
    measure_observables = dcf.get_measure_observables()
    sb_measured_results = measure_observables(circuit=random_circuit,
                                              observables=sb_terms_to_measure,
                                              sampler=dcf.sampler,
                                              stopping_criteria=dcf.stopping_criteria)
    sb_measured_results = dict(zip(sb_terms_to_measure, sb_measured_results))
    dcf = DataCollectorFactory(repetitions=np.inf, sampler=cirq.Simulator(dtype=np.complex128))
    measure_observables = dcf.get_measure_observables()
    bb_measured_results = measure_observables(circuit=random_circuit + bell_meas_circuit,
                                              observables=bb_terms_to_measure,
                                              sampler=dcf.sampler,
                                              stopping_criteria=dcf.stopping_criteria)
    bb_measured_results = dict(zip(bb_terms_to_measure, bb_measured_results))

    for sb_obs, sb_mr in sb_measured_results.items():
        bb_obs = standard_basis_terms_to_bell_basis_terms[sb_obs]
        bb_mr = bb_measured_results[bb_obs]
        print(sb_obs, standard_basis_terms_to_bell_basis_terms[sb_obs], sb_mr.mean, bb_mr.mean)
        assert np.isclose(sb_mr.mean, bb_mr.mean, atol=1.0E-2)

    print("Populate marginal exp dictionary")
    qubit_marginal_basis = qubit_marginal_op_basis(marginal_rank, sys_qubits)
    marginal_dict = defaultdict(dict)
    for key, val in qubit_marginal_basis.items():
        for pterm in val:
            if pterm == cirq.PauliString():
                marginal_dict[key][pterm] = ObservableMeasuredResult(mean=1,
                                                                     variance=0,
                                                                     repetitions=np.inf,
                                                                     circuit_params=dict(),
                                                                     setting=None)
            else:
                print(pterm, sb_measured_results[pterm].mean)
                bb_obs = standard_basis_terms_to_bell_basis_terms[pterm]
                bb_mr = bb_measured_results[bb_obs]
                marginal_dict[key][pterm] = bb_mr  # asign from Bell basis measurement

    for key, val in qubit_marginal_basis.items():
        for pterm in val:
            print(marginals_exp[key][pterm], marginal_dict[key][pterm].mean, test_marginal_dict[key][pterm].mean)
            assert np.isclose(marginals_exp[key][pterm], marginal_dict[key][pterm].mean)
            assert np.isclose(marginals_exp[key][pterm], test_marginal_dict[key][pterm].mean)

if __name__ == "__main__":
    bell_measure_finite_samples()