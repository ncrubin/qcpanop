"""
Test the bell measurement for obtaining all qubit margiinals
"""
from typing import Dict, List

from collections import defaultdict

import numpy as np

import cirq
from cirq.work.observable_measurement_data import (
    BitstringAccumulator,
    ObservableMeasuredResult,
    flatten_grouped_results,
)

from qcpanop.qubit_measurement.cirq_measurement_interface import \
    DataCollectorFactory
from qcpanop.qubit_measurement.qubit_marginal_ops import (
    qubit_marginal_op_basis, get_qubit_marginals,
    get_qubit_marginal_expectations, )
from qcpanop.qubit_measurement.state_utils import get_random_state
from qcpanop.qubit_measurement.circuits import (zeta_circuit, bell_measurement,
                                                create_measurement_circuit, )


# [standard basis] -> [system, ancilla, phase]
qubit_basis_to_bell_basis = {'x': ['I', 'Z', 1],
                             'y': ['Z', 'Z', -1],
                             'z': ['Z', 'I', 1]}

# bell basis measurement equivalents [system, system] -> [system, ancilla, system, ancilla]
two_q_standard_measurement = {'xx': ['X', 'X', 'X', 'X'],
                              'xy': ['X', 'X', 'Y', 'Y'],
                              'xz': ['X', 'X', 'Z', 'Z'],
                              'yx': ['Y', 'Y', 'X', 'X'],
                              'yy': ['Y', 'Y', 'Y', 'Y'],
                              'yz': ['Y', 'Y', 'Z', 'Z'],
                              'zx': ['Z', 'Z', 'X', 'X'],
                              'zy': ['Z', 'Z', 'Y', 'Y'],
                              'zz': ['Z', 'Z', 'Z', 'Z'],
                              'x': ['X', 'I', 'I', 'I'],
                              'y': ['Y', 'I', 'I', 'I'],
                              'z': ['Z', 'I', 'I', 'I']}

# bell basis measurement equivalents [system, system] -> [system, ancilla, system, ancilla]
two_q_bell_measurement = {'xx': ['I', 'Z', 'I', 'Z'],
                          'xy': ['I', 'Z', 'Z', 'Z'],  # -1 phase
                          'xz': ['I', 'Z', 'Z', 'I'],
                          'yx': ['Z', 'Z', 'I', 'Z'],  # -1 phase
                          'yy': ['Z', 'Z', 'Z', 'Z'],
                          'yz': ['Z', 'Z', 'Z', 'I'],  # -1 phase
                          'zx': ['Z', 'I', 'I', 'Z'],
                          'zy': ['Z', 'I', 'Z', 'Z'],  # -1 phase
                          'zz': ['Z', 'I', 'Z', 'I'],
                          'x': ['I', 'Z', 'I', 'I'],
                          'y': ['Z', 'Z', 'I', 'I'],   # -1 phase
                          'z': ['Z', 'I', 'I', 'I']}


def build_paulistring(qubit_list, pauli_type='X'):
    """Builds a PauliString from a list of qubits
    this is an entire funcntion because we might want to build
    different types of PauliStrings for different lists of qubits
    """
    bell_pair_dict = {}
    for qidx, qid in enumerate(qubit_list):
        if isinstance(pauli_type, list):
            bell_pair_dict[qid] = pauli_type[qidx]
        else:
            bell_pair_dict[qid] = pauli_type
    return cirq.PauliString(bell_pair_dict)


def build_twoq_measurement_circuit(system_qubits: List[cirq.Qid],
                                   ancilla_qubits: List[cirq.Qid],
                                   with_measurements) -> cirq.OP_TREE:
    """
    Construct the measurement circuit to measure in the bell basis

    :param system_qubits: List of system qubits
    :param ancilla_qubits:  List of ancilla qubits
    :return: cirq.Circuit for measurements
    """
    measurement_bell = cirq.Circuit()
    for sys_q, anc_q in zip(system_qubits, ancilla_qubits):
        measurement_bell += zeta_circuit(anc_q)
        measurement_bell += bell_measurement(ancilla_qubit=anc_q,
                                             system_qubit=sys_q,
                                             with_measurements=with_measurements)
    return measurement_bell


def build_all_z_paulistrings(qubit_marginal_rank: int,
                             system_qubits: List[cirq.Qid],
                             system_to_ancilla_qubit_map: dict) -> List[cirq.PauliString]:
    """
    Construct all Z-like operators to measure to estimate the k-qubit marginals.
    We include the minus signs for the Odd Y values. This function takes the
    rank of the marginal and a list of qubits as we use the qubit IDs as keys.

    :param qubit_marginal_rank: size of marginal to obtain
    :param system_qubits: all system qubits
    :param system_to_ancilla_qubit_map all ancilla qubits
    :return: dictionary map cirq.PauliString->cirq.PauliString where the key
             is the standard basis operator and the value is the bell-basis
             operator. The bell basis operator comes with the appropriate sign.
             This is not the Hamiltonian sign...just the sign to account for
             measuring y.
    """
    # generate all observables for a k-marginal
    qubit_marginal_basis = qubit_marginal_op_basis(qubit_marginal_rank, system_qubits)
    all_non_identity_paulistrings = []
    for key, val in qubit_marginal_basis.items():
        for pterm in val:
            if pterm != cirq.PauliString():  # check if pterm is identity
                all_non_identity_paulistrings.append(pterm)
    all_non_identity_paulistrings = list(set(all_non_identity_paulistrings))
    result_map = {}
    for pstring in all_non_identity_paulistrings:
        bell_pauli_measurement_string = {}
        phase = 1  # to account for the phase from y
        for qid, pterm in pstring._qubit_pauli_map.items():
            if pterm == cirq.X:
                bell_pauli_measurement_string[qid] = cirq.I
                anc_q = system_to_ancilla_qubit_map[qid]
                bell_pauli_measurement_string[anc_q] = cirq.Z
                phase *= 1
            elif pterm == cirq.Y:
                bell_pauli_measurement_string[qid] = cirq.Z
                anc_q = system_to_ancilla_qubit_map[qid]
                bell_pauli_measurement_string[anc_q] = cirq.Z
                phase *= -1
            elif pterm == cirq.Z:
                bell_pauli_measurement_string[qid] = cirq.Z
                anc_q = system_to_ancilla_qubit_map[qid]
                bell_pauli_measurement_string[anc_q] = cirq.I
                phase *= 1
            else:
                raise ValueError("This should not happen")

        result_map[pstring] = cirq.PauliString(bell_pauli_measurement_string,
                                               coefficient=phase * (np.sqrt(3)**len(pstring._qubit_pauli_map.keys())))
    return result_map


class BellBasisKMarginalSampler:

    def __init__(self, circuit: cirq.Circuit,
                 system_qubits: List[cirq.Qid],
                 system_to_ancilla_map: Dict[cirq.Qid, cirq.Qid],
                 sampler: cirq.Sampler,
                 marginal_rank: int,
                 repetitions=None,
                 variance_bound=None):
        """
        This is an object instead of a function because we want somehwere to
        hold intermediate data from the Bell basis measurements

        :param circuit: circuit without ancilla
        :param system_qubits: list of system qubits
        :param system_to_ancilla_map: dictionary mapping system_qubit to
                                      an ancilla_qubit that is allowed on the
                                      device
        :param sampler: a cirq Sampler
        :param marginal_rank: set the rank of the marginals to be measured
        :param repetitions: None controls how finely to measure
        :param variance_bound: None controls how finely to measure
        """
        # store user input settings
        self.circuit = circuit
        self.system_qubits = system_qubits
        self.system_to_ancilla_map = system_to_ancilla_map
        self.sampler = sampler
        self.marginal_rank = marginal_rank
        self.repetitions = repetitions
        self.variance_boudn = variance_bound

        # generate all standard basis terms and their corresponding
        # bell basis PaulString terms
        self.standard_basis_terms_to_bell_basis_terms = build_all_z_paulistrings(
            qubit_marginal_rank=self.marginal_rank,
            system_qubits=self.system_qubits,
            system_to_ancilla_qubit_map=self.system_to_ancilla_map)
        self.bell_basis_terms_to_standard_basis = dict(
            zip(self.standard_basis_terms_to_bell_basis_terms.values(),
                self.standard_basis_terms_to_bell_basis_terms.keys()))
        # make a minimal set to measure
        self.sb_terms_to_measure = list(
            set(self.standard_basis_terms_to_bell_basis_terms.keys()))
        self.bb_terms_to_measure = list(
            set(self.standard_basis_terms_to_bell_basis_terms.values()))

        # set up bell circuit and full bell measurement circuit -- WITHOUT the measurements
        self.bell_meas_circuit = create_measurement_circuit(system_qubits=self.system_qubits,
                                                       sys_to_ancilla_map=self.system_to_ancilla_map,
                                                            compile=True)
        self.circuit_plus_bell_meas_circuit = self.circuit + self.bell_meas_circuit

        # set up the sampler factory
        dcf = DataCollectorFactory(repetitions=self.repetitions,
                                   variance=variance_bound,
                                   sampler=self.sampler)
        self.data_collector = dcf
        self.measure_observables_func = self.data_collector.get_measure_observables()

    def measure_marginals(self):
        # do the measurement of the bell basis Z-terms
        bb_measured_results = self.measure_observables_func(
            circuit=self.circuit_plus_bell_meas_circuit,
            observables=self.bb_terms_to_measure,
            sampler=self.data_collector.sampler,
            stopping_criteria=self.data_collector.stopping_criteria)
        self.bb_measured_results = dict(zip(self.bb_terms_to_measure, bb_measured_results))

        # Populate marginal exp dictionary
        qubit_marginal_basis = qubit_marginal_op_basis(self.marginal_rank, self.system_qubits)

        marginal_dict = defaultdict(dict)
        for key, val in qubit_marginal_basis.items():
            for pterm in val:
                if pterm == cirq.PauliString():
                    # set identity terms to zero variance because not measured
                    marginal_dict[key][pterm] = ObservableMeasuredResult(mean=1,
                                                                         variance=0,
                                                                         repetitions=np.inf,
                                                                         circuit_params=dict(),
                                                                         setting=None)
                else:
                    bb_obs = self.standard_basis_terms_to_bell_basis_terms[pterm]
                    bb_mr = self.bb_measured_results[bb_obs]
                    marginal_dict[key][pterm] = bb_mr  # asign from Bell basis measurement
        return marginal_dict
