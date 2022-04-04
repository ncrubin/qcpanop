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


def test_bell_basis_expectations():
    # np.random.seed(53)
    n_qubits = 4
    n_ancilla = n_qubits
    qubits = cirq.LineQubit.range(n_qubits + n_ancilla)
    qubit_map = dict(zip(qubits, range(n_qubits + n_ancilla)))
    ancilla_map = dict(zip(qubits[:n_qubits], qubits[n_qubits:]))

    standard_basis_terms_to_bell_basis_terms = build_all_z_paulistrings(
        qubit_marginal_rank=2,
        system_qubits=qubits[:n_qubits],
        system_to_ancilla_qubit_map=ancilla_map)

    # get a random state as a circuit.
    hadmard_circuit = cirq.Circuit([cirq.H.on(xx) for xx in qubits[:n_qubits]])
    random_circuit = cirq.testing.random_circuit(qubits[:n_qubits], n_moments=10,
                                                 op_density=1,
                                                 gate_domain={cirq.rx(np.pi / 3): 1,
                                                              cirq.CNOT: 2,
                                                              cirq.CZ: 2,
                                                              cirq.ISWAP: 2,
                                                              cirq.rz(np.pi / 4): 1,
                                                              cirq.ry(
                                                                  np.pi / 7): 1})
    random_circuit = hadmard_circuit + random_circuit

    # get the margials. This is ground truth
    marginals = get_qubit_marginals(
        random_circuit.final_state_vector(qubit_order=qubits[:n_qubits]),
        qubits[:n_qubits], 2)
    marginals_exp = get_qubit_marginal_expectations(
        random_circuit.final_state_vector(qubit_order=qubits[:n_qubits]),
        qubits[:n_qubits], 2)

    # build bell basis measurement circuit
    measurement_circuit = cirq.Circuit()
    measurement_bell = cirq.Circuit()
    for sys_q, anc_q in zip(qubits[:n_qubits], qubits[n_qubits:]):
        measurement_circuit += zeta_circuit(anc_q)
        measurement_bell += zeta_circuit(anc_q)
        measurement_bell += bell_measurement(ancilla_qubit=anc_q,
                                             system_qubit=sys_q,
                                             with_measurements=False)

    final_state_bell = (random_circuit + measurement_bell).final_state_vector(
        qubit_order=qubits)
    final_state = (random_circuit + measurement_circuit).final_state_vector(
        qubit_order=qubits)

    bell_measured_circuit = random_circuit + measurement_bell

    # build k-expectations for xx, xy, xz, yx, yy, yz, zx, zy, zz

    # xx -> IZ,IZ
    # yy -> ZZ,ZZ
    # zz -> ZI,ZI
    # xy -> IZ,ZZ
    # xz -> IZ,ZI
    # yx -> ZZ,IZ
    # yz -> ZZ,ZI
    # zx -> ZI,IZ
    # zy -> ZI,ZZ

    for key, val in marginals_exp.items():
        print(key)
        print(val)
        bell_pair_list = [(qidx, ancilla_map[qidx]) for qidx in key]
        bell_pair_list = [k for sub in bell_pair_list for k in sub]

        # now collect expectation values
        xx_paulistring = build_paulistring(bell_pair_list, 'X')
        xx_marginal_val = xx_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(xx_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.X.on(key[0]) * cirq.X.on(key[1])])
        # measure ancilla in Z basis gives
        iz_paulistring = build_paulistring(bell_pair_list, ['I', 'Z', 'I', 'Z'])
        bell_marginal_val = iz_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        assert np.isclose(bell_marginal_val * np.sqrt(3) ** 2,
                          val[cirq.X.on(key[0]) * cirq.X.on(key[1])])

        # print("Calculate XX")
        # print(iz_bell_marginal_val * np.sqrt(3)**2)
        # print(xx_marginal_val * np.sqrt(3)**2)
        # print(val[cirq.X.on(key[0]) * cirq.X.on(key[1])])

        # estimate XY
        xy_paulistring = build_paulistring(bell_pair_list, ['X', 'X', 'Y', 'Y'])
        xy_marginal_val = xy_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(xy_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.X.on(key[0]) * cirq.Y.on(key[1])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['xy'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        assert np.isclose(-bell_marginal_val * np.sqrt(3) ** 2,
                          val[cirq.X.on(key[0]) * cirq.Y.on(key[1])])

        xz_paulistring = build_paulistring(bell_pair_list, ['X', 'X', 'Z', 'Z'])
        xz_marginal_val = xz_paulistring.expectation_from_state_vector(
            final_state, qubit_map=qubit_map).real
        assert np.isclose(xz_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.X.on(key[0]) * cirq.Z.on(key[1])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['xz'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        # print(bell_marginal_val * np.sqrt(3)**2, val[cirq.X.on(key[0]) * cirq.Z.on(key[1])])
        assert np.isclose(bell_marginal_val * np.sqrt(3) ** 2,
                          val[cirq.X.on(key[0]) * cirq.Z.on(key[1])])

        yx_paulistring = build_paulistring(bell_pair_list, ['Y', 'Y', 'X', 'X'])
        yx_marginal_val = yx_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(yx_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.Y.on(key[0]) * cirq.X.on(key[1])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['yx'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        # print(-bell_marginal_val * np.sqrt(3)**2, val[cirq.Y.on(key[0]) * cirq.X.on(key[1])])
        assert np.isclose(-bell_marginal_val * np.sqrt(3) ** 2,
                          val[cirq.Y.on(key[0]) * cirq.X.on(key[1])])

        # estimate YY Term
        yy_paulistring = build_paulistring(bell_pair_list, 'Y')
        yy_marginal_val = yy_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(yy_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.Y.on(key[0]) * cirq.Y.on(key[1])])
        # measure system and ancilla in ZZ
        zz_paulistring = build_paulistring(bell_pair_list, ['Z', 'Z', 'Z', 'Z'])
        zz_bell_marginal_val = zz_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        assert np.isclose(zz_bell_marginal_val * (np.sqrt(3) ** 2),
                          yy_marginal_val * (np.sqrt(3) ** 2))
        # print("Calculate YY")
        # print(zz_bell_marginal_val * (np.sqrt(3)**2), yy_marginal_val * (np.sqrt(3)**2), val[cirq.Y.on(key[0]) * cirq.Y.on(key[1])])

        yz_paulistring = build_paulistring(bell_pair_list, ['Y', 'Y', 'Z', 'Z'])
        yz_marginal_val = yz_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(yz_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.Y.on(key[0]) * cirq.Z.on(key[1])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['yz'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        # print(-bell_marginal_val * np.sqrt(3)**2, val[cirq.Y.on(key[0]) * cirq.Z.on(key[1])])
        assert np.isclose(-bell_marginal_val * np.sqrt(3) ** 2,
                          val[cirq.Y.on(key[0]) * cirq.Z.on(key[1])])

        zx_paulistring = build_paulistring(bell_pair_list, ['Z', 'Z', 'X', 'X'])
        zx_marginal_val = zx_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(zx_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.Z.on(key[0]) * cirq.X.on(key[1])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['zx'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        # print(bell_marginal_val * np.sqrt(3)**2, val[cirq.Z.on(key[0]) * cirq.X.on(key[1])])
        assert np.isclose(bell_marginal_val * np.sqrt(3) ** 2,
                          val[cirq.Z.on(key[0]) * cirq.X.on(key[1])])

        zy_paulistring = build_paulistring(bell_pair_list, ['Z', 'Z', 'Y', 'Y'])
        zy_marginal_val = zy_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(zy_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.Z.on(key[0]) * cirq.Y.on(key[1])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['zy'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        # print(-bell_marginal_val * np.sqrt(3)**2, val[cirq.Z.on(key[0]) * cirq.Y.on(key[1])])
        assert np.isclose(-bell_marginal_val * np.sqrt(3) ** 2,
                          val[cirq.Z.on(key[0]) * cirq.Y.on(key[1])])

        # estimate ZZ term
        zz_paulistring = build_paulistring(bell_pair_list, 'Z')
        zz_marginal_val = zz_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(zz_marginal_val * (np.sqrt(3) ** 2),
                          val[cirq.Z.on(key[0]) * cirq.Z.on(key[1])])
        # measure system in Z and ancilla in I
        zi_paulistring = build_paulistring(bell_pair_list, ['Z', 'I', 'Z', 'I'])
        zi_bell_marginal_val = zi_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        assert np.isclose(zi_bell_marginal_val * np.sqrt(3) ** 2,
                          zz_marginal_val * np.sqrt(3) ** 2)

        xi_paulistring = build_paulistring(bell_pair_list, ['X', 'X', 'I', 'I'])
        xi_marginal_val = xi_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(xi_marginal_val * (np.sqrt(3) ** 1),
                          val[cirq.X.on(key[0])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['x'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        # print(bell_marginal_val * np.sqrt(3)**1, val[cirq.X.on(key[0])])
        assert np.isclose(bell_marginal_val * np.sqrt(3) ** 1,
                          val[cirq.X.on(key[0])])

        yi_paulistring = build_paulistring(bell_pair_list, ['Y', 'Y', 'I', 'I'])
        yi_marginal_val = yi_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(yi_marginal_val * (np.sqrt(3) ** 1),
                          val[cirq.Y.on(key[0])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['y'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        # print(-bell_marginal_val * np.sqrt(3)**1, val[cirq.Y.on(key[0])])
        assert np.isclose(-bell_marginal_val * np.sqrt(3) ** 1,
                          val[cirq.Y.on(key[0])])

        zi_paulistring = build_paulistring(bell_pair_list, ['Z', 'Z', 'I', 'I'])
        zi_marginal_val = zi_paulistring.expectation_from_state_vector(final_state,
                                                                       qubit_map=qubit_map).real
        assert np.isclose(zi_marginal_val * (np.sqrt(3) ** 1),
                          val[cirq.Z.on(key[0])])
        bell_paulistring = build_paulistring(bell_pair_list,
                                             two_q_bell_measurement['z'])
        bell_marginal_val = bell_paulistring.expectation_from_state_vector(
            final_state_bell, qubit_map=qubit_map).real
        print(bell_marginal_val * np.sqrt(3) ** 1, val[cirq.Z.on(key[0])])
        assert np.isclose(bell_marginal_val * np.sqrt(3) ** 1,
                          val[cirq.Z.on(key[0])])

def test_bell_measure_er():
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
    print(sb_terms_to_measure)
    print(bb_terms_to_measure)

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
                                           repetitions=np.inf
                                           )
    test_marginal_dict = inst_bbkms.measure_marginals()


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
