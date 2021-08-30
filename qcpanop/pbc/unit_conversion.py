def hatree_to_rydberg(hartree_val: float) -> float:
    return 2. * hartree_val

def rydberg_to_hartree(rydberg_val: float) -> float:
    return 0.5 * rydberg_val

def hartree_to_ev(hartree_val: float) -> float:
    return 27.2114 * hartree_val

def ev_to_hartree(ev_val: float) -> float:
    return ev_val / 27.2114

