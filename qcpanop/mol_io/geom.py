"""
IO-routines for common formats
"""


def create_geometry_string(geometry):
    """This function converts MolecularData geometry to psi4 geometry.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))]. Distances in
            angstrom. Use atomic symbols to specify atoms.

    Returns:
        geo_string: A string giving the geometry for each atom on a line, e.g.:
            H 0. 0. 0.
            H 0. 0. 0.7414
    """
    geo_string = ''
    for item in geometry:
        atom = item[0]
        coordinates = item[1]
        line = '{} {} {} {}'.format(atom,
                                    coordinates[0],
                                    coordinates[1],
                                    coordinates[2])
        if len(geo_string) > 0:
            geo_string += '\n'
        geo_string += line
    return geo_string


def read_xyz(xyz_file, units="angstrom", return_comment=False):
    """
    XYZ is a file where the first line contains the number of atoms,
    then a space,
    then the the following is xyz coordinates.

    By default we assume coordinates are angstorm but the units flag will
    convert to bohr

    :param xyz_file:
    :return:
    """
    if units.lower() not in ["angstrom", "bohr"]:
        raise ValueError("Invalid unit selection")

    atomic_symbols = []
    xyz_coords = []

    with open(xyz_file, 'r') as fid:
        text = fid.read().strip().split('\n')
        for line_number, line in enumerate(text):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                # comment line
                comment_line = line
                continue
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coords.append([float(x), float(y), float(z)])

    if return_comment:
        return atomic_symbols, xyz_coords, comment_line
    else:
        return atomic_symbols, xyz_coords


if __name__ == "__main__":
    atoms, coords = read_xyz('sample_xyzfile.xyz')
    print(create_geometry_string(list(zip(atoms, coords))))
