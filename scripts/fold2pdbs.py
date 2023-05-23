import argparse
import os
import torch
import numpy as np
import biotite.structure as struc

from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm


def place_fourth_atom(a, b, c, bond_angle, torsion, bond_length):
    # Place atom D with respect to atom C at origin.
    d = np.array(
        [
            bond_length * np.cos(np.pi - bond_angle),
            bond_length * np.cos(torsion) * np.sin(bond_angle),
            bond_length * np.sin(torsion) * np.sin(bond_angle),
        ]
    ).T

    # Transform atom D to the correct frame.
    bc = c - b
    bc /= np.linalg.norm(bc)  # Unit vector from B to C.

    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)  # Normal vector of the plane defined by a, b, c.

    M = np.array([bc, np.cross(n, bc), n]).T
    return M @ d + c


def angles2coord(angles, n_ca=1.46, ca_c=1.54, c_n=1.33):
    """Given L x 6 angle matrix,
    reconstruct the Cartesian coordinates of atoms.
    Returns L x 3 coordinate matrix.

    Implements NeRF (Natural Extension Reference Frame) algorithm.
    """
    if isinstance(angles, torch.Tensor):
        phi, psi, omega, theta1, theta2, theta3 = angles.T.numpy()
    else:
        phi, psi, omega, theta1, theta2, theta3 = angles.T

    torsions = np.stack([psi[:-1], omega[:-1], phi[1:]], axis=-1).flatten()
    bond_angles = np.stack([theta2[:-1], theta3[:-1], theta1[1:]], axis=-1).flatten()

    #
    # Place the first three atoms.
    #
    # The first atom (N) is placed at origin.
    a = np.zeros(3)
    # The second atom (Ca) is placed on the x-axis.
    b = np.array([1, 0, 0]) * n_ca
    # The third atom (C) is placed on the xy-plane with bond angle theta1[0]
    c = np.array([np.cos(np.pi - theta1[0]), np.sin(np.pi - theta1[0]), 0]) * ca_c + b

    # Iteratively place the fourth atom based on the last three atoms.

    coords = [a, b, c]
    # cycle through [n, ca, c, n, ca, c, ...]
    for i, bond_length in enumerate([c_n, n_ca, ca_c] * (len(angles) - 1)):
        torsion, bond_angle = torsions[i], bond_angles[i]
        d = place_fourth_atom(a, b, c, bond_angle, torsion, bond_length)
        coords.append(d)

        a, b, c = b, c, d

    return np.array(coords)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help=".pt file for generated folds"
    )
    parser.add_argument("-o", "--outdir", required=True, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_arguments()
    folds = torch.load(args.input)

    for fold_id, fold in enumerate(folds):
        if not os.path.exists(f"{args.outdir}/{fold_id}"):
            os.makedirs(f"{args.outdir}/{fold_id}", exist_ok=True)

        for i, angles in tqdm(enumerate(fold)):
            coords = angles2coord(angles)
            num_residues = len(coords) // 3

            structure = struc.AtomArray(len(coords))
            structure.coord = coords
            structure.atom_name = ["N", "CA", "C"] * (num_residues)
            structure.res_name = ["GLY"] * (len(coords))
            structure.res_id = np.repeat(range(1, num_residues + 1), 3)

            pdb = PDBFile()
            pdb.set_structure(structure)
            pdb.write(f"{args.outdir}/{fold_id}/{i}.pdb")


if __name__ == "__main__":
    main()
