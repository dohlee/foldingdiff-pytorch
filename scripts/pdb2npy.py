import argparse
import biotite.structure as struc
import numpy as np
import warnings; warnings.filterwarnings('ignore')

from biotite.structure.io.pdb import PDBFile

def parse_argument():
    parser = argparse.ArgumentParser(description='pdb2npy')
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_argument()

    pdb = PDBFile.read(args.input)
    
    if pdb.get_model_count() > 1:
        raise ValueError("Multiple models are not supported")
    
    structure = pdb.get_structure(model=1)

    # dihedrals
    phi, psi, omega = struc.dihedral_backbone(structure)

    # angles
    backbone = structure[struc.filter_backbone(structure)]
    n = len(backbone)

    triplet_indices = np.array([
        np.arange(n-2),
        np.arange(1, n-1),
        np.arange(2, n)
    ]).T

    theta1 = struc.index_angle(backbone, triplet_indices[range(0, n-2, 3)])
    theta2 = struc.index_angle(backbone, triplet_indices[range(1, n-2, 3)])
    theta3 = struc.index_angle(backbone, triplet_indices[range(2, n-2, 3)])

    npy = np.array([
        phi,
        psi,
        omega,
        theta1,
        np.hstack([theta2, np.nan]), # theta2 is not defined for the last residue
        np.hstack([theta3, np.nan]), # theta3 is not defined for the last residue
    ]).T
    
    np.save(args.output, npy)



