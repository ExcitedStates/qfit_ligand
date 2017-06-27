"""Calculate Fisher z-score and return multiconformer model based on cutoff."""

import argparse

from .validator import Validator
from .volume import Volume
from .structure import Structure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("map", type=Volume.fromfile,
            help="CCP4 map file.")
    p.add_argument("resolution", type=float,
            help="Resolution of map.")
    p.add_argument("structures", nargs="+", type=Structure.fromfile,
            help="PDB files containing structures.")

    args = p.parse_args()
    return args


def main():

    args = parse_args()

    validator = Validator(args.map, args.resolution)
    structures_sorted = sorted(args.structures, key=lambda structure: structure.q[0], reverse=True)
    multiconformer = structures_sorted[0]
    for structure in structures_sorted[1:]:
        new_multiconformer = multiconformer.combine(structure)
        diff = validator.fisher_z_difference(multiconformer, new_multiconformer)
        print diff
        if diff < 0.5:
            continue
        multiconformer = new_multiconformer
    multiconformer.tofile('multiconformer.pdb')



if __name__ == '__main__':
    main()

