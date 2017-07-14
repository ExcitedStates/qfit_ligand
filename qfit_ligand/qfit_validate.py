"""Calculate Fisher z-score and return multiconformer model based on cutoff."""

import argparse
from string import ascii_uppercase

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
    p.add_argument("-c", "--cutoff", type=float, default=1,
            help="Number of sigmas standard deviation needs to be higher.")
    p.add_argument("-o", "--output", default="multiconformer.pdb",
            help="Name of output file.")

    args = p.parse_args()
    return args


def main():

    args = parse_args()
    args.map.set_spacegroup("P1")

    validator = Validator(args.map, args.resolution)
    structures_sorted = sorted(args.structures, key=lambda structure: structure.q[0], reverse=True)
    multiconformer = structures_sorted[0]
    multiconformer.data['altloc'].fill('A')
    character_index = 0
    nconformers = 1
    for structure in structures_sorted[1:]:
        structure.data['altloc'].fill(ascii_uppercase[nconformers])
        new_multiconformer = multiconformer.combine(structure)
        diff = validator.fisher_z_difference(multiconformer, new_multiconformer, simple=True)
        print diff
        if diff < args.cutoff:
            continue
        nconformers += 1
        multiconformer = new_multiconformer
    if nconformers == 1:
        multiconformer.data['altloc'].fill('')
    multiconformer.tofile(args.output)



if __name__ == '__main__':
    main()

