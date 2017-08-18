"""Calculate Fisher z-score and return multiconformer model based on cutoff."""

import argparse
from string import ascii_uppercase
from itertools import izip
from sys import stdout

import numpy as np

from .validator import Validator
from .volume import Volume
from .structure import Structure


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("map", type=str,
            help="CCP4 map file.")
    p.add_argument("resolution", type=float,
            help="Resolution of map.")
    p.add_argument("conformers", nargs="+",
            help="PDB files containing conformers.")

    p.add_argument("-r", "--radius", type=float, default=1.5,
            help="Radius of masking atoms.")
    p.add_argument("-c", "--cutoff", type=float, default=1, metavar="<float>",
            help="Number of sigmas to rscc needs to increase for a conformer to be included.")
    p.add_argument("--shift-cutoff", type=float, default=3, metavar="<float>",
            help="Geometric center shift cutoff in angstrom.")
    p.add_argument("--rmsd-cutoff", type=float, default=6, metavar="<float>",
            help="RMSD cutoff in angstrom.")
    p.add_argument("-o", "--output", default="multiconformer.pdb", metavar="<string>",
            help="Name of output file.")
    p.add_argument("-z", "--zscore-file", default=None, metavar="<string>",
            help="Name of output file containing fisher z score in sigma.")
    p.add_argument("-s", "--simple", action='store_true',
            help="Use fast simple density creation.")

    args = p.parse_args()
    return args


def main():

    args = parse_args()
    xmap = Volume.fromfile(args.map).fill_unit_cell()
    xmap.set_spacegroup("P1")
    structures = [Structure.fromfile(fname) for fname in args.conformers]

    if args.zscore_file is None:
        zscore_file = stdout
    else:
        zscore_file = open(args.zscore_file, 'w')

    validator = Validator(xmap, args.resolution)
    # Get cross-correlation for each structure and sort accordingly
    for s in structures:
        s.rscc = validator.rscc(s, rmask=args.radius)
    structures_sorted = sorted(structures, 
            key=lambda structure: structure.rscc, reverse=True)
    for fname, s in izip(args.conformers, structures):
        zscore_file.write('{fname}\t{rscc:.3f}\n'.format(fname=fname, rscc=s.rscc))

    # Build up the multiconformer model.
    multiconformer = structures_sorted[0]
    multiconformer.data['altloc'].fill('A')
    starting_conformer = multiconformer
    #fisherz_best = validator.fisher_z(starting_conformer)
    center_best = starting_conformer.coor.mean(axis=0)
    character_index = 0
    nconformers = 1
    for structure in structures_sorted[1:]:
        structure.data['altloc'].fill(ascii_uppercase[nconformers])

        ## Check if correlation of individual conformer is significantly lower
        #fisherz = validator.fisherz(structure)
        #if (fisherz_best - fisherz) > 5:
        #    continue

        # Remove conformer if center of mass has shifted significantly or RMSD
        # is too high or low
        center = structure.coor.mean(axis=0)
        rmsd = starting_conformer.rmsd(structure)
        shift = np.linalg.norm(center_best - center)
        if shift > 3.0 or rmsd > 6.0 or rmsd < 0.2:
            continue

        new_multiconformer = multiconformer.combine(structure)
        diff = validator.fisher_z_difference(multiconformer,
                new_multiconformer, rmask=args.radius, simple=args.simple)
        rscc_multi = validator.rscc(multiconformer, rmask=args.radius, 
                mask_structure=new_multiconformer)
        rscc_new_multi = validator.rscc(new_multiconformer, rmask=args.radius)
        zscore_file.write('{rscc_old:.3f}\t{rscc_new:.3f}\t{zscore:.3f}\t{rmsd:.2f}\t{shift:.2f}\n'.format(
            rscc_old=rscc_multi, rscc_new=rscc_new_multi, zscore=diff,
            rmsd=rmsd, shift=shift))
        if diff < args.cutoff:
            continue
        nconformers += 1
        multiconformer = new_multiconformer

    # If only a single conformer is used, do not use an altloc label
    if nconformers == 1:
        multiconformer.data['altloc'].fill('')

    multiconformer.tofile(args.output)
    zscore_file.close()


if __name__ == '__main__':
    main()

