"""Combine output structures optionally together with receptor."""

from argparse import ArgumentParser
import string
from itertools import izip

from .structure import Structure


def parse_args():

    p = ArgumentParser(description=__doc__)
    p.add_argument("ligands", nargs="+", type=str,
            help="Ligand structures to be combined in multiconformer model.")
    p.add_argument("-r", "--receptor", type=str,
            help="Receptor.")
    p.add_argument("-o", "--output", type=str, default='multiconformer.pdb',
            help="Name of output file.")

    args = p.parse_args()
    return args


def main():

    args = parse_args()

    if args.receptor is not None:
        receptor = Structure.fromfile(args.receptor)
    
    for altloc, fname in izip(string.ascii_uppercase, args.ligands):
        l = Structure.fromfile(fname)
        l.altloc[:] = altloc
        receptor = receptor.combine(l)

    receptor.tofile(args.output)
