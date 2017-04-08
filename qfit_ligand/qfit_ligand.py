"""Hierarchically build a multiconformer ligand."""

import argparse
from math import pi
import os.path
import sys
import logging

from .builders import HierarchicalBuilder
from .structure import Ligand, Structure
from .helpers import mkdir_p


def parse_args():

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str,
            help="X-ray density map.")
    p.add_argument("resolution", type=float,
            help="Map resolution in angstrom.")
    p.add_argument("ligand", type=str,
            help="Ligand file in PDB format.")
    p.add_argument("-r", "--receptor", dest="receptor", type=str, default=None,
            metavar="<file>",
            help="PDB file containing receptor for clash detection.")
    p.add_argument("-b", "--build-stepsize", type=int, default=2, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--stepsize", type=float, default=90, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("-d", "--directory", dest="directory", type=os.path.abspath, 
            default='.', metavar="<dir>",
            help="Directory to store results.")
    p.add_argument("-nc", "--no-clean", action="store_true",
            help="Do not remove output files.")
    p.add_argument("-nl", "--no-local", action="store_true",
            help="Do not perform a local search.")
    p.add_argument("-v", "--verbose", action="store_true",
            help="Be verbose.")
    args = p.parse_args()

    return args


def main():

    args = parse_args()
    mkdir_p(args.directory)
    logging.basicConfig(filename=os.path.join(args.directory, 'qfit_ligand.log'), 
                        level=logging.INFO)
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console_out)

    ligand = Ligand.fromfile(args.ligand)
    if args.receptor is not None:
        receptor = Structure.fromfile(args.receptor)
    else:
        receptor = None

    builder = HierarchicalBuilder(
            ligand, args.xmap, args.resolution, receptor=receptor, 
            build_stepsize=args.build_stepsize, stepsize=args.stepsize, 
            local_search=(not args.no_local), directory=args.directory,
            clean=(not args.no_clean),
            )
    builder()

