"""Hierarchically build a multiconformer ligand."""

import argparse
import os.path
import sys
import logging
logger = logging.getLogger(__name__)

import numpy as np

from .builders import HierarchicalBuilder
from .structure import Ligand, Structure
from .volume import Volume
from .helpers import mkdir_p


def parse_args():

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str,
            help="X-ray density map in CCP4 format with P1 symmetry.")
    p.add_argument("resolution", type=float,
            help="Map resolution in angstrom.")
    p.add_argument("ligand", type=str,
            help="Ligand structure in PDB format. Can also be a whole structure if selection is added with --select option.")
    p.add_argument("-r", "--receptor", type=str, default=None,
            metavar="<file>",
            help="PDB file containing receptor for clash detection.")
    p.add_argument('--selection', default=None, type=str,
            help="Chain and residue id for ligand in main PDB file, e.g. A,105.")
    #p.add_argument("-g", "--global-search", action="store_true",
    #        help="Perform a global search.")
    p.add_argument("-nb", "--no-build", action="store_true",
            help="Do not build ligand.")
    p.add_argument("-nl", "--no-local", action="store_true",
            help="Do not perform a local search.")
    p.add_argument("-b", "--build-stepsize", type=int, default=1, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--stepsize", type=float, default=1, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")
    p.add_argument("--no-scale", action="store_true",
            help="Do not scale density while building ligand.")
    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=0.1, metavar="<float>",
            help="Treshold constraint used during MIQP.")
    p.add_argument("-it", "--intermediate-threshold", type=float, default=0.01, metavar="<float>",
            help="Threshold constraint during intermediate MIQP.")
    p.add_argument("-ic", "--intermediate-cardinality", type=int, default=10, metavar="<int>",
            help="Cardinality constraint used during intermediate MIQP.")
    p.add_argument("-d", "--directory", type=os.path.abspath, 
            default='.', metavar="<dir>",
            help="Directory to store results.")
    p.add_argument("-p", "--processors", type=int,
            default=None, metavar="<int>",
            help="Number of threads to use. Currently this only changes the CPLEX behaviour.")
    p.add_argument("-v", "--verbose", action="store_true",
            help="Be verbose.")
    args = p.parse_args()

    return args


def main():

    args = parse_args()
    mkdir_p(args.directory)
    logging_fname = os.path.join(args.directory, 'qfit_ligand.log') 
    logging.basicConfig(filename=logging_fname, level=logging.INFO)
    logger.info(' '.join(sys.argv))
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console_out)

    xmap = Volume.fromfile(args.xmap)
    if args.selection is None:
        ligand = Ligand.fromfile(args.ligand)
        if args.receptor is not None:
            receptor = Structure.fromfile(args.receptor).select('e', 'H', '!=')
        else:
            receptor = None
    else:
        # Extract ligand and rest of structure
        structure = Structure.fromfile(args.ligand)
        types = (str, int)
        chain, resi = [t(x) for t, x in zip(types, args.selection.split(','))]
        ligand_selection = structure.select('resi', resi, return_ind=True)
        ligand_selection &= structure.select('chain', chain, return_ind=True)
        receptor_selection = np.logical_not(ligand_selection)
        receptor = Structure(structure.data[receptor_selection], 
                             structure.coor[receptor_selection]).select('e', 'H', '!=')
        ligand = Ligand(structure.data[ligand_selection], structure.coor[ligand_selection]).select('altloc', ('', 'A', '1'))
    ligand.q.fill(1)

    builder = HierarchicalBuilder(
            ligand, xmap, args.resolution, receptor=receptor, 
            build=(not args.no_build), build_stepsize=args.build_stepsize, 
            stepsize=args.stepsize, local_search=(not args.no_local), 
            cardinality=args.intermediate_cardinality, 
            threshold=args.intermediate_threshold,
            directory=args.directory, scale=(not args.no_scale),
            threads=args.processors,
            )
    builder()

    builder._MIQP(maxfits=args.cardinality, threshold=args.threshold, exact=False)
    base = 'conformer'
    builder.write_results(base=base)

    nmax = min(len(builder._coor_set), 5)
    if args.threshold not in (0, None):
        nmax = min(nmax, int(1.0 / args.threshold))

    for n in xrange(1, nmax + 1):
        builder._MIQP(maxfits=n, threshold=args.threshold, exact=True)
        base = 'conformer_{:d}'.format(n)
        builder.write_results(base=base)

