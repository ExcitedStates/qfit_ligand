"""Automatically build a multiconformer residue"""

import argparse
import os.path
import sys
import logging
import time
from itertools import izip
from string import ascii_uppercase
logger = logging.getLogger(__name__)

import numpy as np

from .sidechain_builder import SideChainBuilder
from .structure import Residue, Structure
from .volume import Volume
from .helpers import mkdir_p
from .scaler import MapScaler


def parse_args():

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str,
            help="X-ray density map in CCP4 format.")
    p.add_argument("resolution", type=float,
            help="Map resolution in angstrom.")
    p.add_argument("structure", type=str,
            help="")
    p.add_argument('--selection', default=None, type=str,
            help="Chain and residue id for residue in structure, e.g. A,105.")
    p.add_argument("-ns", "--no-scale", action="store_true",
            help="Do not scale density.")
    p.add_argument("-dc", "--density-cutoff", type=float, default=0.5,
            help="Density value cutoff in sigma of X-ray map. Values below this threshold are set to 0 after scaling to absolute density.")

    p.add_argument("-b", "--build-stepsize", type=int, default=1, metavar="<int>",
            help="Number of internal degrees that are sampled/build per iteration.")
    p.add_argument("-s", "--stepsize", type=float, default=5, metavar="<float>",
            help="Stepsize for dihedral angle sampling in degree.")

    p.add_argument("-c", "--cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during MIQP.")
    p.add_argument("-t", "--threshold", type=float, default=None, metavar="<float>",
            help="Treshold constraint used during MIQP.")
    p.add_argument("-it", "--intermediate-threshold", type=float, default=0.01, metavar="<float>",
            help="Threshold constraint during intermediate MIQP.")
    p.add_argument("-ic", "--intermediate-cardinality", type=int, default=5, metavar="<int>",
            help="Cardinality constraint used during intermediate MIQP.")

    p.add_argument("-d", "--directory", type=os.path.abspath, default='.', metavar="<dir>",
            help="Directory to store results.")
    p.add_argument("--debug", action="store_true",
                   help="Write intermediate structures to file for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
            help="Be verbose.")
    args = p.parse_args()

    return args


def main():

    args = parse_args()
    mkdir_p(args.directory)
    time0 = time.time()

    # Setup logger
    logging_fname = os.path.join(args.directory, 'qfit_residue.log')
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(filename=logging_fname, level=level)
    logger.info(' '.join(sys.argv))
    logger.info(time.strftime("%c %Z"))
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(level)
        logging.getLogger('').addHandler(console_out)


    # Extract residue and prepare it
    structure = Structure.fromfile(args.structure)
    types = (str, int)
    chain, resi = [t(x) for t, x in izip(types, args.selection.split(','))]
    residue = structure.select('resi', resi).select('chain', chain)
    residue = Residue(residue.data, residue.coor)
    residue.q.fill(1)
    logger.info("Residue: {}".format(residue.resn[0]))
    structure = structure.select('resi', resi, '!=')
    # Prepare X-ray map
    xmap = Volume.fromfile(args.xmap).fill_unit_cell()
    if not args.no_scale:
        scaler = MapScaler(xmap, mask_radius=1, cutoff=args.density_cutoff)
        scaler(structure.select('record', 'ATOM'))

    builder = SideChainBuilder(
        residue, xmap, args.resolution, receptor=structure, directory=args.directory,
        build_stepsize=args.build_stepsize, stepsize=args.stepsize,
    )
    builder()
    builder.write_results()
    #conformers = builder.get_conformers()
