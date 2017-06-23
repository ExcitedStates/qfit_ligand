"""Scale density based on receptor footprint."""

import argparse
import os.path

from .volume import Volume
from .structure import Structure
from .transformer import Transformer


def parse_args():

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("xmap", type=str,
            help="CCP4 density file with P1 symmetry.")
    p.add_argument("resolution", type=float,
            help="Resolution of electron density.")
    p.add_argument("structure", type=str,
            help="Structure to determine scaling factor.")
    p.add_argument("-s", "--simple", action="store_true",
            help="Make simple density to scale against.")
    p.add_argument("-o", "--output", type=str, default=None,
            help="Name of output file.")
    p.add_argument("-c", "--cutoff", type=float, default=None,
            help="All values below cutoff are set to 0 before scaling.")

    args = p.parse_args()
    if args.output is None:
        args.output = os.path.splitext(args.xmap)[0] + '_scaled.ccp4'

    return args


def main():

    args = parse_args()

    xmap = Volume.fromfile(args.xmap)
    print 'Xmap minimum:', xmap.array.min()
    structure = Structure.fromfile(args.structure)
    print 'Number of atoms:', structure.natoms
    model = Volume.zeros_like(xmap)
    if args.resolution < 3.0:
        rmask = 0.7 + (args.resolution - 0.6) / 3.0
    else:
        rmask = 0.5 * args.resolution
    smax = 1.0 / (2 * args.resolution)
    transformer = Transformer(structure, model, simple=args.simple,
            smax=smax)
    print 'Creating mask'
    transformer.mask(rmask)
    mask = model.array > 0
    transformer.reset()
    model.array.fill(0)
    print 'Initializing'
    transformer.initialize()
    print 'Scaling'
    transformer.density()
    xmap_masked = xmap.array[mask]
    model_masked = model.array[mask]
    model_masked_mean = model_masked.mean()
    xmap_masked_mean = xmap_masked.mean()
    model_masked -= model_masked_mean
    xmap_masked -= xmap_masked_mean


    scaling_factor = (model_masked * xmap_masked).sum() / \
            (xmap_masked * xmap_masked).sum()
    print "Scaling factor: {:.2f}".format(scaling_factor)
    xmap.array -= xmap_masked_mean
    xmap.array *= scaling_factor
    xmap.array += model_masked_mean
    xmap.array -= model.array

    if args.cutoff is not None:
        cutoff_mask = xmap.array < args.cutoff
        xmap.array[cutoff_mask] = 0
    xmap.tofile(args.output)
