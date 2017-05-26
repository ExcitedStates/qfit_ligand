import argparse
import time
from itertools import izip

import numpy as np

from .volume import Volume
from .structure import Ligand
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver


def parse_args():

    p = argparse.ArgumentParser(description="Determine occupancies from ligand set.")

    p.add_argument("xmap", type=str,
            help="CCP4 map file with P1 symmetry.")
    p.add_argument("resolution", type=float,
            help="Map resolution in angstrom.")
    p.add_argument("ligands", nargs="+", type=str,
            help="PDB files containing ligand.")

    p.add_argument("-c", "--cardinality", type=int, default=None,
            help="Cardinality constraint during MIQP. None by default.")
    p.add_argument("-t", "--threshold", type=float, default=None,
            help="Threshold constraint during MIQP. None by default.")
    p.add_argument("-ns", "--no-scale", action="store_true",
            help="Do not scale the density.")
    p.add_argument("-nn", "--no-normalize", action="store_true",
            help="Do not normalize the occupancies.")

    args = p.parse_args()

    return args


def solve():

    args = parse_args()

    xmap = Volume.fromfile(args.xmap)
    resolution = args.resolution
    ligand = Ligand.fromfile(args.ligands[0])
    coor_set = [Ligand.fromfile(fname).coor.copy() for fname in args.ligands]
    if resolution < 3.0:
        rmask = 0.7 + (resolution - 0.6) / 3.0
    else:
        rmask = 0.5 * resolution

    print 'Initializing'
    smax = 1 / (2 * resolution)
    model_map = Volume.zeros_like(xmap)
    transformer = Transformer(ligand, model_map, smax=smax, rmax=1)
    transformer.initialize()
    print 'Starting generation.'
    time0 = time.time()
    for coor in coor_set:
        ligand.coor[:] = coor
        transformer.mask(rmask)
    mask = model_map.array > 0
    model_map.array.fill(0)

    nvalues = mask.sum()
    target = xmap.array[mask]
    models = np.zeros((len(coor_set), nvalues))
    for n, coor in enumerate(coor_set):
        transformer.reset()
        ligand.coor[:] = coor
        transformer.density()
        models[n,:] = transformer.volume.array[mask]
    print 'Time required for density generation:', time.time() - time0

    #qpsolver = QPSolver(target, models, 
    #        scale=(not args.no_scale), normalize=(not args.normalize))
    #qpsolver.initialize()
    #qpsolver()

    miqpsolver = MIQPSolver(target, models, 
            scale=(not args.no_scale), 
            normalize=(not args.no_normalize))
    miqpsolver.initialize()
    miqpsolver(maxfits=args.cardinality, threshold=args.threshold)
    for fname, occ in izip(args.ligands, miqpsolver.occupancies):
        if occ >= 0.0005:
            print fname, occ


    for fname, occ in zip(args.ligands, miqpsolver.occupancies):
        print fname, occ

