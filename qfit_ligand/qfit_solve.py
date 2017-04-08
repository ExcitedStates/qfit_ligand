import argparse
import time

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

    args = p.parse_args()

    return args


def solve():

    args = parse_args()

    xmap = Volume.fromfile(args.xmap)
    resolution = args.resolution
    ligand = Ligand.fromfile(args.ligands[0])
    coor_set = [Ligand.fromfile(fname).coor.copy() for fname in args.ligands]

    print 'Initializing'
    smax = 1 / (2 * resolution)
    model_map = Volume.zeros_like(xmap)
    #print 'ligand'
    #print ligand.q.flags
    #print model_map.array.dtype, model_map.array.flags
    #model_map.array = np.ascontiguousarray(model_map.array, dtype=np.float64)
    #ligand.q = np.ascontiguousarray(ligand.q)
    #print ligand.q.flags
    #print model_map.array.dtype, model_map.array.flags
    transformer = Transformer(ligand, model_map, smax=smax, rmax=1)
    transformer.initialize()
    print 'Starting generation.'
    time0 = time.time()
    for coor in coor_set:
        ligand.coor[:] = coor
        transformer.mask()
    mask = transformer.volume.array > 0

    nvalues = mask.sum()
    target = xmap.array[mask]
    models = np.zeros((len(coor_set), nvalues))
    for n, coor in enumerate(coor_set):
        #transformer.volume.array.fill(0)
        transformer.reset()
        ligand.coor[:] = coor
        transformer.density()
        models[n,:] = transformer.volume.array[mask]
        print n
        #model_map.tofile('density_{:}.ccp4'.format(n))
    print 'Time required for density generation:', time.time() - time0

    qpsolver = QPSolver(target, models)
    qpsolver()
    for fname, occ in zip(args.ligands, qpsolver.occupancies):
        print fname, occ

