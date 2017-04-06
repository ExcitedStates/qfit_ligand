import argparse
import os.path

import numpy as np

from .structure import Ligand, Structure
from .samplers import (
        BondRotator, BondAngleRotator, PrincipalAxisRotator, GlobalRotator, 
        Translator, RotationSets, ClashDetector
        )
from .helpers import mkdir_p


def parse_args():

    p = argparse.ArgumentParser(description="Sample internal degrees of freedom of a ligand.")
    p.add_argument("ligands", type=str, nargs="+",
            help="Ligand files in PDB format.")
    p.add_argument("-r", "--receptor", dest="receptor", type=str, default=None,
            metavar="<file>",
            help="PDB file containing receptor for clash detection.")
    p.add_argument("-s", "--sample", dest="sample",type=str, default='', metavar="<dofs>",
            help="Internal degrees of freedom to sample.")
    p.add_argument("-sz", "--stepsize", type=float, default=20, metavar="<float>",
            help="Step size in degrees for internal degrees of freedom.")
    p.add_argument("-p", "--principal", dest="principal", action='store_true',
            help="Perform flips around the principal axes.")
    p.add_argument("-g", "--global", dest="globalrot", type=float, default=None,
            metavar="<float>",
            help="Perform a global rotational search at a specified angular interval.")
    p.add_argument("-t", "--translate", dest="translate", type=float, default=None,
            metavar="<float>",
            help="Perform a translational search within a specified box size.")
    p.add_argument("-d", "--directory", dest="directory", type=os.path.abspath, 
            default='.', metavar="<dir>",
            help="Directory to store results.")
    args = p.parse_args()

    return args


def main():

    args = parse_args()
    ligand = Ligand.fromfile(args.ligands[0])
    coor_set = [Ligand.fromfile(fname).coor.copy() for fname in args.ligands]
    receptor = None
    if args.receptor is not None:
        receptor = Structure.fromfile(args.receptor)

    mkdir_p(args.directory)

    # Rotate around bond and angles.
    dofs = args.sample.split(',')
    for dof in dofs:
        atoms = dof.split('-')
        if len(atoms) == 2:
            sampler = BondRotator
            sampling_range = np.deg2rad(np.arange(0, 360, args.stepsize))
        elif len(atoms) == 3:
            sampler = BondAngleRotator
            sampling_range = np.deg2rad(np.linspace(-5 , 5, 11, endpoint=True))
        else:
            continue

        new_coor_set = []
        for coor in coor_set:
            ligand.coor[:] = coor
            rotator = sampler(ligand, *atoms)
            for angle in sampling_range:
                rotator(angle)
                new_coor_set.append(rotator.ligand.coor.copy())
        coor_set = new_coor_set
 
    # Rotate around the principal axes.
    if args.principal:
        new_coor_set = []
        for coor in coor_set:
            for axis in xrange(3):
                ligand.coor[:] = coor
                rotator = PrincipalAxisRotator(ligand)
                rotator(np.deg2rad(180), axis)
                new_coor_set.append(rotator.ligand.coor.copy())
        coor_set += new_coor_set

    # Sample general rotations.
    if args.globalrot is not None:
        rotation_set = RotationSets.get_set(args.globalrot)
        print 'Number of rotations sampled: ', rotation_set.shape[0]
        new_coor_set = []
        for coor in coor_set:
            ligand.coor[:] = coor
            rotator = GlobalRotator(ligand)
            for rotmat in rotation_set:
                rotator(rotmat)
                new_coor_set.append(rotator.ligand.coor.copy())
        coor_set = new_coor_set

    # Sample translations
    if args.translate is not None:
        new_coor_set = []
        #translation_set = np.arange(-args.translate, args.translate + 1)
        translation_set = np.linspace(-args.translate, args.translate, 4 * args.translate + 1, endpoint=True)
        print 'Translations sampled: ', translation_set.size ** 3
        for coor in coor_set:
            ligand.coor[:] = coor
            translator = Translator(ligand)
            for x in  translation_set:
                for y in translation_set:
                    for z in translation_set:
                        translator((x, y , z))
                        new_coor_set.append(translator.ligand.coor.copy())
        coor_set = new_coor_set

    # Write out ligands to file
    if args.receptor is not None:
        cd = ClashDetector(receptor.coor, radius=2)

    n = 1
    for coor in coor_set:
        ligand.coor[:] = coor
        if args.receptor is not None:
            not_clashing = cd(ligand.coor) == 0 and not ligand.clashes()
        else:
            not_clashing = not ligand.clashes()
        if not_clashing:
            ligand.tofile(os.path.join(args.directory, 'ligand_{:d}.pdb'.format(n)))
            n += 1
