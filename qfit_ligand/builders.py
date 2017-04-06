from __future__ import division
import subprocess
import glob
import os
import itertools
import logging
logger = logging.getLogger(__name__)

import numpy as np

from .structure import Ligand, BondOrder
from .volume import Volume
from .samplers import GlobalRotator, Translator, BondRotator, RotationSets, ClashDetector
from .transformer import Transformer
from .solvers import QPSolver, MIQPSolver
from .helpers import DJoiner


class HierarchicalBuilder(object):

    """Build a multi-conformer ligand hierarchically."""

    def __init__(self, ligand, xmap, resolution, receptor=None, local_search=True, 
            stepsize=20, build_stepsize=2, directory='.', clean=True, roots=None):
        self.ligand = ligand
        self.stepsize= stepsize
        self.build_stepsize = build_stepsize
        self.directory = directory
        self.xmap = Volume.fromfile(xmap)
        self.resolution = resolution
        self.local_search = local_search
        self.clean = clean
        self.receptor = receptor
        if self.resolution < 3.0:
            self._rmask = 0.7 + (self.resolution - 0.6) / 3.0
        else:
            self._rmask = 0.5 * self.resolution

        for k, v in locals().iteritems():
            logger.info("{:} = {:}".format(k, v))

        self._trans_box = [(-0.2, 0.21, 0.1), (-0.2, 0.21, 0.1), (-0.2, 0.21, 0.1)]
        self._sampling_range = np.deg2rad(np.arange(0, 360, self.stepsize))
        self._djoiner = DJoiner(directory)

        if self.receptor is not None:
            self._cd = ClashDetector(self.ligand, self.receptor, 0.75)

        self._rigid_clusters = self.ligand.rigid_clusters()
        if roots is None:
            self._clusters_to_sample = [
                    cluster for cluster in self._rigid_clusters if len(cluster) > 1]
        else:
            self._clusters_to_sample = []
            for root in roots:
                for cluster in self._rigid_clusters:
                    if root in cluster:
                        self._clusters_to_sample.append(cluster)
        logger.info("Number of clusters to sample: {:}".format(len(self._clusters_to_sample)))
        self._starting_coor_set = [ligand.coor.copy()]
        self._coor_set = []
        self._all_coor_set = []
        self.conformers = []
        self._occupancies = []

        # Initialize density creation
        smax = 1 / (2 * self.resolution)
        model_map = Volume.zeros_like(self.xmap)
        self._transformer = Transformer(self.ligand, model_map, smax=smax, rmax=3)


    def __call__(self):

        for self._cluster_index, self._cluster in enumerate(self._clusters_to_sample):
            self._iteration = 0
            self._coor_set = list(self._starting_coor_set)
            logger.debug("Cluster index: {:}".format(self._cluster_index))
            logger.info("Iteration: {:}".format(self._iteration))
            logger.debug("Number of conformers: {:}".format(len(self._coor_set)))
            if self.local_search:
                self._local_search()
                self._convert()
                self._QP()
                self._update_conformers()
                self._convert()
                self._MIQP()
                self._update_conformers()
            self._build_ligand()
            self._all_coor_set += self._coor_set
            logger.info("Number of conformers: {:}".format(len(self._coor_set)))
            logger.info("Number of final conformers: {:}".format(len(self._all_coor_set)))

        self._coor_set = self._all_coor_set
        self._cluster_index += 1
        self._iteration = 0
        self._convert()
        self._QP()
        self._update_conformers()
        self._convert()
        self._MIQP()
        self._update_conformers()

        self._write_results()
        if self.clean:
            self._clean()

    def _clashing(self):
        if self.receptor is None:
            return self.ligand.clashes()
        else:
            return self.ligand.clashes() or self._cd() != 0

    def _local_search(self):
        """Perform a local rigid body search on the cluster."""

        logger.info("Performing local search.")

        # Set occupancies of rigid cluster and its direct neighboring atoms to
        # 1 for clash detection and MIQP
        self.ligand.q.fill(0)
        self.ligand.q[self._cluster] = 1
        for atom in self._cluster:
            self.ligand.q[self.ligand.connectivity()[atom]] = 1
        center = self.ligand.coor[self._cluster].mean()
        new_coor_set = []
        for coor in self._coor_set:
            self.ligand.coor[:] = coor
            rotator = GlobalRotator(self.ligand, center=center)
            for rotmat in RotationSets.get_local_set():
                rotator(rotmat)
                translator = Translator(self.ligand)
                iterator = itertools.product(*[
                    np.arange(*trans) for trans in self._trans_box])
                for translation in iterator:
                        translator(translation)
                        if not self._clashing():
                            new_coor_set.append(self.ligand.coor.copy())
        self._coor_set = new_coor_set

    def _build_ligand(self):
        """Build up the ligand hierarchically."""

        logger.info("Building up ligand.")
        # Sampling order of bonds
        bond_order = BondOrder(self.ligand, self._cluster[0])
        bonds = bond_order.order
        depths = bond_order.depth
        nbonds = len(bonds)
        starting_bond_index = 0
        finished_building = False
        while not finished_building:
            end_bond_index = min(starting_bond_index + self.build_stepsize, nbonds)
            for bond_index in xrange(starting_bond_index, end_bond_index):
                # Set the occupancies of build clusters and their direct
                # neighbors to 1 for clash detection and MIQP.
                nbonds_sampled = bond_index + 1
                self.ligand.q.fill(0)
                for cluster in self._rigid_clusters:
                    for sampled_bond in bonds[:nbonds_sampled]:
                        if sampled_bond[0] in cluster or sampled_bond[1] in cluster:
                            self.ligand.q[cluster] = 1
                            for atom in cluster:
                                self.ligand.q[self.ligand.connectivity()[atom]] = 1

                # Sample by rotating around a bond
                bond = bonds[bond_index]
                atoms = [self.ligand.atomname[bond[0]], self.ligand.atomname[bond[1]]]
                new_coor_set = []
                for coor in self._coor_set:
                    self.ligand.coor[:] = coor
                    rotator = BondRotator(self.ligand, *atoms)
                    for angle in self._sampling_range:
                        rotator(angle)
                        if not self._clashing():
                            new_coor_set.append(self.ligand.coor.copy())
                self._coor_set = new_coor_set

                # Perform an MIQP if either the end bond index has been reached
                # or if the end of a sidechain has been reached, i.e. if the
                # next depth level is equal or smaller than current depth. If
                # we step out of bonds in the depths list it means we are done.
                end_iteration = (nbonds_sampled == end_bond_index)
                try:
                    end_sidechain = depths[bond_index] >= depths[bond_index + 1]
                except IndexError:
                    finished_building = True
                if end_iteration or end_sidechain:
                    self._iteration += 1
                    self._convert()
                    self._QP()
                    self._update_conformers()
                    self._convert()
                    self._MIQP()
                    self._update_conformers()

                    # Stop this building iteration and move on to next
                    starting_bond_index += 1
                    if end_sidechain:
                        starting_bond_index = nbonds_sampled
                    break

    def _convert(self):

        print 'Converting to densities ({:})'.format(len(self._coor_set))
        self._transformer.volume.array.fill(0)
        for coor in self._coor_set:
            self.ligand.coor[:] = coor
            self._transformer.mask(self._rmask)
        mask = self._transformer.volume.array > 0
        self._transformer.volume.array.fill(0)

        nvalues = mask.sum()
        self._target = self.xmap.array[mask]
        self._models = np.zeros((len(self._coor_set), nvalues), dtype=np.float64)
        for n, coor in enumerate(self._coor_set):
            self.ligand.coor[:] = coor
            self._transformer.density()
            self._models[n] = self._transformer.volume.array[mask]
            self._transformer.reset()

    def _QP(self):
        #ligand_files_fname = self._djoiner('_ligand_files_{:d}_{:d}.txt').format(
        #    self._cluster_index, self._iteration)
        #with open(ligand_files_fname, 'w') as f:
        #    ligand_fname = self._djoiner('intermediate_{:d}_{:d}_{:d}.pdb')
        #    for n, coor in enumerate(self._coor_set):
        #        self.ligand.coor[:] = coor
        #        fname = ligand_fname.format(self._cluster_index, self._iteration, n)
        #        self.ligand.tofile(fname)
        #        f.write(fname)
        #        f.write('\n')
        print 'QP'
        qpsolver = QPSolver(self._target, self._models)
        print 'Initializing'
        qpsolver.initialize()
        print 'Solving'
        qpsolver()
        self._occupancies = qpsolver.occupancies
        print 'Done'

    def _MIQP(self):
        print 'MIQP'
        miqpsolver = MIQPSolver(self._target, self._models)
        print 'Initializing'
        miqpsolver.initialize()
        print 'Solving'
        miqpsolver(maxfits=5)
        self._occupancies = miqpsolver.occupancies
        print 'Done'

    def _update_conformers(self):
        print 'Number of conformers before: ', len(self._coor_set)
        print self._occupancies
        new_coor_set = []
        cutoff = min(1 / len(self._coor_set), 0.01)
        for n, coor in enumerate(self._coor_set):
            if self._occupancies[n] >= cutoff:
                new_coor_set.append(coor)
        self._coor_set = new_coor_set
        self._occupancies = self._occupancies[self._occupancies >= cutoff]
        print 'Number of conformers: ', len(self._coor_set)

    def _write_results(self):
        logger.info("Writing results to file.")
        fname_base = self._djoiner('conformer_{:d}.pdb')
        for n, (coor, occ) in enumerate(zip(self._coor_set, self._occupancies), start=1):
            self.ligand.q.fill(occ)
            self.ligand.coor[:] = coor
            self.ligand.tofile(fname_base.format(n))

    def _clean(self):
        """Remove all files"""
        logger.info("Removing files.")
        files_to_remove = glob.glob(self._djoiner('_*'))
        for fname in files_to_remove:
            os.remove(fname)
