from __future__ import division

import gzip
import logging
import operator
import os
from collections import defaultdict, Sequence
from itertools import izip, product

logger = logging.getLogger(__name__)

import numpy as np
from scipy.misc import comb as sp_comb
from scipy.spatial.distance import pdist as sp_pdist, squareform as sp_squareform

from .elements import ELEMENTS
from .residues import RESIDUES
from .samplers import Rz, aa_to_rotmat

class Structure(object):

    attributes = 'record atomid atomname resn altloc chain resi icode x y z q b e charge'.split()
    dtype = [('record', np.str_, 6), ('atomid', np.int32),
             ('atomname', np.str_, 4), ('altloc', np.str_, 1),
             ('resn', np.str_, 4), ('chain', np.str_, 2),
             ('resi', np.int32), ('icode', np.str_, 1),
             ('q', np.float64), ('b', np.float64),
             ('e', np.str_, 2), ('charge', np.str_, 2),
             ]

    def __init__(self, data, coor, resolution=None):
        self.natoms = data['atomid'].size
        self.data = data
        self.coor = coor
        self.x = self.coor[:, 0]
        self.y = self.coor[:, 1]
        self.z = self.coor[:, 2]
        for attr in self.attributes:
            if attr not in list('xyz'):
                setattr(self, attr, data[attr])
        self._connectivity = None
        self.resolution = resolution

    @classmethod
    def fromfile(cls, fname):
        pdbfile = PDBFile.read(fname)
        dd = pdbfile.coor
        natoms = len(dd['atomid'])
        data = np.zeros(natoms, dtype=cls.dtype)
        for attr in cls.attributes:
            if attr not in list('xyz'):
                data[attr] = dd[attr]
        # Make the coordinates a separate array as they will be changed a lot
        coor = np.asarray(zip(dd['x'], dd['y'], dd['z']), dtype=np.float64)
        return cls(data, coor, pdbfile.resolution)

    def tofile(self, fname):
        PDBFile.write(fname, self)

    def rmsd(self, structure):
        #diff = (self.coor - structure.coor).ravel()
        #return np.sqrt(3 * np.inner(diff, diff) / diff.size)
        return np.sqrt(((self.coor - structure.coor) ** 2).mean() * 3)

    def combine(self, structure):
        if self.resolution == structure.resolution:
            resolution = self.resolution
        else:
            resolution = None

        return self.__class__(np.hstack((self.data, structure.data)),
                              np.vstack((self.coor, structure.coor)), resolution)

    def select(self, identifier, values, loperator='==', return_ind=False):
        """A simple way of selecting atoms"""
        if loperator in ('==', '!='):
            oper = operator.eq
        elif loperator == '<':
            oper = operator.lt
        elif loperator == '>':
            oper = operator.gt
        elif loperator == '>=':
            oper = operator.ge
        elif loperator == '<=':
            oper = operator.le
        else:
            raise ValueError('Logic operator not recognized.')

        if not isinstance(values, Sequence) or isinstance(values, basestring):
            values = (values,)

        selection = oper(self.data[identifier], values[0])
        if len(values) > 1:
            for v in values[1:]:
                    selection |= oper(self.data[identifier], v)
        if loperator == '!=':
            np.logical_not(selection, out=selection)

        if return_ind:
            return selection
        else:
            return self.__class__(self.data[selection], self.coor[selection], self.resolution)

    def _get_property(self, ptype):
        elements, ind = np.unique(self.data['e'], return_inverse=True)
        values = []
        for e in elements:
            try:
                value = getattr(ELEMENTS[e.capitalize()], ptype)
            except KeyError:
                logger.warning("Unknown element {:s}. Using Carbon parameter instead.".format(e))
                value = getattr(ELEMENTS['C'], ptype)
            values.append(value)
        out = np.asarray(values, dtype=np.float64)[ind]
        return out

    @property
    def covalent_radius(self):
        return self._get_property('covrad')

    @property
    def vdw_radius(self):
        return self._get_property('vdwrad')


class Residue(Structure):

    def __init__(self, data, coor, resolution=None):
        super(Residue, self).__init__(data, coor, resolution)
        resnames = set(self.resn)
        if len(resnames) > 1:
            raise ValueError("Input is more than 1 residue")
        resname = resnames.pop()
        self._residue_data = RESIDUES[resname]
        self.nchi = self._residue_data['nchi']
        self.nrotamers = len(self._residue_data['rotamers'])
        self.rotamers = self._residue_data['rotamers']

        self._init_clash_detection()

    def _init_clash_detection(self):
        # Setup the condensed distance based arrays for clash detection and fill them
        self._ndistances = self.natoms * (self.natoms - 1) / 2
        self._clash_mask = np.ones(self._ndistances, bool)
        self._clash_radius2 = np.zeros(self._ndistances, float)
        radii = self.covalent_radius
        bonds = self._residue_data['bonds']
        offset = sp_comb(self.natoms, 2)
        for i in xrange(self.natoms - 1):
            starting_index = int(offset - sp_comb(self.natoms - i, 2)) - i - 1
            atomname1 = self.atomname[i]
            covrad1 = radii[i]
            for j in xrange(i + 1, self.natoms):
                bond1 = [atomname1, self.atomname[j]]
                bond2 = bond1[::-1]
                covrad2 = radii[j]
                index = starting_index + j
                self._clash_radius2[index] = covrad1 + covrad2 + 0.5
                if bond1 in bonds or bond2 in bonds:
                    self._clash_mask[index] = False
        self._clash_radius2 *= self._clash_radius2
        self._clashing = np.zeros(self._ndistances, bool)
        self._dist2_matrix = np.empty(self._ndistances, float)

        # All atoms are active from the start
        self.active = np.ones(self.natoms, bool)
        self._active_mask = np.ones(self._ndistances, bool)

    def set_active(self, selection=None, value=True):
        if selection is None:
            self.active.fill(value)
        else:
            self.active[selection] = value
        offset = sp_comb(self.natoms, 2) - 1
        for i, active in enumerate(self.active[:-1]):
            starting_index = int(offset - sp_comb(self.natoms - i, 2)) - i
            end = starting_index + self.natoms - (i + 1)
            self._active_mask[starting_index: end] = active

    def activate(self, selection=None):
        self.set_active(selection)

    def deactivate(self, selection=None):
        self.set_active(selection, value=False)

    def clashes(self):

        """Checks if there are any internal clashes.
        Deactivated atoms are not taken into account.
        """

        dm = self._dist2_matrix
        coor = self.coor
        dot = np.dot
        k = 0
        for i in xrange(self.natoms - 1):
            u = coor[i]
            for j in xrange(i + 1, self.natoms):
                u_v = u - coor[j]
                dm[k] = dot(u_v, u_v)
                k += 1
        np.less_equal(dm, self._clash_radius2, self._clashing)
        self._clashing &= self._clash_mask
        self._clashing &= self._active_mask
        nclashes = self._clashing.sum()
        return nclashes

    def get_chi(self, chi_index):
        atoms = self._residue_data['chi'][chi_index]
        selection = self.select('atomname', atoms, return_ind=True).nonzero()[0]
        ordered_sel = []
        for atom in atoms:
            for sel in selection:
                if atom == self.atomname[sel]:
                    ordered_sel.append(sel)
                    break
        coor = self.coor[ordered_sel]
        b1 = coor[0] - coor[1]
        b2 = coor[3] - coor[2]
        b3 = coor[2] - coor[1]
        n1 = np.cross(b3, b1)
        n2 = np.cross(b3, b2)
        m1 = np.cross(n1, n2)

        norm = np.linalg.norm
        normfactor = norm(n1) * norm(n2)
        sinv = norm(m1) / normfactor
        cosv = np.inner(n1, n2) / normfactor
        angle = np.rad2deg(np.arctan2(sinv, cosv))
        # Check sign of angle
        u = np.cross(n1, n2)
        if np.inner(u, b3) < 0:
            angle *= -1
        return angle

    def set_chi(self, chi_index, value):
        atoms = self._residue_data['chi'][chi_index]
        selection = self.select('atomname', atoms, return_ind=True)
        coor = self.coor[selection]
        origin = coor[1].copy()
        coor -= origin
        zaxis = coor[2]
        zaxis /= np.linalg.norm(zaxis)
        yaxis = coor[0] - np.inner(coor[0], zaxis) * zaxis
        yaxis /= np.linalg.norm(yaxis)
        xaxis = np.cross(yaxis, zaxis)
        backward = np.asmatrix(np.zeros((3, 3), float))
        backward[0] = xaxis
        backward[1] = yaxis
        backward[2] = zaxis
        forward = backward.T

        atoms_to_rotate = self._residue_data['chi-rotate'][chi_index]
        selection = self.select('atomname', atoms_to_rotate, return_ind=True)
        coor_to_rotate = np.dot(self.coor[selection] - origin, backward.T)
        rotation = Rz(np.deg2rad(value - self.get_chi(chi_index)))

        R = forward * rotation
        self.coor[selection] = np.dot(coor_to_rotate, R.T) + origin


class _RecursiveNeighborChecker(object):

    """ Get all neighbors starting from a root and neighbor

    Used to detect which atoms to rotate given a connectivity matrix and two
    atoms along which to rotate.
    """

    def __init__(self, root, neighbor, connectivity):

        self.root = root
        self.neighbors = [root]
        self._find_neighbors_recursively(neighbor, connectivity)
        self.neighbors.remove(root)

    def _find_neighbors_recursively(self, neighbor, conn):
        self.neighbors.append(neighbor)
        neighbors = np.flatnonzero(conn[neighbor])
        for n in neighbors:
            if n not in self.neighbors:
                self._find_neighbors_recursively(n, conn)


class Ligand(Structure):

    """Ligand class is like a Structure, but has a topology added to it."""

    def __init__(self, *args, **kwargs):
        super(Ligand, self).__init__(*args, **kwargs)
        self._get_connectivity()

    def _get_connectivity(self):
        """Determine connectivity matrix of ligand and associated distance
        cutoff matrix for later clash detection.
        """

        dist_matrix = sp_squareform(sp_pdist(self.coor))
        covrad = self.covalent_radius
        natoms = self.natoms
        cutoff_matrix = np.repeat(covrad, natoms).reshape(natoms, natoms)
        # Add 0.5 A to give covalently bound atoms more room
        cutoff_matrix = cutoff_matrix + cutoff_matrix.T + 0.5
        connectivity_matrix = (dist_matrix < cutoff_matrix)
        # Atoms are not connected to themselves
        np.fill_diagonal(connectivity_matrix, False)
        self.connectivity = connectivity_matrix
        self._cutoff_matrix = cutoff_matrix

    def clashes(self):
        """Checks if there are any internal clashes.
        Atoms with occupancy of 0 are not taken into account.
        """
        dist_matrix = sp_squareform(sp_pdist(self.coor))
        mask = np.logical_not(self.connectivity)
        occupancy_matrix = (self.q.reshape(1, -1) * self.q.reshape(-1, 1)) > 0
        mask &= occupancy_matrix
        np.fill_diagonal(mask, False)
        clash_matrix = dist_matrix < self._cutoff_matrix
        if np.any(np.logical_and(clash_matrix, mask)):
            return True
        return False

    def bonds(self):
        """Print bonds"""
        indices = np.nonzero(self.connectivity)
        for a, b in izip(*indices):
            print self.atomname[a], self.atomname[b]

    def ring_paths(self):
        def ring_path(T, v1, v2):
            v1path = []
            v = v1
            while v is not None:
                v1path.append(v)
                v = T[v]
            v = v2
            v2path = []
            while v not in v1path:
                v2path.append(v)
                v = T[v]
            ring = v1path[0:v1path.index(v) + 1] + v2path
            return ring
        ring_paths = []
        T = {}
        conn = self.connectivity
        for root in xrange(self.natoms):
            if root in T:
                continue
            T[root] = None
            fringe = [root]
            while fringe:
                a = fringe[0]
                del fringe[0]
                # Scan the neighbors of a
                for n in np.flatnonzero(conn[a]):
                    if n in T and n == T[a]:
                        continue
                    elif n in T and (n not in fringe):
                        ring_paths.append(ring_path(T, a, n))
                    elif n not in fringe:
                        T[n] = a
                        fringe.append(n)
        return ring_paths

    def rotatable_bonds(self):

        """Determine all rotatable bonds.

        A rotatable bond is currently described as two neighboring atoms with
        more than 1 neighbor and which are not part of the same ring.
        """

        conn = self.connectivity
        rotatable_bonds = []
        rings = self.ring_paths()
        for atom in xrange(self.natoms):
            neighbors = np.flatnonzero(conn[atom])
            if len(neighbors) == 1:
                continue
            for neighbor in neighbors:
                neighbor_neighbors = np.flatnonzero(conn[neighbor])
                new_bond = False
                if len(neighbor_neighbors) == 1:
                    continue
                # Check whether the two atoms are part of the same ring.
                same_ring = False
                for ring in rings:
                    if atom in ring and neighbor in ring:
                        same_ring = True
                        break
                if not same_ring:
                    new_bond = True
                    for b in rotatable_bonds:
                        # Check if we already found this bond.
                        if atom in b and neighbor in b:
                            new_bond = False
                            break
                if new_bond:
                    rotatable_bonds.append((atom, neighbor))
        return rotatable_bonds

    def rigid_clusters(self):

        """Find rigid clusters / seeds in the molecule.

        Currently seeds are either rings or terminal ends of the molecule, i.e.
        the last two atoms.
        """

        conn = self.connectivity
        rings = self.ring_paths()
        clusters = []
        for root in xrange(self.natoms):
            # Check if root is Hydrogen
            element = self.e[root]
            if element == 'H':
                continue
            # Check if root has already been clustered
            clustered = False
            for cluster in clusters:
                if root in cluster:
                    clustered = True
                    break
            if clustered:
                continue
            # If not, start new cluster
            cluster = [root]
            # Check if atom is part of a ring, if so add all atoms. This
            # step combines multi-ring systems.
            ring_atom = False
            for atom, ring in product(cluster, rings):
                if atom in ring:
                    ring_atom = True
                    for a in ring:
                        if a not in cluster:
                            cluster.append(a)

            # If root is not part of a ring, check if it is connected to a
            # terminal heavy atom.
            if not ring_atom:
                neighbors = np.flatnonzero(conn[root])
                for n in neighbors:
                    if self.e[n] == 'H':
                        continue
                    neighbor_neighbors = np.flatnonzero(conn[n])
                    # Hydrogen neighbors don't count
                    hydrogen_neighbors = (self.e[neighbor_neighbors] == 'H').sum()
                    if len(neighbor_neighbors) - hydrogen_neighbors == 1:
                        cluster.append(n)

            if len(cluster) > 1:
                clusters.append(cluster)
        # Add all left-over single unclustered atoms
        for atom in xrange(self.natoms):
            found = False
            for cluster in clusters:
                if atom in cluster:
                    found = True
                    break
            if not found:
                clusters.append([atom])
        return clusters

    def atoms_to_rotate(self, bond_or_root, neighbor=None):
        """Return indices of atoms to rotate given a bond."""

        if neighbor is None:
            root, neighbor = bond_or_root
        else:
            root = bond_or_root

        neighbors = [root]
        atoms_to_rotate = self._find_neighbors_recursively(neighbor, neighbors, self.connectivity)
        atoms_to_rotate.remove(root)
        return atoms_to_rotate

    def _find_neighbors_recursively(self, neighbor, neighbors, conn):
        neighbors.append(neighbor)
        local_neighbors = np.flatnonzero(conn[neighbor])
        for ln in local_neighbors:
            if ln not in neighbors:
                self._find_neighbors_recursively(ln, neighbors, conn)
        return neighbors

    def rotate_along_bond(self, bond, angle):
        atoms_to_rotate = self.atoms_to_rotate(bond)
        origin = self.coor[bond[0]]
        end = self.coor[bond[1]]
        axis = end - origin
        axis /= np.linalg.norm(axis)

        coor = self.coor[atoms_to_rotate]
        coor -= origin
        rotmat = aa_to_rotmat(axis, angle)
        self.coor[atoms_to_rotate] = np.dot(coor, rotmat.T) + origin


class BondOrder(object):

    """Determine bond rotation order given a ligand and root."""

    def __init__(self, ligand, atom):
        self.ligand = ligand
        self._conn = self.ligand.connectivity
        self.clusters = self.ligand.rigid_clusters()
        self.bonds = self.ligand.rotatable_bonds()
        self._checked_clusters = []
        self.order = []
        self.depth = []
        self._bondorder(atom)

    def _bondorder(self, atom, depth=0):
        for cluster in self.clusters:
            if atom in cluster:
                break
        if cluster in self._checked_clusters:
            return
        depth += 1
        self._checked_clusters.append(cluster)
        neighbors = []
        for atom in cluster:
            neighbors += np.flatnonzero(self._conn[atom]).tolist()

        for n in neighbors:
            for ncluster in self.clusters:
                if n in ncluster:
                    break
            if ncluster == cluster:
                continue
            for b in self.bonds:
                if b[0] in cluster and b[1] in ncluster:
                    bond = (b[0], b[1])
                elif b[1] in cluster and b[0] in ncluster:
                    bond = (b[1], b[0])
                try:
                    if (bond[1], bond[0]) not in self.order and bond not in self.order:
                        self.order.append(bond)
                        self.depth.append(depth)
                except UnboundLocalError:
                    pass
            self._bondorder(n, depth)


class PDBFile(object):

    @classmethod
    def read(cls, fname):
        cls.coor = defaultdict(list)
        cls.resolution = None
        if fname.endswith('.gz'):
            fopen = gzip.open
            mode = 'rb'
        else:
            fopen = open
            mode = 'r'

        with fopen(fname, mode) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    values = CoorRecord.parse_line(line)
                    for field in CoorRecord.fields:
                        cls.coor[field].append(values[field])
                elif line.startswith('MODEL'):
                    raise NotImplementedError("MODEL record is not implemented.")
                elif line.startswith('REMARK   2 RESOLUTION'):
                    cls.resolution = float(line.split()[-2])
        return cls

    @staticmethod
    def write(fname, structure):
        with open(fname, 'w') as f:
            for fields in izip(*[getattr(structure, x) for x in CoorRecord.fields]):
                if len(fields[-2]) == 2 or len(fields[2]) == 4:
                    f.write(CoorRecord.line2.format(*fields))
                else:
                    f.write(CoorRecord.line1.format(*fields))


class ModelRecord(object):
    fields = 'record modelid'
    columns = [(0, 6), (11, 15)]
    dtypes = (str, int)
    line = '{:6s}' + ' ' * 5 + '{:6d}\n'

    @classmethod
    def parse_line(cls, line):
        values = {}
        for field, column, dtype in izip(cls.fields, cls.columns, cls.dtypes):
            values[field] = dtype(line[slice(*column)].strip())
        return values


class CoorRecord(object):
    fields = 'record atomid atomname altloc resn chain resi icode x y z q b e charge'.split()
    columns = [(0, 6), (6, 11), (12, 16), (16, 17), (17, 20), (21, 22),
               (22, 26), (26, 27), (30, 38), (38, 46), (46, 54), (54, 60),
               (60, 66), (76, 78), (78, 80),
               ]
    dtypes = (str, int, str, str, str, str, int, str, float, float, float,
              float, float, str, str)
    line1 = ('{:6s}{:5d}  {:3s}{:1s}{:3s} {:1s}{:4d}{:1s}   '
             '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}' + ' ' * 10 + '{:>2s}{:>2s}\n')
    line2 = ('{:6s}{:5d} {:<4s}{:1s}{:3s} {:1s}{:4d}{:1s}   '
             '{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}' + ' ' * 10 + '{:>2s}{:2s}\n')

    @classmethod
    def parse_line(cls, line):
        values = {}
        for field, column, dtype in izip(cls.fields, cls.columns, cls.dtypes):
            values[field] = dtype(line[slice(*column)].strip())
        return values
