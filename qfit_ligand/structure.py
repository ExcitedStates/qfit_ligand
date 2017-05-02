import os
from collections import defaultdict, Sequence
import operator
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy.spatial.distance import pdist as sp_pdist, squareform as sp_squareform

from .elements import ELEMENTS

class Structure(object):

    attributes = 'record atomid atomname resn altloc chain resi icode x y z q b e charge'.split()
    dtype = [('record', np.str_, 6), ('atomid', np.int32),
             ('atomname', np.str_, 4), ('altloc', np.str_, 1),
             ('resn', np.str_, 4), ('chain', np.str_, 2),
             ('resi', np.int32), ('icode', np.str_, 1),
             ('q', np.float64), ('b', np.float64),
             ('e', np.str_, 2), ('charge', np.str_, 2),
             ]

    def __init__(self, data, coor):
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

    @classmethod
    def fromfile(cls, fname):
        dd = PDBFile.read(fname).coor
        natoms = len(dd['atomid'])
        data = np.zeros(natoms, dtype=cls.dtype)
        for attr in cls.attributes:
            if attr not in list('xyz'):
                data[attr] = dd[attr]
        # Make the coordinates a separate array as they will be changed a lot
        coor = np.asarray(zip(dd['x'], dd['y'], dd['z']), dtype=np.float64)
        return cls(data, coor)

    def tofile(self, fname):
        PDBFile.write(fname, self)

    def rmsd(self, structure):
        return np.sqrt(((self.coor - structure.coor) ** 2).mean() * 3)

    def select(self, identifier, values, loperator='==', return_ind=False):
        """A simple way of selecting atoms"""
        if loperator == '==':
            oper = operator.eq
        elif loperator == '<':
            oper = operator.lt
        elif loperator == '>':
            oper = operator.gt
        elif loperator == '>=':
            oper = operator.ge
        elif loperator == '<=':
            oper = operator.le
        elif loperator == '!=':
            oper = operator.ne
        else:
            raise ValueError('Logic operator not recognized.')

        if not isinstance(values, Sequence) or isinstance(values, basestring):
            values = (values,)

        selection = oper(self.data[identifier], values[0])
        if len(values) > 1:
            for v in values[1:]:
                if loperator == '!=':
                    selection &= oper(self.data[identifier], v)
                else:
                    selection |= oper(self.data[identifier], v)

        if return_ind:
            return selection
        else:
            return Structure(self.data[selection], self.coor[selection])

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


class Ligand(Structure):

    """Ligand class is like a Structure, but has an added topology added to it."""

    def connectivity(self):
        if self._connectivity is None:
            dist_matrix = sp_squareform(sp_pdist(self.coor))
            covrad = self.covalent_radius
            cutoff_matrix = np.repeat(covrad, self.natoms).reshape(self.natoms, self.natoms)
            cutoff_matrix = cutoff_matrix + cutoff_matrix.T + 0.7
            connectivity_matrix = (dist_matrix < cutoff_matrix)
            np.fill_diagonal(connectivity_matrix, False)
            self._connectivity = connectivity_matrix
            self._cutoff_matrix = cutoff_matrix
        return self._connectivity

    def clashes(self):
        dist_matrix = sp_squareform(sp_pdist(self.coor))
        mask = np.logical_not(self.connectivity())
        occupancy_matrix = (self.q.reshape(1, -1) * self.q.reshape(-1, 1)) > 0
        mask &= occupancy_matrix
        np.fill_diagonal(mask, False)
        clash_matrix = dist_matrix < self._cutoff_matrix
        if np.any(np.logical_and(clash_matrix, mask)):
            return True
        return False

    def bonds(self):
        connectivity = self.connectivity()
        indices = np.nonzero(connectivity)
        for a, b in zip(*indices):
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
        conn = self.connectivity()
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
        """
        Determine all rotatable bonds. 
        A rotatable bond is currently described as two neighboring atoms with
        more than 1 neighbor and which are not part of the same ring.
        """

        conn = self.connectivity()
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
                # Check if atom is SP1 hybridized, i.e. angle with its
                # two neighbors is 180 degree.
                #if neighbors.sum() == 2:
                #    n1, n2 = neighbors
                #    origin = self.coor[atom]
                #    v1 = self.coor[n1] - origin
                #    v2 = self.coor[n2] - origin
                #    angle = np.rad2deg(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
                #    if (angle - 180) <= 2:
                #        break
        return rotatable_bonds

    def rigid_clusters(self):

        conn = self.connectivity()
        rings = self.ring_paths()
        clusters = []
        checked = []
        for root in xrange(self.natoms):
            if root in checked:
                continue
            cluster = [root]
            for atom in cluster:
                for ring in rings:
                    if atom in ring:
                        for a in ring:
                            if a not in cluster:
                                cluster.append(a)
                neighbors = np.flatnonzero(conn[atom])
                for n in neighbors:
                    if n in cluster:
                        continue
                    neighbor_neighbors = np.flatnonzero(conn[n])
                    if len(neighbor_neighbors) == 1:
                        cluster.append(n)
                checked.append(atom)
            if len(cluster) > 1:
                clusters.append(cluster)
        for atom in xrange(self.natoms):
            found = False
            for cluster in clusters:
                if atom in cluster:
                    found = True
                    break
            if not found:
                clusters.append([atom])
        return clusters


class BondOrder(object):

    def __init__(self, ligand, atom):
        self.ligand = ligand
        self.conn = self.ligand.connectivity()
        self.clusters = self.ligand.rigid_clusters()
        self.bonds = self.ligand.rotatable_bonds()
        self.checked_clusters = []
        self.order = []
        self.depth = []
        self.bondorder(atom)

    def bondorder(self, atom, depth=0):
        for cluster in self.clusters:
            if atom in cluster:
                break
        if cluster in self.checked_clusters:
            return
        depth += 1
        self.checked_clusters.append(cluster)
        neighbors = []
        for atom in cluster:
            neighbors += np.flatnonzero(self.conn[atom]).tolist()
        
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
            self.bondorder(n, depth)


class PDBFile(object):

    @classmethod
    def read(cls, fname):
        cls.coor = defaultdict(list)
        with open(fname) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    values = CoorRecord.parse_line(line)
                    for field in CoorRecord.fields:
                        cls.coor[field].append(values[field])
                elif line.startswith('MODEL'):
                    raise NotImplementedError("MODEL record is not implemented.")
        return cls

    @staticmethod
    def write(fname, structure):
        with open(fname, 'w') as f:
            for fields in zip(*[getattr(structure, x) for x in CoorRecord.fields]):
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
        for field, column, dtype in zip(cls.fields, cls.columns, cls.dtypes):
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
        for field, column, dtype in zip(cls.fields, cls.columns, cls.dtypes):
            values[field] = dtype(line[slice(*column)].strip())
        return values
