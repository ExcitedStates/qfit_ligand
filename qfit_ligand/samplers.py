import os.path
from collections import defaultdict
import itertools

import numpy as np
from scipy.spatial.distance import cdist as sp_cdist


class ClashDetector(object):

    """Detect clashes between ligand and receptor using spatial hashing."""

    def __init__(self, ligand, receptor, scaling_factor=1.0):

        self.ligand = ligand
        self.scaling_factor = scaling_factor
        receptor_radius = receptor.covalent_radius
        self.ligand_radius = self.ligand.covalent_radius
        self.voxelspacing = self.scaling_factor * (receptor_radius.max() + self.ligand_radius.max())

        self.grid = defaultdict(list)
        self.radius = defaultdict(list)
        keys = np.round(receptor.coor / self.voxelspacing)
        for key, coor, radius in itertools.izip(keys, receptor.coor, receptor_radius):
            key = tuple(key)
            iterator = itertools.product([-1, 0, 1], repeat=3)
            for trans in iterator:
                new_key = tuple(x + tx for x, tx in itertools.izip(key, trans))
                self.grid[new_key].append(coor)
                self.radius[new_key].append(radius)
        for key, value in self.grid.iteritems():
            self.grid[key] = np.asarray(value)
        for key, value in self.radius.iteritems():
            self.radius[key] = np.asarray(value)
        self._keys = np.zeros_like(self.ligand.coor)
        self.receptor = receptor

    def __call__(self):
        clashes = False
        np.round(self.ligand.coor / self.voxelspacing, out=self._keys)
        for key, coor, radius in itertools.izip(self._keys, self.ligand.coor, self.ligand_radius):
            key = tuple(key)
            coor2 = self.grid[key]
            if coor2 == []:
                continue
            dist2 = coor - coor2
            dist2 *= dist2
            cutoff = self.scaling_factor * (radius + self.radius[key])
            cutoff *= cutoff
            clashes = np.any(dist2.sum(axis=1) < cutoff)

            if clashes:
                break
        return clashes


class Translator(object):

    def __init__(self, ligand):
        self.ligand = ligand
        self.coor_to_rotate = self.ligand.coor.copy()

    def __call__(self, trans):
        self.ligand.coor[:] = self.coor_to_rotate + np.asarray(trans)


class GlobalRotator(object):

    """Rotate ligand around its center."""

    def __init__(self, ligand, center=None):

        self.ligand = ligand
        self._center = center
        if self._center is None:
            self._center = ligand.coor.mean(axis=0)
        self._coor_to_rotate = np.asmatrix(self.ligand.coor - self._center)

    def __call__(self, rotmat):
        self.ligand.coor[:] = (np.asmatrix(rotmat) * self._coor_to_rotate.T).T + self._center


class PrincipalAxisRotator(object):

    """Rotate ligand along the principal axes."""
    
    def __init__(self, ligand):
        self.ligand = ligand
        self._center = ligand.coor.mean(axis=0)
        self._coor_to_rotate = self.ligand.coor - self._center
        gyration_tensor = np.asmatrix(self._coor_to_rotate).T * np.asmatrix(self._coor_to_rotate)
        eig_values, eig_vectors = np.linalg.eigh(gyration_tensor)
        # Sort eigenvalues such that lx <= ly <= lz
        sort_ind = np.argsort(eig_values)
        self.principal_axes = np.asarray(eig_vectors[:, sort_ind].T)
        
        self.aligners = [ZAxisAligner(axis) for axis in self.principal_axes]

    def __call__(self, angle, axis=2):
        aligner = self.aligners[axis]
        R = aligner.forward_rotation * np.asmatrix(Rz(angle)) * aligner.backward_rotation
        self.ligand.coor[:] = (R * self._coor_to_rotate.T).T + self._center


# TODO Make a super class combining the BondRotator with the AngleRotator or at
# refactorize code.
class BondAngleRotator(object):

    """Rotate ligand along a bond angle defined by three atoms."""

    def __init__(self, ligand, a1, a2, a3, key='atomname'):
        # Atoms connected to a1 will stay fixed.
        self.ligand = ligand
        self.atom1 = a1
        self.atom2 = a2
        self.atom3 = a3

        # Determine which atoms will be moved by the rotation.
        self._root = getattr(ligand, key).tolist().index(a2)
        self._conn = ligand.connectivity()
        self.atoms_to_rotate = [self._root]
        self._foundroot = 0
        curr = getattr(ligand, key).tolist().index(a3)
        self._find_neighbours_recursively(curr)
        if self._foundroot > 1:
            raise ValueError("Atoms are part of a ring. Bond angle cannot be rotated.")

        # Find the rigid motion that aligns the axis of rotation onto the z-axis.
        self._coor_to_rotate = self.ligand.coor[self.atoms_to_rotate].copy()
        # Move root to origin
        self._t = self.ligand.coor[self._root]
        self._coor_to_rotate -= self._t
        # The rotation axis is the cross product between a1 and a3.
        a1_coor = self.ligand.coor[getattr(ligand, key).tolist().index(a1)]
        axis = np.cross(a1_coor - self._t, self._coor_to_rotate[1])

        # Align the rotation axis to the z-axis for the coordinates
        aligner = ZAxisAligner(axis)
        self._forward = aligner.forward_rotation
        self._coor_to_rotate = (aligner.backward_rotation * 
                np.asmatrix(self._coor_to_rotate.T)).T

    def _find_neighbours_recursively(self, curr):
        self.atoms_to_rotate.append(curr)
        bonds = np.flatnonzero(self._conn[curr])
        for b in bonds:
            if b == self._root:
                self._foundroot += 1
            if b not in self.atoms_to_rotate:
                self._find_neighbours_recursively(b)

    def __call__(self, angle):

        # Since the axis of rotation is already aligned with the z-axis, we can
        # freely rotate them and perform the inverse operation to realign the
        # axis to the real world frame.
        R = self._forward * np.asmatrix(Rz(angle))
        self.ligand.coor[self.atoms_to_rotate] = (R * self._coor_to_rotate.T).T + self._t


class BondRotator(object):

    """Rotate ligand along the bond of two atoms."""

    def __init__(self, ligand, a1, a2, key='atomname'):
        # Atoms connected to a1 will stay fixed.
        self.ligand = ligand
        self.atom1 = a1
        self.atom2 = a2

        # Determine which atoms will be moved by the rotation.
        self._root = getattr(ligand, key).tolist().index(a1)
        self._conn = ligand.connectivity()
        self.atoms_to_rotate = [self._root]
        self._foundroot = 0
        curr = getattr(ligand, key).tolist().index(a2)
        self._find_neighbours_recursively(curr)
        #if self._foundroot > 1:
        #    raise ValueError("Atoms are part of a ring. Bond cannot be rotated.")

        # Find the rigid motion that aligns the axis of rotation onto the z-axis.
        self._coor_to_rotate = self.ligand.coor[self.atoms_to_rotate].copy()
        # Move root to origin
        self._t = self.ligand.coor[self._root]
        self._coor_to_rotate -= self._t
        # Find angle between rotation axis and x-axis
        axis = self._coor_to_rotate[1] / np.linalg.norm(self._coor_to_rotate[1,:-1])
        aligner = ZAxisAligner(axis)

        # Align the rotation axis to the z-axis for the coordinates
        self._forward = aligner.forward_rotation
        self._coor_to_rotate = (aligner.backward_rotation * 
                np.asmatrix(self._coor_to_rotate.T)).T

    def _find_neighbours_recursively(self, curr):
        self.atoms_to_rotate.append(curr)
        bonds = np.flatnonzero(self._conn[curr])
        for b in bonds:
            if b == self._root:
                self._foundroot += 1
            if b not in self.atoms_to_rotate:
                self._find_neighbours_recursively(b)

    def __call__(self, angle):

        # Since the axis of rotation is already aligned with the z-axis, we can
        # freely rotate them and perform the inverse operation to realign the
        # axis to the real world frame.
        R = self._forward * np.asmatrix(Rz(angle))
        self.ligand.coor[self.atoms_to_rotate] = (R * self._coor_to_rotate.T).T + self._t


class ZAxisAligner(object):
    
    """Find the rotation that aligns a vector to the Z-axis."""

    def __init__(self, axis):
        # Find angle between rotation axis and x-axis
        axis = axis / np.linalg.norm(axis[:-1])
        xaxis_angle = np.arccos(axis[0])
        if axis[1] < 0:
            xaxis_angle *= -1
        # Rotate around Z-axis
        self._Rz = Rz(xaxis_angle)
        axis = np.dot(self._Rz.T, axis.reshape(3, -1)).ravel()
        # Find angle between rotation axis and z-axis
        zaxis_angle = np.arccos(axis[2] / np.linalg.norm(axis))
        if axis[0] < 0:
            zaxis_angle *= -1
        self._Ry = Ry(zaxis_angle)
        # Check whether the transformation is correct.
        # Rotate around the Y-axis to align to the Z-axis.
        axis = np.dot(self._Ry.T, axis.reshape(3, -1)).ravel() / np.linalg.norm(axis)
        if not np.allclose(axis, [0, 0, 1]):
            print axis
            raise ValueError("Axis is not aligned to z-axis.")
        self.backward_rotation = np.asmatrix(self._Ry).T * np.asmatrix(self._Rz).T
        self.forward_rotation = np.asmatrix(self._Rz) * np.asmatrix(self._Ry)


def Rz(theta):
    """Rotate along z-axis."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.asarray([[cos_theta, -sin_theta, 0],
                       [sin_theta,  cos_theta, 0],
                       [        0,          0, 1]])


def Ry(theta):
    """Rotate along y-axis."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.asarray([[ cos_theta, 0, sin_theta],
                       [         0, 1,         0],
                       [-sin_theta, 0, cos_theta]])


class RotationSets(object):    

    # Rotation sets available: (fname, nrot, angle).
    SETS = (('E.npy', 1, 360.0),
            ('c48u1.npy', 24, 62.8),
            ('c600v.npy', 60, 44.48),
            ('c48n9.npy', 216, 36.47),
            ('c600vc.npy', 360, 27.78),
            ('c48u27.npy', 648, 20.83),
            ('c48u83.npy', 1992, 16.29),
            ('c48u181.npy', 4344, 12.29),
            ('c48n309.npy', 7416, 9.72),
            ('c48n527.npy', 12648, 8.17),
            ('c48u815.npy', 19560, 7.4),
            ('c48u1153.npy', 27672, 6.6),
            ('c48u1201.npy', 28824, 6.48),
            ('c48u1641.npy', 39384, 5.75),
            ('c48u2219.npy', 53256, 5.27),
            ('c48u2947.npy', 70728, 4.71),
            ('c48u3733.npy', 89592, 4.37),
            ('c48u4749.npy', 113976, 4.0),
            ('c48u5879.npy', 141096, 3.74),
            ('c48u7111.npy', 170664, 3.53),
            ('c48u8649.npy', 207576, 3.26),
            )
    LOCAL = (('local_5_10.npy', 10, 5.00),
             ('local_5_100.npy', 100, 5.00),
             ('local_5_1000.npy', 1000, 5.00),
             ('local_10_10.npy', 10, 10.00),
             ('local_10_100.npy', 100, 10.00),
             ('local_10_1000.npy', 1000, 10.00),
             )

    _DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), 'data')

    @classmethod
    def get_set(cls, angle):
        angles = zip(*cls.SETS)[-1]
        diff = [abs(a - angle) for a in angles]
        fname = cls.SETS[diff.index(min(diff))][0]
        with open(os.path.join(cls._DATA_DIRECTORY, fname)) as f:
            quat_weights = np.load(f)
        return cls.quat_to_rotmat(quat_weights[:, :4])

    @classmethod
    def get_local_set(cls, fname='local_10_10.npy'):
        quats = np.load(os.path.join(cls._DATA_DIRECTORY, fname))
        return cls.quat_to_rotmat(quats)
            
    @classmethod
    def local(cls, max_angle, nrots=100):
        quats = []
        radian_max_angle = np.deg2rad(max_angle)
        while len(quats) < nrots - 1:
            quat = cls.random_rotmat(matrix=False)
            angle = 2 * np.arccos(quat[0])
            if angle <= radian_max_angle:
                quats.append(quat)
        quats.append(np.asarray([1, 0, 0, 0], dtype=np.float64))
        return np.asarray(quats)

    @staticmethod
    def quat_to_rotmat(quaternions):

        quaternions = np.asarray(quaternions)

        w = quaternions[:, 0]
        x = quaternions[:, 1]
        y = quaternions[:, 2]
        z = quaternions[:, 3]

        Nq = w**2 + x**2 + y**2 + z**2
        s = np.zeros(Nq.shape, dtype=np.float64)
        s[Nq >  0.0] = 2.0/Nq[Nq > 0.0]
        s[Nq <= 0.0] = 0

        X = x*s
        Y = y*s
        Z = z*s

        rotmat = np.zeros((quaternions.shape[0],3,3), dtype=np.float64)
        rotmat[:,0,0] = 1.0 - (y*Y + z*Z)
        rotmat[:,0,1] = x*Y - w*Z
        rotmat[:,0,2] = x*Z + w*Y

        rotmat[:,1,0] = x*Y + w*Z
        rotmat[:,1,1] = 1.0 - (x*X + z*Z)
        rotmat[:,1,2] = y*Z - w*X

        rotmat[:,2,0] = x*Z - w*Y
        rotmat[:,2,1] = y*Z + w*X
        rotmat[:,2,2] = 1.0 - (x*X + y*Y)

        np.around(rotmat, decimals=8, out=rotmat)

        return rotmat

    @classmethod
    def random_rotmat(cls, matrix=True):
        """Return a random rotation matrix"""

        s1 = 1
        while s1 >= 1.0:
            e1 = np.random.random() * 2 - 1
            e2 = np.random.random() * 2 - 1
            s1 = e1**2 + e2**2
                
        s2 = 1
        while s2 >= 1.0:
            e3 = np.random.random() * 2 - 1
            e4 = np.random.random() * 2 - 1
            s2 = e3**2 + e4**2
        
        q0 = e1
        q1 = e2
        q2 = e3 * np.sqrt((1 - s1)/s2 )
        q3 = e4 * np.sqrt((1 - s1)/s2 )

        quat = [q0, q1, q2, q3]
        if matrix:
            return cls.quat_to_rotmat(np.asarray(quat).reshape(1, 4))[0]
        else:
            return quat
