from __future__ import division

import numpy as np
from scipy.integrate import quadrature

from .atomsf import ATOM_STRUCTURE_FACTORS
from ._extensions import dilate_points, mask_points


class Transformer(object):

    """Transform a structure to a density."""

    def __init__(self, ligand, volume, smin=0, smax=0.5, rmax=3.0,
                 rstep=0.01, simple=False):
        self.ligand = ligand
        self.volume = volume
        self.smin = smin
        self.smax = smax
        self.rmax = rmax
        self.rstep = rstep
        self.simple = simple
        self.asf = ATOM_STRUCTURE_FACTORS
        self._initialized = False

        # Calculate transforms
        alpha, beta, gamma = np.deg2rad(self.volume.angles)
        a, b, c = self.volume.lattice_parameters
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        omega = np.sqrt(
                1 - cos_alpha * cos_alpha - cos_beta * cos_beta - cos_gamma * cos_gamma +
                2 * cos_alpha * cos_beta * cos_gamma
                )

        self.lattice_to_cartesian = np.asarray([
            [1, cos_gamma, cos_beta],
            [0, sin_gamma, (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
            [0,         0, omega / sin_gamma],
            ])
        self.cartesian_to_lattice = np.asarray([
            [1, -cos_gamma / sin_gamma, 
                (cos_alpha * cos_gamma - cos_beta) / (omega * sin_gamma)],
            [0, 1 / sin_gamma,
                (cos_beta * cos_gamma - cos_alpha) / (omega * sin_gamma)],
            [0, 0,
                sin_gamma / omega],
            ])
        self.grid_to_cartesian = self.lattice_to_cartesian * self.volume.voxelspacing
        self._grid_coor = np.zeros_like(self.ligand.coor)

    def mask(self, rmax=None):
        #transform = self.cartesian_to_lattice
        #np.dot(self.ligand.coor - self.volume.origin, transform.T, out=self._grid_coor)
        #self._grid_coor -= self.volume.offset
        transform = np.asmatrix(self.cartesian_to_lattice)
        self._grid_coor = ((transform * (self.ligand.coor - self.volume.origin).T).T 
                / self.volume.voxelspacing - self.volume.offset)
        if rmax is None:
            rmax = self.rmax

        lmax = np.asarray(
                [rmax / vs for vs in self.volume.voxelspacing],
                dtype=np.float64)
        mask_points(self._grid_coor, self.ligand.q, lmax, rmax, 
                    self.grid_to_cartesian, 1.0, self.volume.array)

    def reset(self):
        transform = np.asmatrix(self.cartesian_to_lattice)
        self._grid_coor = ((transform * (self.ligand.coor - self.volume.origin).T).T 
                / self.volume.voxelspacing - self.volume.offset)
        #transform = self.cartesian_to_lattice
        #np.dot(self.ligand.coor - self.volume.origin, transform.T, out=self._grid_coor)
        #self._grid_coor /= self.volume.voxelspacing
        #self._grid_coor -= self.volume.offset
        lmax = np.asarray(
                [self.rmax / vs for vs in self.volume.voxelspacing],
                dtype=np.float64)
        mask_points(self._grid_coor, self.ligand.q, lmax, self.rmax, 
                    self.grid_to_cartesian, 0.0, self.volume.array)

    def initialize(self):
        self.radial_densities = []
        for n in xrange(self.ligand.natoms):
            if self.simple:
                rdens = self.simple_radial_density(self.ligand.e[n], self.ligand.b[n])[1]
            else:
                rdens = self.radial_density(self.ligand.e[n], self.ligand.b[n])[1]
            self.radial_densities.append(rdens)
        self.radial_densities = np.ascontiguousarray(self.radial_densities)
        self._initialized = True

    def density(self):
        """Transform structure to a density in a volume."""

        if not self._initialized:
            self.initialize()

        transform = np.asmatrix(self.cartesian_to_lattice)
        self._grid_coor = ((transform * (self.ligand.coor - self.volume.origin).T).T 
                / self.volume.voxelspacing - self.volume.offset)
        #transform = self.cartesian_to_lattice
        #np.dot(self.ligand.coor - self.volume.origin, transform.T, out=self._grid_coor)
        #self._grid_coor /= self.volume.voxelspacing
        #self._grid_coor -= self.volume.offset
        lmax = np.asarray(
                [self.rmax / vs for vs in self.volume.voxelspacing],
                dtype=np.float64)

        dilate_points(self._grid_coor, self.ligand.q, lmax, 
                self.radial_densities, self.rstep, self.rmax,
                self.grid_to_cartesian, self.volume.array)

    def simple_radial_density(self, element, bfactor):
        """Calculate electron density as a function of radius."""

        assert bfactor > 0, "B-factor should be bigger than 0"

        asf = self.asf[element.capitalize()]
        bw = [-4 * np.pi * np.pi / (asf[1][i] + bfactor) for i in xrange(6)]
        aw = [asf[0][i] * (-bw[i] / np.pi) ** 1.5 for i in xrange(6)]
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        r2 = r * r
        density = (aw[0] * np.exp(bw[0] * r2) + aw[1] * np.exp(bw[1] * r2) + 
                   aw[2] * np.exp(bw[2] * r2) + aw[3] * np.exp(bw[3] * r2) + 
                   aw[4] * np.exp(bw[4] * r2) + aw[5] * np.exp(bw[5] * r2)
                   )
        return r, density

    def radial_density(self, element, bfactor):
        """Calculate electron density as a function of radius."""
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        r[0] = 0.00001
        density = np.zeros_like(r)
        for n, x in enumerate(r):
            args = (x, self.asf[element.capitalize()], bfactor)
            integrand, err = quadrature(self._scattering_integrand, self.smin,
                                        self.smax, args=args)
            density[n] = (8.0 / x) * integrand
        return r, density

    @staticmethod
    def _scattering_integrand(s, r, asf, bfactor):
        """Integral function to be approximated to obtain radial density."""
        s2 = s * s
        f = (asf[0][0] * np.exp(-asf[1][0] * s2) +
             asf[0][1] * np.exp(-asf[1][1] * s2) +
             asf[0][2] * np.exp(-asf[1][2] * s2) +
             asf[0][3] * np.exp(-asf[1][3] * s2) +
             asf[0][4] * np.exp(-asf[1][4] * s2) +
             asf[0][5])
        return f * np.exp(-bfactor * s2) * np.sin(4 * np.pi * r * s) * s
