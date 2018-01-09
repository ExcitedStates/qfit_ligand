from __future__ import division

from itertools import izip

import numpy as np
from scipy.integrate import fixed_quad

from .atomsf import ATOM_STRUCTURE_FACTORS, ELECTRON_SCATTERING_FACTORS
from ._extensions import dilate_points, mask_points, correlation_gradients


class Transformer(object):

    """Transform a structure to a density."""

    def __init__(self, ligand, volume, smin=0, smax=0.5, rmax=3.0,
                 rstep=0.01, simple=False, scattering='xray'):
        self.ligand = ligand
        self.volume = volume
        self.smin = smin
        self.smax = smax
        self.rmax = rmax
        self.rstep = rstep
        self.simple = simple
        if scattering == 'xray':
            self.asf = ATOM_STRUCTURE_FACTORS
        elif scattering == 'electron':
            self.asf = ELECTRON_SCATTERING_FACTORS
        else:
            raise ValueError("Scattering source not supported. Choose 'xray' or 'electron'")
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
        self._grid_coor_rot = np.zeros_like(self.ligand.coor)

    def _coor_to_grid_coor(self):
        if np.allclose(self.volume.origin, 0):
            coor = self.ligand.coor
        else:
            coor = self.ligand.coor - self.volume.origin
        np.dot(coor, self.cartesian_to_lattice.T, self._grid_coor)
        self._grid_coor /= self.volume.voxelspacing
        self._grid_coor -= self.volume.offset

    def mask(self, rmax=None):
        self._coor_to_grid_coor()
        if rmax is None:
            rmax = self.rmax
        lmax = np.asarray(
                [rmax / vs for vs in self.volume.voxelspacing],
                dtype=np.float64)
        for symop in self.volume.spacegroup.symop_list:
            np.dot(self._grid_coor, symop.R.T, self._grid_coor_rot)
            self._grid_coor_rot += symop.t * self.volume.shape[::-1]

            mask_points(self._grid_coor_rot, self.ligand.q, lmax, rmax,
                        self.grid_to_cartesian, 1.0, self.volume.array)

    def reset(self):
        self._coor_to_grid_coor()
        lmax = np.asarray(
                [self.rmax / vs for vs in self.volume.voxelspacing],
                dtype=np.float64)
        for symop in self.volume.spacegroup.symop_list:
            np.dot(self._grid_coor, symop.R.T, self._grid_coor_rot)
            self._grid_coor_rot += symop.t * self.volume.shape[::-1]

            mask_points(self._grid_coor_rot, self.ligand.q, lmax, self.rmax,
                        self.grid_to_cartesian, 0.0, self.volume.array)

    def initialize(self, derivative=False):
        self.radial_densities = []
        for n in xrange(self.ligand.natoms):
            if self.simple:
                rdens = self.simple_radial_density(self.ligand.e[n], self.ligand.b[n])[1]
            else:
                rdens = self.radial_density(self.ligand.e[n], self.ligand.b[n])[1]
            self.radial_densities.append(rdens)
        self.radial_densities = np.ascontiguousarray(self.radial_densities)

        if derivative:
            self.radial_derivatives = np.zeros_like(self.radial_densities)
            if self.simple:
                for n, (e, b) in enumerate(izip(self.ligand.e, self.ligand.b)):
                    self.radial_derivatives[n] = self.simple_radial_derivative(e, b)[1]
            else:
                #self.radial_derivatives[n] = self.radial_derivative(e, b)[1]
                # Use edge_order = 1, since the gradient is anti-symmetric in r
                self.radial_derivatives = np.gradient(self.radial_densities, self.rstep, edge_order=1, axis=1)

        self._initialized = True

    def density(self):
        """Transform structure to a density in a volume."""

        if not self._initialized:
            self.initialize()

        self._coor_to_grid_coor()
        lmax = np.asarray(
                [self.rmax / vs for vs in self.volume.voxelspacing],
                dtype=np.float64)
        for symop in self.volume.spacegroup.symop_list:
            np.dot(self._grid_coor, symop.R.T, self._grid_coor_rot)
            self._grid_coor_rot += symop.t * self.volume.shape[::-1]
            dilate_points(self._grid_coor_rot, self.ligand.q, lmax,
                          self.radial_densities, self.rstep, self.rmax,
                          self.grid_to_cartesian, self.volume.array)

    def simple_radial_density(self, element, bfactor):
        """Calculate electron density as a function of radius."""

        #assert bfactor > 0, "B-factor should be bigger than 0"

        asf = self.asf[element.capitalize()]
        four_pi2 = 4 * np.pi * np.pi
        bw = []
        for i in xrange(6):
            try:
                bw.append(-four_pi2 / (asf[1][i] + bfactor))
            except ZeroDivisionError:
                bw.append(0)
        aw = [asf[0][i] * (-bw[i] / np.pi) ** 1.5 for i in xrange(6)]
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        r2 = r * r
        density = (aw[0] * np.exp(bw[0] * r2) + aw[1] * np.exp(bw[1] * r2) +
                   aw[2] * np.exp(bw[2] * r2) + aw[3] * np.exp(bw[3] * r2) +
                   aw[4] * np.exp(bw[4] * r2) + aw[5] * np.exp(bw[5] * r2)
                   )
        return r, density

    def simple_radial_derivative(self, element, bfactor):
        """Calculate gradient."""
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        r2 = r * r
        asf = self.asf[element.capitalize()]
        four_pi2 = 4 * np.pi * np.pi
        bw = [-four_pi2 / (asf[1][i] + bfactor) for i in xrange(6)]
        aw = [asf[0][i] * (-bw[i] / np.pi) ** 1.5 for i in xrange(6)]
        derivative = np.zeros(r.size, np.float64)
        for i in xrange(6):
            derivative += (2.0 * bw[i] * aw[i]) * np.exp(bw[i] * r2)
        derivative *= r
        return r, derivative

    def correlation_gradients(self, target):

        self._coor_to_grid_coor()
        lmax = np.asarray(
                [self.rmax / vs for vs in self.volume.voxelspacing],
                dtype=np.float64)
        gradients = np.zeros_like(self.ligand.coor)

        correlation_gradients(
            self._grid_coor, self.ligand.q, lmax, self.radial_derivatives, self.rstep,
            self.rmax, self.grid_to_cartesian, target.array, gradients)
        return gradients

    def radial_density(self, element, bfactor):
        """Calculate electron density as a function of radius."""
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        density = np.zeros_like(r)
        for n, x in enumerate(r):
            asf = self.asf[element.capitalize()]
            args = (x, asf, bfactor)
            # Use a fixed number of quadrature points, 50 is more than enough
            #integrand, err = quadrature(self._scattering_integrand, self.smin,
            #                            self.smax, args=args)#, tol=1e-5, miniter=13, maxiter=15)
            integrand, err = fixed_quad(self._scattering_integrand, self.smin,
                                        self.smax, args=args, n=50)
            density[n] = integrand
        return r, density

    def radial_derivative(self, element, bfactor):
        r = np.arange(0, self.rmax + self.rstep + 1, self.rstep)
        derivative = np.zeros_like(r)
        for n, x in enumerate(r):
            asf = self.asf[element.capitalize()]
            args = (x, asf, bfactor)
            integrand, err = quadrature(self._scattering_integrand_derivative, self.smin,
                                        self.smax, args=args)
            derivative[n] = integrand
        return r, derivative

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
        w = 8 * f * np.exp(-bfactor * s2) * s
        a = 4 * np.pi * s
        if r > 1e-4:
            return w / r * np.sin(a * r)
        else:
            # Return 4th order Tayler expansion
            return w * a * (1 - a * a * r * r / 6.0)

    @staticmethod
    def _scattering_integrand_derivative(s, r, asf, bfactor):
        s2 = s * s
        f = asf[0][5]
        for a, b in izip(asf[0], asf[1]):
            #f += asf[0][i] * np.exp(-asf[1][i] * s2)
            f += a * np.exp(-b * s2)
        a = 4 * np.pi * s
        w = 8 * f * np.exp(-bfactor * s2) * s
        ar = a * r
        if r > 1e-4:
            return w / r * (a * np.cos(ar) - np.sin(ar) / r)
        else:
            # Return 4th order Tayler expansion
            ar2 = ar * ar
            a3 = a * a * a
            return w * a3 * r * (ar2 - 8) / 24.0
