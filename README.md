# qfit\_ligand

## Requirements

* Python2.7
* NumPy
* SciPy
* cvxopt
* IBM ILOG CPLEX Optimization Studio (Community Edition)

Optional requirements used for installation

* pip
* git


## Installation

To obtain the requirement, first install `numpy`, `scipy` and `cvxopt` using
`pip`

    pip install numpy scipy cvxopt

Next, obtain a copy of CPLEX, the Community Edition will do. After downloading
it from the [IBM website][1], install the Python interface

    cd <CPLEX_ROOT>/cplex/python/2.7/x86_64_<PLATFORM>
    python setup.py install

where `CPLEX_ROOT` is the directory where you install CPLEX, and `PLATFORM` is
a platform dependent string, such as `linux` for Linux systems and `osx` for
macOSX.

You are all set now to install `qfit_ligand`. Installation of `qfit_ligand` is
as simple as

    git clone https://github.com/excitedstates/qfit_ligand
    cd qfit_ligand
    python setup.py install


## Usage

The main program that comes with installing `qfit_ligand` is the eponymously named
`qfit_ligand` command line tool. It's calling signature is

    qfit_ligand <CCP4-map-P1> <resolution> <PDB>

where `CCP4-map-P1` is an electron density map in CCP4 format with P1 symmetry.
If your current density has a different spacegroup, reduce the symmetry to P1
(see below). `resolution` is the resolution of the electron density, and `PDB`
is the initial single-conformer ligand conformation in PDB-format.

To see all options, type

    qfit_ligand -h

The main options are `-s` to give the angular stepsize in degree, and `-b` to
provide the number of degrees of freedom that are sampled simultaneously.


## Converting your spacegroup to P1

To convert the electron density to spacegroup P1, you can either use the
`extend_to_p1` binary in the `xraytools` repository, or extend your structure
factors to P1 using the CCTBX command line tool
`phenix.reflection_file_converter` and convert it to a real space electron
density.


[1]: https://www-01.ibm.com/software/websphere/products/optimization/cplex-studio-community-edition/ "IBM website"

