# qfit\_ligand

## Requirements

* Python2.7
* NumPy
* SciPy
* cvxopt
* IBM ILOG CPLEX Optimization Studio (Community Edition)

Optional

* CCTBX

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

where `<CPLEX_ROOT>` is the directory where you install CPLEX, and `<PLATFORM>` is
a platform dependent string, such as `linux` for Linux systems and `osx` for
macOSX.

You are all set now to install `qfit_ligand`. Installation of `qfit_ligand` is
as simple as

    git clone https://github.com/excitedstates/qfit_ligand
    cd qfit_ligand
    # Optional: configure config.py to point to CCTBX directory
    python setup.py install


## Usage

The main program that comes with installing `qfit_ligand` is the eponymously named
`qfit_ligand` command line tool. It's calling signature is

    qfit_ligand <CCP4-map-P1> <resolution> <PDB>

where `<CCP4-map-P1>` is an electron density map in CCP4 format with P1
symmetry.  If your current density has a different spacegroup, reduce the
symmetry to P1 (see below). `<resolution>` is the resolution of the electron
density, and `<PDB>` is the initial single-conformer ligand conformation in
PDB-format.

To see all options, type

    qfit_ligand -h

The main options are `-s` to give the angular stepsize in degree, and `-b` to
provide the number of degrees of freedom that are sampled simultaneously.


## Converting your spacegroup to P1

If you have access to CCTBX and configured the `config.py` file, you should
have access to a working command line tool `qfit_mtz_to_ccp4` which takes as
input an mtz-file and outputs a CCP4 P1 density. The mtz file for a particular
PDB can be downloaded from the PDBe, e.g. with `wget`

    wget http://www.ebi.ac.uk/pdbe/coordinates/files/<PDB-ID>_map.mtz

where `<PDB-ID>` is the 4-letter PDB identifyer.


[1]: https://www-01.ibm.com/software/websphere/products/optimization/cplex-studio-community-edition/ "IBM website"

