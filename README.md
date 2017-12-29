# qFit-ligand

## Requirements

* Python2.7
* NumPy
* SciPy
* cvxopt
* IBM ILOG CPLEX Optimization Studio (Community Edition)

Optional

* CCTBX or Phenix

Optional requirements used for installation

* pip
* git


## Installation

To obtain the requirement, first install `numpy`, `scipy` and `cvxopt` using
`pip`

    pip install numpy scipy cvxopt

Next, obtain a copy of CPLEX, the Community Edition will do. 

    pip install -c IBMDecisionOptimization docplex cplex

Alternatively, if you have a license for the full version of CPLEX, you 
can download it from the [IBM website][1], and install the Python interface

    cd <CPLEX_ROOT>/cplex/python/2.7/x86_64_<PLATFORM>
    python setup.py install

where `<CPLEX_ROOT>` is the directory where you install CPLEX, and `<PLATFORM>` is
a platform dependent string, such as `linux` for Linux systems and `osx` for
macOSX.

You are now all set now to install `qfit_ligand`. Installation of `qfit_ligand` is
as simple as

    git clone https://github.com/excitedstates/qfit_ligand
    cd qfit_ligand
    python setup.py install


## Usage

The main program that comes with installing `qfit_ligand` is the eponymously named
`qfit_ligand` command line tool. It has two calling interfaces

    qfit_ligand <CCP4-map> <resolution> <PDB-of-ligand> -r <PDB-of-receptor>
    qfit_ligand <CCP4-map> <resolution> <PDB-of-ligand-and-receptor> --selection <chain>,<resi>

where `<CCP4-map>` is a 2mFo-DFc electron density map in CCP4 format, and
`<resolution>` is its corresponding resolution in angstrom. In the first
command,`<PDB-of-ligand>` is a PDB file containing solely the ligand and
`<PDB-of-receptor>` a PDB file containing the receptor (and other ligands).
In the second command, `<PDB-of-ligand-and-receptor` is a PDB file containing
both the ligand and receptor, and `--selection` requires the chain and residue
id of the ligand as a comma separated value, e.g. `A,1234`. Note that the
receptor (and other ligands) are used to determine the scaling factor of the
density map and used for collision detection during conformer sampling.

To see all options, type

    qfit_ligand -h

The main options are `-s` to give the angular stepsize in degree, and `-b` to
provide the number of degrees of freedom that are sampled simultaneously.
Reasonably values are `-s 1 -b 1`, `-s 6 -b 2`, and `-s 24 -b 3`. Decreasing
`-s` and especially increasing `-b` further will be RAM memory intensive and
will take significantly longer.


## Converting MTZ to CCP4

If you have access to CCTBX/Phenix use `phenix.mtz2map` to convert a MTZ file
to CCP4. Make sure it outputs the 2mFo-DFc map. Read the documentation for
available options.


## Licence

The code is licensed under the Apache Version 2.0 licence (see `LICENSE`).

The `spacegroups.py` module is based on the `SpaceGroups.py` module of the
`pymmlib` package, originally licensed under the Artistic License 2.0. See the
`license` directory for a copy and its full license.


[1]: https://www-01.ibm.com/software/websphere/products/optimization/cplex-studio-community-edition/ "IBM website"

