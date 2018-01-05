import os.path
from sys import exit

from setuptools import setup
from setuptools.extension import Extension
import numpy as np

try:
    import cplex
except ImportError as err:
    msg = ('\nCPLEX is required to install qfit_ligand. '
           'Obtain it from the IBM website, or install it with conda:\n'
           '    conda install -c ibmdecisionoptimization cplex\n'
          )
    print msg
    exit()


def main():

    packages = ['qfit_ligand']
    package_data = {'qfit_ligand': [os.path.join('data', '*.npy'),]
    }

    ext_modules = [Extension("qfit_ligand._extensions",
                      [os.path.join("src", "_extensions.c")],
                      include_dirs=[np.get_include()],
                      ),
                   ]
    install_requires = [
        'numpy>=1.11',
        'scipy>=0.19',
        'cvxopt>=1.1.9',
    ]

    setup(name="qfit_ligand",
          version='0.1.0',
          author='Gydo C.P. van Zundert',
          author_email='gydo.vanzundert@schrodinger.com',
          packages=packages,
          package_data=package_data,
          ext_modules=ext_modules,
          install_requires=install_requires,
          entry_points={
              'console_scripts': [
                  'qfit_ligand = qfit_ligand.qfit_ligand:main',
                  'qfit_residue = qfit_ligand.qfit_residue:main',
                  'qfit_solve = qfit_ligand.qfit_solve:solve',
                  'qfit_density = qfit_ligand.qfit_density:main',
                  'qfit_scale = qfit_ligand.qfit_scale:main',
                  'qfit_combine = qfit_ligand.qfit_combine:main',
                  ]
              },
         )


if __name__=='__main__':
    main()
