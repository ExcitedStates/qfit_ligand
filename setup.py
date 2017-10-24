import os.path

from setuptools import setup
from setuptools.extension import Extension
import numpy as np

def main():

    packages = ['qfit_ligand']

    ext_modules = [Extension("qfit_ligand._extensions",
                      [os.path.join("src", "_extensions.c")],
                      include_dirs=[np.get_include()],
                      extra_compile_args=['-ffast-math', '-std=c99'],
                      ),
                   ]

    setup(name="qfit_ligand",
          version='0.1.0',
          author='Gydo C.P. van Zundert',
          author_email='gydo.vanzundert@schrodinger.com',
          packages=packages,
          ext_modules=ext_modules,
          install_requires=['numpy', 'scipy', 'cvxopt', 'cplex'],
          entry_points={
              'console_scripts': [
                  'qfit_ligand = qfit_ligand.qfit_ligand:main',
                  'qfit_solve = qfit_ligand.qfit_solve:solve',
                  'qfit_density = qfit_ligand.qfit_density:main',
                  'qfit_scale = qfit_ligand.qfit_scale:main',
                  'qfit_combine = qfit_ligand.qfit_combine:main',
                  ]
              },
         )


if __name__=='__main__':
    main()
