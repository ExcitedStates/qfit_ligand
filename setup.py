#! env/bin/python
import os.path
from setuptools import setup
from setuptools.extension import Extension
import numpy as np


def main():


    packages = ['qfit_ligand']
    package_data = {'qfit_ligand': [os.path.join('data', '*.npy')]}
    data_files = [('qfit_ligand', ['config.py']),]

    ext_modules = [Extension("qfit_ligand._extensions",
                      [os.path.join("src", "_extensions.c")],
                      include_dirs=[np.get_include()],
                      extra_compile_args=['-ffast-math'],
                      ),
                   ]

    setup(name="qfit_ligand",
          version='0.1.0',
          author='Gydo C.P. van Zundert',
          author_email='gydo.vanzundert@schrodinger.com',
          packages=packages,
          package_data = package_data,
          ext_modules=ext_modules,
          data_files=data_files,
          install_requires=['numpy', 'scipy', 'cvxopt', 'cplex'],
          entry_points={
              'console_scripts': [
                  'qfit_ligand = qfit_ligand.hierarchical:main',
                  'qpsolve = qfit_ligand.solve:solve',
                  'density = qfit_ligand.density:main',
                  ]
              },
         )


if __name__=='__main__':
    main()
