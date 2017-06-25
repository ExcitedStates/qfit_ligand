"""Convert crystallographic mtz file to P1 ccp4 density."""

import os
import argparse
import subprocess

from .config import CCTBX_DIR

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mtz_file", type=file,
            help="Crystallographic mtz file.")
    p.add_argument("-o", "--outfile", type=str, default=None,
            help=("Output ccp4 file name containing density. "
                  "Default is adding .ccp4 to mtz base filename.")
            )
    p.add_argument("--label", type=str, default="FWT,PHWT")
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    mtz_fname = args.mtz_file.name

    CCTBX_BINDIR = os.path.join(CCTBX_DIR, 'build', 'bin')
    mtz_fname_prefix = os.path.splitext(mtz_fname)[0]

    # Convert MTZ to CCP4
    cmd = [os.path.join(CCTBX_BINDIR, 'phenix.mtz2map'), mtz_fname]
    if 'FMODEL' in args.label:
        cmd.append("include_fmodel=True")
    print ' '.join(cmd)
    subprocess.call(cmd)
    # Rename the output file, since a lame '_1' is added.
    ccp4_fname_prefix = os.path.split(mtz_fname_prefix)[1]
    ccp4_fname_init = ccp4_fname_prefix + '_fmodel.ccp4'
    if args.outfile is None:
        ccp4_fname_final = ccp4_fname_prefix + '.ccp4'
    else:
        ccp4_fname_final = args.outfile
    print "Renaming {:s} to {:s}".format(os.path.abspath(ccp4_fname_init), os.path.abspath(ccp4_fname_final))
    os.rename(ccp4_fname_init, ccp4_fname_final)
