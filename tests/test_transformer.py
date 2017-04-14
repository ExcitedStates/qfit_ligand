from qfit_ligand.structure import Structure
from qfit_ligand.volume import Volume, CCP4Parser
from qfit_ligand.transformer import Transformer

voxelspacing = (0.25, 0.5, 1)
angles = (120, 70, 120)
v2 = Volume.fromfile('1xvp_p1.ccp4')
print v2.angles
print v2.voxelspacing

v = Volume.zeros(v2.shape, voxelspacing=v2.voxelspacing, angles=angles)
s = Structure.fromfile('CID.pdb')
r = 2.6

smax = 0.5 / r
t = Transformer(s, v, smax=smax)
t.mask(1)
v.tofile('mask.ccp4')
t.reset()
t.density()
v.tofile('density.ccp4')
