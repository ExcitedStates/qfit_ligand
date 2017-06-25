from __future__ import division, absolute_import
from struct import unpack as _unpack, pack as _pack
import os.path
from sys import byteorder as _BYTEORDER
import warnings

import numpy as np
from scipy.ndimage import zoom, gaussian_filter

from .spacegroups import GetSpaceGroup
from ._extensions import extend_to_p1


class Volume(object):

    def __init__(self, array, voxelspacing=1.0, origin=(0, 0, 0), 
                 angles=(90, 90, 90), offset=(0, 0, 0), spacegroup=None, cell_shape=None):

        self.array = array
        if isinstance(voxelspacing, float):
            voxelspacing = tuple([voxelspacing] * 3)
        self.voxelspacing = voxelspacing
        self.origin = origin
        self.angles = angles
        self.offset = offset
        self.lattice_parameters = [x * vs 
                for x, vs in zip(self.array.shape[::-1], self.voxelspacing)]

        self.a, self.b, self.c = self.lattice_parameters
        self.alpha, self.beta, self.gamma = self.angles
        if spacegroup is not None:
            self.spacegroup = GetSpaceGroup(spacegroup)
        else:
            self.spacegroup = None
        self.cell_shape = cell_shape

    @classmethod
    def fromfile(cls, fid, fmt=None):
        p = parse_volume(fid)
        return cls(p.density, voxelspacing=p.voxelspacing, 
                origin=p.origin, angles=p.angles, offset=p.offset,
                spacegroup=p.spacegroup, cell_shape=p.cell_shape)

    @classmethod
    def zeros(cls, shape, voxelspacing=1.0, origin=(0, 0, 0), 
              angles=(90, 90, 90), offset=(0, 0, 0), spacegroup=None, cell_shape=None):
        return cls(np.zeros(shape, dtype=np.float64), voxelspacing, 
                origin, angles, offset, spacegroup, cell_shape)

    @classmethod
    def zeros_like(cls, volume):
        return cls(np.zeros_like(volume.array), volume.voxelspacing,
                volume.origin, volume.angles, volume.offset, 
                volume.spacegroup, volume.cell_shape)
        
    @property
    def shape(self):
        return self.array.shape

    def duplicate(self):
        return Volume(self.array.copy(), voxelspacing=self.voxelspacing,
                      origin=self.origin, angles=self.angles, offset=self.offset)

    def fill_unit_cell(self):
        if self.cell_shape is None:
            raise ValueError("cell_shape attribute is None.")
        out = Volume.zeros(self.cell_shape, voxelspacing=self.voxelspacing, 
                origin=self.origin, angles=self.angles, offset=(0,0,0), 
                spacegroup=self.spacegroup, cell_shape=self.cell_shape)
        offset = np.asarray(self.offset, np.int32)
        for symop in self.spacegroup.symop_list:
            trans = np.hstack((symop.R, symop.t.reshape(3, -1)))
            trans[:, -1] *= out.shape[::-1]
            extend_to_p1(self.array, offset, trans, out.array)
        return out

    def set_spacegroup(self, spacegroup):
        self.spacegroup = GetSpaceGroup(spacegroup)

    def tofile(self, fid, fmt=None):
        if fmt is None:
            fmt = os.path.splitext(fid)[-1][1:]
        if fmt in ('ccp4', 'map', 'mrc'):
            to_mrc(fid, self)
        elif fmt in ('xplor', 'cns'):
            to_xplor(fid, self)
        else:
            raise ValueError("Format is not supported.")


# Volume parsers
def parse_volume(fid, fmt=None):
    try:
        fname = fid.name
    except AttributeError:
        fname = fid

    if fmt is None:
        fmt = os.path.splitext(fname)[-1][1:]
    if fmt in ('ccp4', 'map'):
        p = CCP4Parser(fname)
    elif fmt == 'mrc':
        p = MRCParser(fname)
    else:
        raise ValueError('Extension of file is not supported.')
    return p


class CCP4Parser(object):

    HEADER_SIZE = 1024
    HEADER_TYPE = ('i' * 10 + 'f' * 6 + 'i' * 3 + 'f' * 3 + 'i' * 3 +
                   'f' * 27 + 'c' * 8 + 'f' * 1 + 'i' * 1 + 'c' * 800)
    HEADER_FIELDS = (
          'nc nr ns mode ncstart nrstart nsstart nx ny nz xlength ylength '
          'zlength alpha beta gamma mapc mapr maps amin amax amean ispg '
          'nsymbt lskflg skwmat skwtrn extra xstart ystart zstart map '
          'machst rms nlabel label'
          ).split()
    HEADER_CHUNKS = [1] * 25 + [9, 3, 12] + [1] * 3 + [4, 4, 1, 1, 800]

    def __init__(self, fid):

        if isinstance(fid, str):
            fhandle = open(fid)
        elif isinstance(fid, file):
            fhandle = fid
        else:
            raise ValueError("Input should either be a file or filename.")

        self.fhandle = fhandle
        self.fname = fhandle.name

        # first determine the endiannes of the file
        self._get_endiannes()
        # get the header
        self._get_header()
        self.params = tuple(self.header[key] for key in ('xlength', 'ylength', 'zlength'))
        self.angles = tuple(self.header[key] for key in ('alpha', 'beta', 'gamma'))
        self.shape = tuple(self.header[key] for key in ('nx', 'ny', 'nz'))
        self.voxelspacing = tuple(length / n 
                for length, n in zip(self.params, self.shape))
        self.spacegroup = int(self.header['ispg'])
        self.cell_shape = [self.header[key] for key in 'nz ny nx'.split()]
        self._get_offset()
        self._get_origin()
        # Get the symbol table and ultimately the density
        self._get_symbt()
        self._get_density()

    def _get_endiannes(self):
        self.fhandle.seek(212)
        m_stamp = hex(ord(self.fhandle.read(1)))
        if m_stamp == '0x44':
            endian = '<'
        elif m_stamp == '0x11':
            endian = '>'
        else:
            raise ValueError('Endiannes is not properly set in file. Check the file format.')
        self._endian = endian
        self.fhandle.seek(0)

    def _get_header(self):
        header = _unpack(self._endian + self.HEADER_TYPE,
                         self.fhandle.read(self.HEADER_SIZE))
        self.header = {}
        index = 0
        for field, nchunks in zip(self.HEADER_FIELDS, self.HEADER_CHUNKS):
            end = index + nchunks
            if nchunks > 1:
                self.header[field] = header[index: end]
            else:
                self.header[field] = header[index]
            index = end
        self.header['label'] = ''.join(self.header['label'])

    def _get_offset(self):
        self.offset = [0] * 3
        self.offset[self.header['mapc'] - 1] = self.header['ncstart']
        self.offset[self.header['mapr'] - 1] = self.header['nrstart']
        self.offset[self.header['maps'] - 1] = self.header['nsstart']

    def _get_origin(self):
        self.origin = (0, 0, 0)

    def _get_symbt(self):
        self.symbt = self.fhandle.read(self.header['nsymbt'])

    def _get_density(self):

        # Determine the dtype of the file based on the mode
        mode = self.header['mode']
        if mode == 0:
            dtype = 'i1'
        elif mode == 1:
            dtype = 'i2'
        elif mode == 2:
            dtype = 'f4'

        # Read the density
        storage_shape = tuple(self.header[key] for key in ('ns', 'nr', 'nc'))
        self.density = np.fromfile(self.fhandle,
                              dtype=self._endian + dtype).reshape(storage_shape)

        # Reorder axis so that nx is fastest changing.
        maps, mapr, mapc = [self.header[key] for key in ('maps', 'mapr', 'mapc')]
        if maps == 3 and mapr == 2 and mapc == 1:
            pass
        elif maps == 3 and mapr == 1 and mapc == 2:
            self.density = np.swapaxes(self.density, 1, 2)
        elif maps == 2 and mapr == 1 and mapc == 3:
            self.density = np.swapaxes(self.density, 1, 2)
            self.density = np.swapaxes(self.density, 1, 0)
        elif maps == 1 and mapr == 2 and mapc == 3:
            self.density = np.swapaxes(self.density, 0, 2)
        else:
            raise ValueError("Density storage order ({:} {:} {:}) not supported.".format(maps, mapr, mapc))
        self.density = np.ascontiguousarray(self.density, dtype=np.float64)


class MRCParser(CCP4Parser):

    def _get_origin(self):
        origin_fields = 'xstart ystart zstart'.split()
        origin = [self.header[field] for field in origin_fields]
        return origin


def to_mrc(fid, volume, labels=[], fmt=None):

    if fmt is None:
        fmt = os.path.splitext(fid)[-1][1:]

    if fmt not in ('ccp4', 'mrc', 'map'):
        raise ValueError('Format is not recognized. Use ccp4, mrc, or map.')

    dtype = volume.array.dtype.name
    if dtype == 'int8':
        mode = 0
    elif dtype in ('int16', 'int32'):
        mode = 1
    elif dtype in ('float32', 'float64'):
        mode = 2
    else:
        raise TypeError("Data type ({:})is not supported.".format(dtype))

    if fmt in ('ccp4', 'map'):
        nxstart, nystart, nzstart = volume.offset
    else:
        nxstart, nystart, nzstart = [0, 0, 0]

    voxelspacing = volume.voxelspacing
    nz, ny, nx = volume.shape
    xl, yl, zl = volume.lattice_parameters
    alpha, beta, gamma = volume.angles
    mapc, mapr, maps = [1, 2, 3]
    ispg = 1
    nsymbt = 0
    lskflg = 0
    skwmat = [0.0]*9
    skwtrn = [0.0]*3
    fut_use = [0.0]*12
    if fmt == 'mrc':
        origin = volume.origin
    else:
        origin = [0, 0, 0]
    str_map = list('MAP ')
    if _BYTEORDER == 'little':
        machst = list('\x44\x41\x00\x00')
    elif _BYTEORDER == 'big':
        machst = list('\x44\x41\x00\x00')
    else:
        raise ValueError("Byteorder {:} is not recognized".format(byteorder))
    labels = [' '] * 800
    nlabels = 0
    min_density = volume.array.min()
    max_density = volume.array.max()
    mean_density = volume.array.mean()
    std_density = volume.array.std()

    with open(fid, 'wb') as out:
        out.write(_pack('i', nx))
        out.write(_pack('i', ny))
        out.write(_pack('i', nz))
        out.write(_pack('i', mode))
        out.write(_pack('i', nxstart))
        out.write(_pack('i', nystart))
        out.write(_pack('i', nzstart))
        out.write(_pack('i', nx))
        out.write(_pack('i', ny))
        out.write(_pack('i', nz))
        out.write(_pack('f', xl))
        out.write(_pack('f', yl))
        out.write(_pack('f', zl))
        out.write(_pack('f', alpha))
        out.write(_pack('f', beta))
        out.write(_pack('f', gamma))
        out.write(_pack('i', mapc))
        out.write(_pack('i', mapr))
        out.write(_pack('i', maps))
        out.write(_pack('f', min_density))
        out.write(_pack('f', max_density))
        out.write(_pack('f', mean_density))
        out.write(_pack('i', ispg))
        out.write(_pack('i', nsymbt))
        out.write(_pack('i', lskflg))
        for f in skwmat:
            out.write(_pack('f', f))
        for f in skwtrn:
            out.write(_pack('f', f))
        for f in fut_use:
            out.write(_pack('f', f))
        for f in origin:
            out.write(_pack('f', f))
        for c in str_map:
            out.write(_pack('c', c))
        for c in machst:
            out.write(_pack('c', c))
        out.write(_pack('f', std_density))
        # max 10 labels
        # nlabels = min(len(labels), 10)
        # TODO labels not handled correctly
        #for label in labels:
        #     list_label = [c for c in label]
        #     llabel = len(list_label)
        #     if llabel < 80:
        #
        #     # max 80 characters
        #     label = min(len(label), 80)
        out.write(_pack('i', nlabels))
        for c in labels:
            out.write(_pack('c', c))
        # write density
        modes = [np.int8, np.int16, np.float32]
        volume.array.astype(modes[mode]).tofile(out)


class XPLORParser(object):
    """
    Class for reading XPLOR volume files created by NIH-XPLOR or CNS.
    """

    def __init__(self, fid):

        if isinstance(fid, file):
            fname = fid.name
        elif isinstance(fid, str):
            fname = fid
            fid = open(fid)
        else:
            raise TypeError('Input should either be a file or filename')

        self.source = fname
        self._get_header()

    def _get_header(self):

        header = {}
        with open(self.source) as volume:
            # first line is blank
            volume.readline()

            line = volume.readline()
            nlabels = int(line.split()[0])

            label = [volume.readline() for n in range(nlabels)]
            header['label'] = label

            line = volume.readline()
            header['nx']      = int(line[0:8])
            header['nxstart'] = int(line[8:16])
            header['nxend']   = int(line[16:24])
            header['ny']      = int(line[24:32])
            header['nystart'] = int(line[32:40])
            header['nyend']   = int(line[40:48])
            header['nz']      = int(line[48:56])
            header['nzstart'] = int(line[56:64])
            header['nzend']   = int(line[64:72])

            line = volume.readline()
            header['xlength'] = float(line[0:12])
            header['ylength']   = float(line[12:24])
            header['zlength'] = float(line[24:36])
            header['alpha']    = float(line[36:48])
            header['beta'] = float(line[48:60])
            header['gamma']   = float(line[60:72])

            header['order'] = volume.readline()[0:3]

            self.header = header

    @property
    def voxelspacing(self):
        return self.header['xlength']/float(self.header['nx'])

    @property
    def origin(self):
        return [self.voxelspacing * x for x in
                [self.header['nxstart'], self.header['nystart'], self.header['nzstart']]]

    @property
    def density(self):
        with open(self.source) as volumefile:
            for n in range(2 + len(self.header['label']) + 3):
                volumefile.readline()
            nx = self.header['nx']
            ny = self.header['ny']
            nz = self.header['nz']

            array = np.zeros((nz, ny, nx), dtype=np.float64)

            xextend = self.header['nxend'] - self.header['nxstart'] + 1
            yextend = self.header['nyend'] - self.header['nystart'] + 1
            zextend = self.header['nzend'] - self.header['nzstart'] + 1

            nslicelines = int(np.ceil(xextend*yextend/6.0))
            for i in range(zextend):
                values = []
                nslice = int(volumefile.readline()[0:8])
                for m in range(nslicelines):
		    line = volumefile.readline()
		    for n in range(len(line)//12):
			value = float(line[n*12: (n+1)*12])
		        values.append(value)
                array[i, :yextend, :xextend] = np.float64(values).reshape(yextend, xextend)

        return array


def to_xplor(outfile, volume, label=[]):

    nz, ny, nx = volume.shape
    voxelspacing = volume.voxelspacing
    xstart, ystart, zstart = [int(round(x)) for x in volume.start]
    xlength, ylength, zlength = volume.dimensions
    alpha = beta = gamma = 90.0

    nlabel = len(label)
    with open(outfile,'w') as out:
        out.write('\n')
        out.write('{:>8d} !NTITLE\n'.format(nlabel+1))
	# CNS requires at least one REMARK line
	out.write('REMARK\n')
        for n in range(nlabel):
            out.write(''.join(['REMARK ', label[n], '\n']))

        out.write(('{:>8d}'*9 + '\n').format(nx, xstart, xstart + nx - 1,
                                             ny, ystart, ystart + ny - 1,
                                             nz, zstart, zstart + nz - 1))
        out.write( ('{:12.5E}'*6 + '\n').format(xlength, ylength, zlength,
                                                alpha, beta, gamma))
        out.write('ZYX\n')
        #FIXME very inefficient way of writing out the volume ...
        for z in range(nz):
            out.write('{:>8d}\n'.format(z))
            n = 0
            for y in range(ny):
                for x in range(nx):
                    out.write('%12.5E'%volume.array[z,y,x])
                    n += 1
                    if (n)%6 is 0:
                        out.write('\n')
            if (nx*ny)%6 > 0:
                out.write('\n')
        out.write('{:>8d}\n'.format(-9999))
        out.write('{:12.4E} {:12.4E} '.format(volume.array.mean(), volume.array.std()))
