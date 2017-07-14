from __future__ import division
import numpy as np

from .volume import Volume
from .transformer import Transformer


class Validator(object):

    def __init__(self, xmap, resolution):
        self.xmap = xmap
        self.resolution = resolution

    def rscc(self, structure, rmask=1, mask_structure=None, simple=True):
        model_map = Volume.zeros_like(self.xmap)
        model_map.set_spacegroup("P1")
        if mask_structure is None:
            transformer = Transformer(structure, model_map, simple=simple)
        else:
            transformer = Transformer(mask_structure, model_map, simple=simple)
        transformer.mask(rmask)
        mask = model_map.array > 0
        model_map.array.fill(0)
        if mask_structure is not None:
            transformer = Transformer(structure, model_map, simple=simple)
        transformer.density()

        target_values = self.xmap.array[mask]
        model_values = model_map.array[mask]
        target_values -= target_values.mean()
        target_values /= target_values.std()
        model_values -= model_values.mean()
        model_values /= model_values.std()
        corr = (target_values * model_values).sum() / mask.sum()
        return corr

    def fisher_z_difference(self, structure1, structure2, rmask=1, simple=True):
        # Create mask of combined structures
        combined = structure1.combine(structure2)
        model_map = Volume.zeros_like(self.xmap)
        transformer = Transformer(combined, model_map)
        transformer.mask(rmask)
        mask = model_map.array > 0
        nvoxels = mask.sum()
        mv = nvoxels * self.xmap.voxel_volume

        # Get density values of xmap, and both structures
        target_values = self.xmap.array[mask]
        transformer = Transformer(structure1, model_map, simple=simple)
        model_map.array.fill(0)
        transformer.density()
        model1_values = model_map.array[mask]
        transformer = Transformer(structure2, model_map, simple=simple)
        model_map.array.fill(0)
        transformer.density()
        model2_values = model_map.array[mask]

        # Get correlation score for structure
        target_values -= target_values.mean()
        target_values /= target_values.std()

        model1_mean = model1_values.mean()
        model1_std = model1_values.std()
        corr1 = ((target_values * 
                 (model1_values - model1_mean)).sum() / 
                 model1_std) / nvoxels
        model2_mean = model2_values.mean()
        model2_std = model2_values.std()
        corr2 = ((target_values * (model2_values - model2_mean)).sum() / 
                 model2_std) / nvoxels
        # Transform to Fisher Z-score
        sigma = 1.0 / np.sqrt(mv / self.resolution - 3)
        fisher1 = 0.5 * np.log((1 + corr1) / (1 - corr1))
        fisher2 = 0.5 * np.log((1 + corr2) / (1 - corr2))
        return (fisher2 - fisher1) / sigma
