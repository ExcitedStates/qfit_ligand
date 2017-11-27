import logging
logger = logging.getLogger(__name__)

import numpy as np

from .transformer import Transformer
from .volume import Volume

class MapScaler(object):

    def __init__(self, xmap, mask_radius=1.5, cutoff=0):
        self.xmap = xmap
        self.mask_radius = mask_radius
        self.cutoff = cutoff
        self._model_map = Volume.zeros_like(xmap)

    def __call__(self, structure):
        logger.info("Scaling map")
        # smax doesnt have any impact here.
        transformer = Transformer(structure, self._model_map, simple=True, rmax=3)
        logger.info("Making mask.")
        transformer.mask(self.mask_radius)
        mask = self._model_map.array > 0
        transformer.reset()
        logger.info("Initializing")
        transformer.initialize()
        transformer.density()

        # Set values below cutoff to zero, to penalize the solvent more
        if self.cutoff is not None:
            mean = self.xmap.array.mean()
            std = self.xmap.array.std()
            cutoff_value = self.cutoff * std + mean

        xmap_masked = self.xmap.array[mask]
        model_masked = self._model_map.array[mask]
        model_masked_mean = model_masked.mean()
        xmap_masked_mean = xmap_masked.mean()
        model_masked -= model_masked_mean
        xmap_masked -= xmap_masked_mean
        scaling_factor = np.dot(model_masked, xmap_masked) / np.dot(xmap_masked, xmap_masked)
        logger.info("Scaling factor: {:.2f}".format(scaling_factor))

        self.xmap.array -= xmap_masked_mean
        self.xmap.array *= scaling_factor
        self.xmap.array += model_masked_mean

        # Subtract the receptor density from the map
        self.xmap.array -= self._model_map.array

        if self.cutoff is not None:
            cutoff_value = (cutoff_value - xmap_masked_mean) * scaling_factor + model_masked_mean
            cutoff_mask = self.xmap.array < cutoff_value
            self.xmap.array[cutoff_mask] = 0

        #self.xmap.tofile('map_scaled.ccp4')
