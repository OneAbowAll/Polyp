import numpy as np
import scipy

import log
from segmentator.region import Region


class RegionMap:
    def __init__(self, image, label_map, regions: dict[int, Region]):
        self.image = image
        self.label_map = label_map
        self.regions = regions

    def getRegions(self) -> list[Region]:
        return list(self.regions.values())

    def getRegion (self, region_id: int) -> Region:
        return self.regions[region_id]

    def extractFromImage(self, region_id):
        """
        Estrai dalla foto originale una regione
        """
        region = self.regions[region_id]
        [minY, minX, maxY, maxX] = region.bbox
        return self.image[minY:maxY, minX:maxX].copy()

    def extractMask(self, region_id):
        """
        Estrai dalla pseudo_label la maschera della region con id = region_id.
        Se nella region e' presente un altra maschera con id diverso questa verra' esclusa.
        """
        region = self.regions[region_id]
        [minY, minX, maxY, maxX] = region.bbox
        cropped = self.label_map[minY:maxY, minX:maxX].copy()

        binary_mask = cropped != region.id
        cropped[binary_mask] = 0
        return cropped

    def generateDistanceMask(self, region_id):
        """
        Generated distance mask will be normalized, max_value = 1, min_value = 0.
        """

        mask = self.extractMask(region_id)
        distance_mask = scipy.ndimage.distance_transform_edt(mask)

        max = np.max(distance_mask)
        if max == 0:
            log.print_info(f"Max for region {region_id} -> {max}")
        return distance_mask / max

    def generateSDF(self, region_id):
        mask = self.extractMask(region_id)
        inverse_mask = np.where(mask == region_id, 0, 2)

        inside_distance_mask = scipy.ndimage.distance_transform_edt(mask)
        outside_distance_mask = scipy.ndimage.distance_transform_edt(inverse_mask)

        return -inside_distance_mask + outside_distance_mask