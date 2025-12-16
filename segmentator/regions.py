import numpy as np
from skimage.measure import label, regionprops

from segmentator.region import Region
from segmentator.regionmap import RegionMap


def getRegions(image):
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)

    indexStart = 0
    label_map = np.zeros(image.shape[:2], dtype=np.uint16)
    for color in unique_colors:
        if np.all(color == 0):
            continue

        binary_mask = np.all(image==color, axis=-1)
        current_mask, mask_count = label(binary_mask, return_num=True)

        valid_pixels = current_mask > 0 #Restituisce un array/matrice booleana che sara' vera dove c'e' qualcosa e falsa dove c'e' lo sfondo
        label_map[valid_pixels] = current_mask[valid_pixels] + indexStart #Fare label_map[valid_pixel] va a lavorare automaticamente su gli indici contenenti True, e' molto strano ma e' vero!

        indexStart += mask_count

    result: dict[int, Region] = {}
    regions = regionprops(label_map)

    padding_amount = 20
    for region in regions:
        if region.area > 200:
            [minY, minX, maxY, maxX] = region.bbox
            newBbox = (
                max(minY - padding_amount, 0),
                max(minX - padding_amount, 0),
                min(maxY + padding_amount, image.shape[0]-1),
                min(maxX + padding_amount, image.shape[1]-1)
            )
            result[region.label] = Region(region.label, padding_amount, newBbox)

    return label_map, result