import os
import random
from typing import Tuple

import numpy as np
from PIL import Image as Image, ImageDraw
from skimage.color import label2rgb
from skimage.measure import label, regionprops

from image_test import OUTPUT_PATH
from segmentator.region import Region
from segmentator.regionmap import RegionMap


def getRegions(image):
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    color_list:list[Tuple[int, int, int]] = (unique_colors.tolist())
    region_to_type: dict[int, int] = {}

    indexStart = 0
    label_map = np.zeros(image.shape[:2], dtype=np.uint16)
    for color in unique_colors:
        if np.all(color == 0):
            continue

        binary_mask = np.all(image==color, axis=-1)
        current_mask, mask_count = label(binary_mask, return_num=True)

        valid_pixels = current_mask > 0 #Restituisce un array/matrice booleana che sara' vera dove c'e' qualcosa e falsa dove c'e' lo sfondo
        label_map[valid_pixels] = current_mask[valid_pixels] + indexStart #Fare label_map[valid_pixel] va a lavorare automaticamente su gli indici contenenti True, e' molto strano ma e' vero!

        for i in range(0, mask_count):
            c = color.tolist()
            region_to_type[indexStart + i + 1] = color_list.index(c)

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
            result[region.label] = Region(region.label, region_to_type[region.label], padding_amount, newBbox)

    return label_map, result, unique_colors


def outputRegionMap(image, regionMap : RegionMap):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    region_list = regionMap.getRegions()
    for region in region_list:
        miny, minx, maxy, maxx = region.bbox
        random_color = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        draw.rectangle(
            xy = [(minx, miny), (maxx, maxy)],
            outline = random_color,
            width = 2
        )

        draw.text(
            xy=((minx + maxx)/2, (miny + maxy)/2.0),
            text = f"{region.id}",
            fill = random_color
        )
    image.save(os.path.join(OUTPUT_PATH, "debug_regions.png"))


def outputLabelMap(label_map, filepath):
    image_float = label2rgb(label_map, bg_label=0, bg_color=(0, 0, 0))
    image_uint8 = (image_float * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)
    pil_image.save(os.path.join(OUTPUT_PATH, filepath))

def outputDistanceMask(distance_mask, filepath):
    shape = distance_mask.shape
    result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


    for i in range(shape[0]):
        for j in range(shape[1]):
            v = lerp(0, 255, distance_mask[i][j])
            result[i][j] = [v, v, v]

    pil_image = Image.fromarray(result)
    pil_image.save(os.path.join(OUTPUT_PATH, filepath))


def outputSDF(sdf, filepath):
    shape = sdf.shape

    min = np.min(sdf)
    max = np.max(sdf)

    print(f"min: {min}, max: {max}")
    result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


    for i in range(shape[0]):
        for j in range(shape[1]):
            v1 = reMap(sdf[i][j], max, min, 255, 0)
            v2 = reMap(sdf[i][j], max, min, 0, 255)
            result[i][j] = [v1, 0, v2]

    pil_image = Image.fromarray(result)
    pil_image.save(os.path.join(OUTPUT_PATH, filepath))


def outputImage(image, filepath):
    image = Image.fromarray(image)
    image.save(os.path.join(OUTPUT_PATH, filepath))

def reMap(value, maxInput, minInput, maxOutput, minOutput):

    value = maxInput if value > maxInput else value
    value = minInput if value < minInput else value

    inputSpan = maxInput - minInput
    outputSpan = maxOutput - minOutput

    scaledThrust = float(value - minInput) / float(inputSpan)

    return minOutput + (scaledThrust * outputSpan)


def lerp(a, b, v):
    return a + v * (b - a)