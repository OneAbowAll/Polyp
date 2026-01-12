import os
import random
from typing import Tuple

import PIL
import cv2
import numpy as np
import pygame
import scipy
import skimage
from PIL import ImageDraw
import PIL.Image as Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pygame import Color
from skimage.color import label2rgb, rgb2gray, gray2rgb

from skimage.io import imread

import log
import segmentator.regions as regions
from segmentator.generator import RegionGenerator
from segmentator.regionmap import RegionMap

label_path = r"S:\test_labels2.png"
real_photo_path = r"example.png"
output_path = r"S:\ritm_output"

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
    image.save(os.path.join(output_path, "debug_regions.png"))

def outputLabelMap(label_map, filepath):
    image_float = label2rgb(label_map, bg_label=0, bg_color=(0, 0, 0))
    image_uint8 = (image_float * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)
    pil_image.save(os.path.join(output_path, filepath))

def reMap(value, maxInput, minInput, maxOutput, minOutput):

    value = maxInput if value > maxInput else value
    value = minInput if value < minInput else value

    inputSpan = maxInput - minInput
    outputSpan = maxOutput - minOutput

    scaledThrust = float(value - minInput) / float(inputSpan)

    return minOutput + (scaledThrust * outputSpan)

def lerp(a, b, v):
    return a + v * (b - a)
def outputDistanceMask(distance_mask, filepath):
    shape = distance_mask.shape
    result = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


    for i in range(shape[0]):
        for j in range(shape[1]):
            v = lerp(0, 255, distance_mask[i][j])
            result[i][j] = [v, v, v]

    pil_image = Image.fromarray(result)
    pil_image.save(os.path.join(output_path, filepath))

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
    pil_image.save(os.path.join(output_path, filepath))

def outputImage(image, filepath):
    image = Image.fromarray(image)
    image.save(os.path.join(output_path, filepath))

if __name__ == '__main__':
    regGenerator = RegionGenerator()

    #Load photos and process them ----------------------------------------------------------
    full_res_image = PIL.Image.open(r"S:\2022-01-OCDA-FL1S-P1-CROP\IMAGES\NK1_2646.JPG")
    full_res_image.apply_transparency()

    full_res_size = (full_res_image.width, full_res_image.height)

    psuedolabel_image = PIL.Image.open(r"S:\2022-01-OCDA-FL1S-P1-CROP\output\PseudoLabel_NK1_2646.png")
    psuedolabel_image.apply_transparency()

    psuedolabel_image = psuedolabel_image.resize((full_res_image.width//2, full_res_image.height//2), resample=PIL.Image.Resampling.NEAREST)
    real_image = full_res_image.resize((full_res_image.width//2, full_res_image.height//2), resample=PIL.Image.Resampling.LANCZOS)

    real_image = np.array(real_image)
    psuedolabel_image = np.array(psuedolabel_image)
    full_res_image = np.array(full_res_image)

    """
    if psuedolabel_image.shape[-1] == 4:
        psuedolabel_image = psuedolabel_image[..., :3]  # Remove alpha

    if real_image.shape[-1] == 4:
        real_image = real_image[..., :3]  # Remove alpha
    """
    #---------------------------------------------------------------------------------------

    #Generate Labels -----------------------------------------------------------------------
    log.print_info("Gathering all regions...")

    label_image, regions = regions.getRegions(psuedolabel_image)
    regionMap = RegionMap(real_image, label_image, regions)

    if False:
        outputLabelMap(label_image, "label_map.png")
        outputRegionMap(psuedolabel_image, regionMap)
        log.print_info(f"Found {len(regions)} regions.")
        for region in regions.values():
            print(region)
        #---------------------------------------------------------------------------------------

        #Debug Output---------------------------------------------------------------------------
        for region in regionMap.getRegions():
            if not os.path.isdir(rf"S:\ritm_output\{region.id}"):
                os.mkdir(rf"S:\ritm_output\{region.id}")

            outputImage(regionMap.extractFromImage(region.id), rf"{region.id}\image.png")
            outputDistanceMask(regionMap.generateDistanceMask(region.id), rf"{region.id}\distance_mask.png")
            outputSDF(regionMap.generateSDF(region.id), rf"{region.id}\sdf.png")
            outputLabelMap(regionMap.extractMask(region.id), rf"{region.id}\mask.png")
        #---------------------------------------------------------------------------------------

    #Generate new segmentation--------------------------------------------------------------
    result = regGenerator.generate(full_res_image, regionMap)
    #---------------------------------------------------------------------------------------

    #Upscale segmentation and apply to original image---------------------------------------
    result = cv2.resize(result, full_res_size, interpolation=cv2.INTER_LANCZOS4)
    pil_img2 = Image.fromarray(result)
    pil_img2.save(rf"S:\ritm_output\result_accumulate_high.png")

    #Apply mask to original image
    result_mask = result[:, :, 3] > 0
    output = full_res_image.copy()

    maskColor = result[result_mask]
    alpha = maskColor[:, 3]
    maskColor = maskColor[:, :3]

    originalColor = output[result_mask]

    alpha = alpha / 255.0
    alpha = alpha[:, None] #Questo serve per avere questo array della dimensione giusta per l'operazione a riga sotto.

    finalColor = (originalColor*(1-alpha)) + (maskColor*alpha)
    output[result_mask] = finalColor.astype(np.uint8)
    pil_img2 = Image.fromarray(output)
    pil_img2.save(rf"S:\ritm_output\output.png")
    #---------------------------------------------------------------------------------------

    #Setup PyGame --------------------------------------------------------------------------
    W, H = 1350, 900
    DELTA_TIME = 0

    #Setup PyGame
    pygame.init()
    pygame.display.set_caption("Polyp Detector")
    SCREEN = pygame.display.set_mode((W, H))
    CLOCK = pygame.time.Clock()

    w, h, d = real_image.shape

    region = regionMap.getRegion(region_id=1)
    positive_points:list[Tuple[int, int]] = []
    negative_points:list[Tuple[int, int]] = []

    [minY, minX, maxY, maxX] = region.bbox
    cropped = real_image[minY:maxY, minX:maxX]
    pil_image = Image.fromarray(cropped)
    base_texture = pygame.image.frombytes(pil_image.tobytes(), pil_image.size, "RGB")

    screen_center = (W / 2, H / 2)
    img_rect = (screen_center[0] - base_texture.width/2, screen_center[1] - base_texture.height/2, base_texture.width, base_texture.height)

    running = True
    while running:
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouseX, mouseY = event.pos
                    positive_points.append((mouseY - img_rect[1], mouseX - img_rect[0]))

                if event.button == 3:
                    mouseX, mouseY = event.pos
                    negative_points.append((mouseY - img_rect[1], mouseX - img_rect[0]))

                result = regGenerator.generateSegmentationFromClicks(real_image, cropped, region, positive_points, negative_points)
                base_texture = pygame.image.frombytes(result.tobytes(), pil_image.size, "RGB")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    positive_points = []
                    negative_points = []

                    base_texture = pygame.image.frombytes(pil_image.tobytes(), pil_image.size, "RGB")

        # fill the screen with a color to wipe away anything from last frame
        SCREEN.fill("purple")

        # RENDER YOUR GAME HERE
        SCREEN.blit(base_texture, img_rect)

        for p in positive_points:
            pygame.draw.circle(SCREEN, (0, 255, 0), (p[1] + img_rect[0], p[0] + img_rect[1]), 3)

        for p in negative_points:
            pygame.draw.circle(SCREEN, (255, 0, 0), (p[1] + img_rect[0], p[0] + img_rect[1]), 3)

        # flip() the display to put your work on screen
        pygame.display.flip()
        CLOCK.tick(60)  # limits FPS to 60

    pygame.quit()

