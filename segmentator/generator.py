import math
import os.path
import random
from typing import Tuple

import numpy as np
import skimage
import torch
from PIL import Image, ImageDraw
from jinja2.compiler import generate
from mpmath.math2 import sqrt2
from pyglm.glm import clamp

import log
from models.isegm.inference import clicker
from models.isegm.inference import utils as ritmutils
from models.isegm.inference.predictors import get_predictor
from segmentator.region import Region
from segmentator.regionmap import RegionMap


def loadNetwork():
    predictor = None
    predictor_params = {'brs_mode': 'NoBRS'}

    model_path = 'models/ritm_corals.pth'

    if not torch.cuda.is_available():
        print("CUDA NOT AVAILABLE!")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    try:
        ritm_net = ritmutils.load_is_model(model_path, device, cpu_dist_maps=False)
        ritm_net.to(device)
        # initialize predictor
        predictor = get_predictor(ritm_net, device=device, **predictor_params)

    except Exception as e:
        print("Could not load the Ritm network. You might need to run update.py.")

    return predictor

class RegionGenerator:
    def __init__(self):
        self.predictor = loadNetwork()

    def generate(self, regionMap: RegionMap):
        #Test with first region
        regions = regionMap.getRegions()
        ritm_clicker = clicker.Clicker()  # handles clicked point (original code of ritm)

        for region in regions:
            #Crop image to region
            cropped = regionMap.extractFromImage(region.id)
            distance_mask = regionMap.generateSDF(region.id)

            self.predictor.set_input_image(cropped)

            init_mask = None

            for k in range(10):
                positives = self.generatePositivePoints(distance_mask, region,3)

                main_size = max(region.height, region.width)
                negatives = self.generateNegativePoints(distance_mask, region,3, main_size/4)

                for p in positives:
                    click = clicker.Click(is_positive=True, coords=p)
                    ritm_clicker.add_click(click)

                for p in negatives:
                    click = clicker.Click(is_positive=False, coords=p)
                    ritm_clicker.add_click(click)

                pred = self.predictor.get_prediction(ritm_clicker, prev_mask=init_mask)

                # from prediction to segmentation mask
                segm_mask = pred > 0.5
                output = cropped.copy()
                output[segm_mask, 0] = 255
                output[segm_mask, 1] = 255
                output[segm_mask, 2] = 255

                pil_img2 = Image.fromarray(output)

                # drawing context TODO: REMOVE THIS OR MAKE IT OPTIONAL FOR DEBUGGING
                draw = ImageDraw.Draw(pil_img2)

                for click in ritm_clicker.get_clicks():
                    x = click.coords[1]
                    y = click.coords[0]
                    if click.is_positive:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)

                    draw.circle((x, y), 2.0, fill=color)

                if not os.path.isdir(rf"S:\ritm_output\{region.id}"):
                    os.mkdir(rf"S:\ritm_output\{region.id}")

                txt = rf"S:\ritm_output\{region.id}\test_{k}.png"
                pil_img2.save(txt)

                # reset clicks
                ritm_clicker.reset_clicks()

    def generatePositivePoints(self, distance_map, region:Region, amount, minDistance = 0):
        result = []

        for _ in range(amount):
            ok = False
            while not ok:
                ok = True
                p = self.generatePositivePoint(distance_map, region)

                for other in result:
                    dy = other[0] - p[0]
                    dx = other[1] - p[1]

                    distance = math.sqrt((dy*dy)+(dx*dx))

                    if distance < minDistance:
                        ok = False
                        break

                if ok:
                    result.append(p)

        return result

    def generateNegativePoints(self, distance_map, region: Region, amount, minDistance=0):
        result = []

        for _ in range(amount):
            ok = False
            while not ok:
                ok = True
                p = self.generateNegativePoint(distance_map, region)

                for other in result:
                    dy = other[0] - p[0]
                    dx = other[1] - p[1]

                    distance = math.sqrt((dy * dy) + (dx * dx))

                    if distance < minDistance:
                        ok = False
                        break

                if ok:
                    result.append(p)

        return result
    def generatePositivePoint(self, distance_map, region: Region):
        """
        Takes in input a normalized distance map, the close is to the center = 1 viceversa for 0.
        """

        #Generate random point in region with non 0 value
        p = region.randomPointInRegion()

        while distance_map[p] > 0:
            p = region.randomPointInRegion()
        return p

    def generateNegativePoint(self, distance_map, region: Region):
        """
        Takes in input a normalized distance map, the close is to the center = 1 viceversa for 0.
        """
        p = region.randomPointInRegion()

        while  distance_map[p] < 0:
            p = region.randomPointInRegion()

        return p

    def generateSegmentationFromClicks(self, image, label_map, region, positive :list[Tuple[int, int]], negative :list[Tuple[int, int]]):
        [minY, minX, maxY, maxX] = region.bbox

        #Crop image to region
        cropped = image[minY:maxY, minX:maxX]

        self.predictor.set_input_image(cropped)

        init_mask = None
        ritm_clicker = clicker.Clicker()  # handles clicked point (original code of ritm)

        for p in positive:
            click = clicker.Click(is_positive=True, coords=(p[0], p[1]))
            ritm_clicker.add_click(click)

        for p in negative:
            click = clicker.Click(is_positive=False, coords=p)
            ritm_clicker.add_click(click)

        pred = self.predictor.get_prediction(ritm_clicker, prev_mask=init_mask)

        segm_mask = pred > 0.5
        output = cropped.copy()
        output[segm_mask, 0] = 255
        output[segm_mask, 1] = 255
        output[segm_mask, 2] = 255
        return Image.fromarray(output)
