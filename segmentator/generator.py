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
from skimage.color import rgb2gray, gray2rgb

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

        result_all_segments = regionMap.image.copy()
        result_accumulate = regionMap.image.copy()

        for region in regions:
            log.print_info(f"Segmenting Region_Id: {region.id}...")
            [minY, minX, maxY, maxX] = region.bbox

            #Crop image to region
            cropped = regionMap.extractFromImage(region.id)
            distance_mask = regionMap.generateSDF(region.id)
            accumulate_mask = np.zeros((cropped.shape[0], cropped.shape[1]), dtype=np.uint8)

            self.predictor.set_input_image(cropped)

            #Genera una maschera iniziale che ritm possa usare
            pseudo_mask = regionMap.extractMask(region.id)
            init_mask = np.zeros(pseudo_mask.shape, dtype=np.int32)

            init_mask[pseudo_mask > 0] = 1
            init_mask = init_mask.astype(np.float32)
            init_mask = torch.from_numpy(init_mask).unsqueeze(0).unsqueeze(0)
            init_mask = init_mask.to("cuda:0")

            #Genera piu' segmentazioni e salva il risultato nella accumulate_mask
            for k in range(10):
                main_size = max(region.height, region.width)

                #TODO: i positive points devono essere spreaddati, in qualche modo
                positives = self.generatePositivePoints(distance_mask, region,4, main_size/20)
                negatives = self.generateNegativePoints(distance_mask, region,8, main_size/8)

                #Genera i punti
                for p in positives:
                    click = clicker.Click(is_positive=True, coords=p)
                    ritm_clicker.add_click(click)

                for p in negatives:
                    click = clicker.Click(is_positive=False, coords=p)
                    ritm_clicker.add_click(click)

                #Esegui RITM
                pred = self.predictor.get_prediction(ritm_clicker, prev_mask=init_mask)
                segm_mask = pred > 0.5

                #Accumula segmentazione generata
                accumulate_mask[segm_mask] += 1


                #Disegna la segmentazione sull'immagine originale (anche qui serve per debugging)
                result_all_segments[minY:maxY, minX:maxX][segm_mask, 0] = random.randint(0, 255)
                result_all_segments[minY:maxY, minX:maxX][segm_mask, 1] = random.randint(0, 255)
                result_all_segments[minY:maxY, minX:maxX][segm_mask, 2] = random.randint(0, 255)

                #Porta la prediction in un formato "disegnabile" (principalmente per debugging)
                output = cropped.copy()
                output[segm_mask, 0] = 255
                output[segm_mask, 1] = 255
                output[segm_mask, 2] = 255
                pil_img2 = Image.fromarray(output)

                #Disegna segmentazione ee punti generati alla k-esima iterazione
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

                #Reset per la prossima iterazione
                ritm_clicker.reset_clicks()

            #Usiamo i "voti" delle segmentazioni del ritm per generare una segmentazione adeguata
            average = cropped.copy()

            final_mask = accumulate_mask > 5
            average[final_mask, 0] = 255
            average[final_mask, 1] = 255
            average[final_mask, 2] = 255

            #Salva per dubugging
            pil_img2 = Image.fromarray(average)
            pil_img2.save(rf"S:\ritm_output\{region.id}\average.png")

            #Aggiungi al risultato finale evidenziando le varie segmentazioni
            [minY, minX, maxY, maxX] = region.bbox
            selection_alpha = 0.6
            selection_color = np.array([random.randint(1, 255), random.randint(1, 255), random.randint(1, 255)])

            regionOfInterest = result_accumulate[minY:maxY, minX:maxX]
            originalColor = regionOfInterest[final_mask]

            finalColor = (originalColor*(1-selection_alpha)) + (selection_color*selection_alpha)
            regionOfInterest[final_mask] = finalColor.astype(np.uint8)


        result_image = Image.fromarray(result_all_segments)
        result_image.save(r"S:\ritm_output\complete.png")

        pil_img2 = Image.fromarray(result_accumulate)
        pil_img2.save(rf"S:\ritm_output\result_accumulate.png")

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

                    if distance_map[p]>15:
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

    def generateSegmentationFromClicks(self, image, edited, region, positive :list[Tuple[int, int]], negative :list[Tuple[int, int]]):
        [minY, minX, maxY, maxX] = region.bbox

        #Crop image to region
        cropped = image[minY:maxY, minX:maxX]

        self.predictor.set_input_image(edited)

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
