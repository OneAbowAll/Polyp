import os
import multiprocessing
import time
from multiprocessing.queues import Queue

import PIL
import cv2
import numpy as np
import PIL.Image as Image

import segmentator.regions as Regions
from segmentator.generator import RegionGenerator
from segmentator.regionmap import RegionMap

SOURCE_IMAGES_FOLDER = r"S:\2022-01-OCDA-FL1S-P1-CROP\IMAGES"
PSEUDOLABEL_FOLDER = r"S:\2022-01-OCDA-FL1S-P1-CROP\output"

OUTPUT_PATH = r"S:\ritm_output"

NO_BG = True
WORKERS_AMOUNT = 4

DEBUG_PRINT = False

def generateRegionMap(files_to_proces, regions_output : Queue):
    for file_name in files_to_proces:
        full_res_image_path = os.path.join(SOURCE_IMAGES_FOLDER, file_name)
        if os.path.isdir(full_res_image_path):
            continue

        file_name = file_name.split(".")[0]
        label_image_path = os.path.join(PSEUDOLABEL_FOLDER, f"PseudoLabel_{file_name}.png")

        if not os.path.exists(label_image_path):
            continue

        if os.path.exists(os.path.join(OUTPUT_PATH, "output", f"{file_name}.png")):
            continue

        #Load photos and process them ----------------------------------------------------------
        full_res_image_file = PIL.Image.open(full_res_image_path)
        full_res_image_file.apply_transparency()

        psuedolabel_image_file = PIL.Image.open(label_image_path)
        psuedolabel_image_file.apply_transparency()

        full_res_size = (full_res_image_file.width, full_res_image_file.height)
        low_res_size = (full_res_image_file.width // 2, full_res_image_file.height // 2)

        psuedolabel_image_file = psuedolabel_image_file.resize(low_res_size, resample=PIL.Image.Resampling.NEAREST)
        real_image = full_res_image_file.resize(low_res_size, resample=PIL.Image.Resampling.LANCZOS)

        real_image = np.array(real_image)
        psuedolabel_image = np.array(psuedolabel_image_file)

        label_image, regions, type_colors = Regions.getRegions(psuedolabel_image)
        regions_output.put(RegionMap(file_name, real_image, label_image, regions, type_colors))

        full_res_image_file.close()
        psuedolabel_image_file.close()

    regions_output.put(None) #Segnala al consumatore che questo thread ha finito di lavorare

if __name__ == '__main__':

    start_time = time.time()

    regGenerator = RegionGenerator()
    photos = os.listdir(SOURCE_IMAGES_FOLDER)

    chunk_size = len(photos) // WORKERS_AMOUNT

    workers = []
    regionsQueue = multiprocessing.Queue(maxsize=WORKERS_AMOUNT)
    finished_workers = 0
    finished_files = 0
    for i in range(WORKERS_AMOUNT):
        (a, b) = (i*chunk_size, ((i+1)*chunk_size)if i != WORKERS_AMOUNT-1 else len(photos))

        worker_target = photos[a:b]
        worker = multiprocessing.Process(target=generateRegionMap, args=(worker_target, regionsQueue))

        workers.append(worker)
        worker.start()

    while finished_workers < WORKERS_AMOUNT:
        start_time_region = time.time()
        regionMap = regionsQueue.get()

        if regionMap is None:
            finished_workers += 1
            continue

        #Generate new segmentation--------------------------------------------------------------
        result = regGenerator.generate(regionMap, regionMap.types_colors)
        #---------------------------------------------------------------------------------------

        #Upscale segmentation and apply to original image---------------------------------------
        full_res_image_path = os.path.join(SOURCE_IMAGES_FOLDER, f"{regionMap.name}.jpg")
        full_res_image_file = PIL.Image.open(full_res_image_path)

        result = cv2.resize(result, (full_res_image_file.width, full_res_image_file.height), interpolation=cv2.INTER_LANCZOS4)

        if not NO_BG:
            full_res_image_file.apply_transparency()
            full_res_image =  np.array(full_res_image_file)

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
        else:
            output = result[:, :, :3]

        pil_img = Image.fromarray(output)
        pil_img.save(os.path.join(r"S:\ritm_output\output", f"{regionMap.name}.png"))
        full_res_image_file.close()

        finished_files += 1
        print(f"Time to process image: {time.time() - start_time_region} seconds.")
        print(f"Currently at {finished_files}/{len(photos)}")
        print(f"RegionMapQueue status: {regionsQueue.qsize()}")

    for w in workers:
        w.join()

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds.")
    #---------------------------------------------------------------------------------------

    input("Premi qualsiasi tasto per chiudere...")

    """Setup PyGame --------------------------------------------------------------------------
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
    """
