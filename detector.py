import sys
import fiftyone as fo
import fiftyone.zoo as zoo
import fiftyone.utils.yolo as utils
from ultralytics import YOLO
from PIL import Image
import os
import glob
import numpy as np
import subprocess
import memory

def atBorder(bbox):

     # bbox format is XYWH
     x1 = bbox[0]
     y1 = bbox[1]
     x2 = x1 + bbox[2]
     y2 = y1 + bbox[3]

     if x1 < 0.01 or y1 < 0.01:
         return True

     if x2 > 0.99 or y2 > 0.99:
         return True



def applyYOLOseg2(dataset_dir, output_dir):

    images_names = [os.path.basename(x) for x in glob.glob(os.path.join(dataset_dir, '*.JPG')) if '_scissor' not in os.path.basename(x)]
    images_names.sort()

    #images_names = [os.path.basename(x) for x in glob.glob(os.path.join(dataset_dir, '*.jpg'))]


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

#    model = YOLO("best-nano.pt")

    image_number = len(images_names)

    for k in range(1,image_number):

        image_name = images_names[k]

        memory.get_gpu_memory("before subprocess")

        subprocess.run([
        sys.executable, "detect_one.py",
        dataset_dir,
        image_name,
        output_dir
        ], check=True)
        
#        proc = subprocess.Popen([
#                "python", "detect_one.py",
#                dataset_dir,
#                image_name,
#                output_dir 
#        ])
        
#        proc.wait()
        memory.get_gpu_memory("after subprocess")

        """
        scissor_name = image_name[:-4] + "_scissor" + image_name[-4:]
        
        scissor_path = os.path.join(dataset_dir, scissor_name)

        gbbox = [0.0, 0.0, 0.0, 0.0]  # full image
        txt_file = os.path.join(dataset_dir, image_name[:-4] + ".txt")
        with open(txt_file, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            gbbox = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]

        gbbox[1] = 4000 - gbbox[1]  # flip y coordinate
        gbbox[3] = 4000 - gbbox[3]

        gbbox[1], gbbox[3] = gbbox[3], gbbox[1]

        assert gbbox[1] < gbbox[3]

        sx = gbbox[2] - gbbox[0]
        sy = gbbox[3] - gbbox[1]

        if not os.path.exists(scissor_path):
            img_path = os.path.join(dataset_dir, image_name)
            img = Image.open(img_path)
            width, height = img.size
            left = int(gbbox[0] )
            top = int(gbbox[1] )
            right = int(gbbox[2] )
            bottom = int(gbbox[3])
            cropped_img = img.crop((left, top, right, bottom))
        
            cropped_img.save(os.path.join(dataset_dir, scissor_name))
            
        
        memory.get_gpu_memory("before predict")
        results = model.predict(os.path.join(dataset_dir, scissor_name), conf=0.8, imgsz=(sx,sy), retina_masks=True)
        memory.get_gpu_memory("after predict")
           
        # Delete the cropped image file after prediction
        scissor_file_path = os.path.join(dataset_dir, scissor_name)
        if os.path.exists(scissor_file_path):
            os.remove(scissor_file_path)


        #results = model.predict(os.path.join(dataset_dir, image_name), conf=0.6, imgsz=(4000,6000), retina_masks=True)

        if results is not None:
            if len(results) > 0:
            if results[0].masks is not None:

                n = len(results[0].masks)
                for i in range(n):

                mask = results[0].masks[i]
                bbox = results[0].boxes[i].xyxy[0].cpu().numpy()
                confidence = float(results[0].boxes[i].conf.cpu().numpy())
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(round(bbox[2] - bbox[0]))
                h = int(round(bbox[3] - bbox[1]))

                if not atBorder([x / float(sx), y / float(sy), w /float(sx), h / float(sy)]):
                    offx =  x
                    offy =  y
                    suffix = "_{:05d}_{:05d}_{:.5f}.png".format(int(gbbox[0])+offx, int(gbbox[1])+offy, confidence)
                    filename = scissor_name[:-12] + suffix
                    filename = os.path.join(output_dir, filename)
                    mask = mask.data.cpu().numpy()
                    mask2 = mask[0].copy()
                    mask = mask[0, offy:offy+h, offx:offx+w]
                    mask = mask * 255
                    mask = mask.astype(np.uint8)
                    pil_img = Image.fromarray(mask)
                    pil_img.save(filename)
        """




# def outputHighScore(output_dir, high_score_dir):
#
#     if not os.path.exists(high_score_dir):
#         os.mkdir(high_score_dir)
#
#     images_names = [os.path.basename(x) for x in glob.glob(os.path.join(output_dir, '*.jpg'))]
#
#     for image_name in images_names:
#         pass
#
# def createDataset():
#
#     name1 = "dataset-polyps-train"
#     name2 = "dataset-polyps-val"
#
#     try:
#         data_path = "C:/detector/dataset/training/images"
#         labels_path = "C:/detector/dataset/training/annotations.json"
#
#         # Create the dataset
#         dataset1 = fo.Dataset.from_dir(
#             dataset_type=fo.types.COCODetectionDataset,
#             data_path=data_path,
#             labels_path=labels_path,
#             name=name1,
#         )
#
#     except:
#         dataset1 = fo.load_dataset(name1)
#
#     try:
#         data_path = "C:/detector/dataset/validation/images"
#         labels_path = "C:/detector/dataset/validation/annotations.json"
#
#         # Create the dataset
#         dataset2 = fo.Dataset.from_dir(
#             dataset_type=fo.types.COCODetectionDataset,
#             data_path=data_path,
#             labels_path=labels_path,
#             name=name2,
#         )
#
#     except:
#         dataset2 = fo.load_dataset(name2)
#
#     return dataset1, dataset2
#
def apply_yolo(path_to_images, path_to_output):
     # Load a model
     model = YOLO("best-nano.pt")
     applyYOLOseg2(path_to_images, path_to_output)

