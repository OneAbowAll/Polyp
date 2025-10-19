import fiftyone as fo
import fiftyone.zoo as zoo
import fiftyone.utils.yolo as utils
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import sys
import time

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
 


def main():

        print("------START DETECTION")
        dataset_dir = sys.argv[1]
        image_name = sys.argv[2]
        output_dir = sys.argv[3]    

        print(f"Processing {dataset_dir}, {image_name}, {output_dir}")
        

        scissor_name = image_name[:-4] + "_scissor" + image_name[-4:]
        
        scissor_path = os.path.join(dataset_dir, scissor_name)

        gbbox = [0.0, 0.0, 0.0, 0.0]  # full image
        txt_file = os.path.join(dataset_dir, image_name[:-4] + ".txt")
        with open(txt_file, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            gbbox = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]

        start_time = time.time()
        with Image.open(os.path.join(dataset_dir, image_name)) as img:
            _, height = img.size
        print(f"Image load took {time.time() - start_time:.2f} seconds")

        gbbox[1] = height - gbbox[1]  # flip y coordinate
        gbbox[3] = height - gbbox[3]

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
            
            start_time = time.time()
            cropped_img = img.crop((left, top, right, bottom))
            print(f"Cropping took {time.time() - start_time:.2f} seconds")

            start_time = time.time()
            cropped_img.save(os.path.join(dataset_dir, scissor_name))
            print(f"Saving took {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        model = YOLO("best-nano.pt")
        print(f"Model load took {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        results = model.predict(os.path.join(dataset_dir, scissor_name), conf=0.8, imgsz=(sx,sy), retina_masks=True)
        print(f"Prediction took {time.time() - start_time:.2f} seconds")

        # Delete the cropped image file after prediction
        start_time = time.time()
        scissor_file_path = os.path.join(dataset_dir, scissor_name)
        if os.path.exists(scissor_file_path):
          os.remove(scissor_file_path)
        print(f"Deleting cropped image took {time.time() - start_time:.2f} seconds")


        #results = model.predict(os.path.join(dataset_dir, image_name), conf=0.6, imgsz=(4000,6000), retina_masks=True)
        start_time = time.time()
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
                            mask = mask[0, offy:offy+h, offx:offx+w]
                            mask = mask * 255
                            mask = mask.astype(np.uint8)
                            pil_img = Image.fromarray(mask)
                            pil_img.save(filename)

        print(f"Mask saving took {time.time() - start_time:.2f} seconds")
        print("------END DETECTION")

if __name__ == '__main__':
    main()