import json, sys, os
from shapely.geometry import Polygon
from other import *
import cv2
import numpy as np


if __name__ == "__main__":
    masked = False
    mask_regions = get_mask_regions("videos/ISLab/mask/mask.json", "ISLab-13.jpg")
    vid = cv2.VideoCapture("videos/ISLab/input/ISLab-13.mp4")    
    idx = 0
    while True:
        _, im = vid.read()
        if im is None:  # read finishes if there are no more frames
            break
        # idx += 1
        if idx in list(range(1080, 1231, 30)):
            if masked:
                alpha = 0.3
                int_coords = lambda x: np.array(x).round().astype(np.int32)
                overlay = im.copy()
                for poly2 in mask_regions:
                    poly2 = Polygon(poly2)
                    exterior = [int_coords(poly2.exterior.coords)]
                    cv2.fillPoly(overlay, exterior, color=(0, 255, 255))
                cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
            cv2.imwrite(os.path.join('results',str(idx) + '.jpg'), im)
            # break
        idx += 1
    print("done")

