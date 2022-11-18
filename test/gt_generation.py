# According to tampered region mask to generate splicing trace ground truth.
from PIL import Image
import numpy as np
import sys
import os
import cv2 as cv
import matplotlib.pyplot as plt

def see(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


if __name__=='__main__':
    tampered_region_folder = r'' # The folder which saves the tampered region mask.
    save_folder = r'' # A folder to save splicing trace.
    if not os.path.exists(save_folder) or not os.path.exists(tampered_region_folder):
        print('The folder not exists.')
        sys.exit(0)
    for idx, item in enumerate(os.listdir(tampered_region_folder)):
        region_mask_path = os.path.join(tampered_region_folder,item)
        region_mask = Image.open(region_mask_path).convert('L')
        region_mask = np.array(region_mask, dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        _gt_1 = cv.dilate(region_mask, kernel)
        _gt_1 = np.array(_gt_1, dtype=np.uint8)
        outer_edge = _gt_1 - region_mask  # outer edge
        _gt_2 = cv.erode(region_mask, kernel)
        _gt_2 = np.array(_gt_2, dtype=np.uint8)
        inner_edge = region_mask - _gt_2  # innder edge
        dou_edge = np.where(_gt_2, 50, 0) + np.where(outer_edge, 100, 0) + np.where(inner_edge, 255, 0)
        # We take the outer edge as the single-pixel edge of authentic, and the inner as the edge of splicing region.
        # At the same time, the inner part of splicing region we also label in dou_edge.
        # If you test on splicing trace ground truth, just use "np.where((dou_edge==255)|(dou_edge==100),255,0)"
        # If you want test on tampered region mask, just use "np.where((dou_edge==50)|(dou_edge==100),255,0)"
        # see(dou_edge)

        # Then you can save the splicing trace ground truth.
        dou_edge = Image.fromarray(np.uint8(dou_edge))
        dou_edge.save(os.path.join(save_folder,item))