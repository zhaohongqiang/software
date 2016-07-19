#!/usr/bin/env python3

import cv2
import shm

from collections import namedtuple

from vision import options
from vision.vision_common import red, green, blue, white, cyan, yellow, \
                                 draw_angled_arrow, \
                                 get_angle_from_rotated_rect
from vision.modules.base import ModuleBase


CONTOUR_HEURISTIC_LIMIT = 5
CONTOUR_SCALED_HEURISTIC_LIMIT = 2

options = [
     options.IntOption('block_size_cover', 1001, 0, 1500),
     options.IntOption('c_thresh_cover', 30, -100, 100),
     options.IntOption('blur_size_cover',  31, 1, 255, lambda x: x % 2 == 1),
     options.IntOption('block_size_cutout', 1001, 0, 1500),
     options.IntOption('c_thresh_cutout', 20, -100, 100),
     options.IntOption('blur_size_cutout',  31, 1, 255, lambda x: x % 2 == 1),
     options.DoubleOption('min_area_percent', 0.005, 0, 0.15),
     options.DoubleOption('min_rectangularity', 0.6, 0, 1),
     options.BoolOption('debugging', True)
]

class Bins(ModuleBase):
  def process(self, mat):

    """Currently a stub, below are the shm variables to set
    shm.bin1.x
    shm.bin1.y
    shm.bin1.covered
    shm.bin1.p

    shm.bin2.x
    shm.bin2.y
    shm.bin2.covered
    shm.bin2.p

    shm.handle.x
    shm.handle.y
    shm.handle.p
    """
    self.post('orig', mat)

    lab_image = cv2.cvtColor(mat, cv2.COLOR_RGB2LAB)
    lab_split = cv2.split(lab_image)

    lab_athreshed = cv2.adaptiveThreshold(lab_split[1], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                          self.options['block_size_cutout'], self.options['c_thresh_cutout'])
    lab_bthreshed = cv2.adaptiveThreshold(lab_split[2], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                          self.options['block_size_cover'], self.options['c_thresh_cover'])

    finalaThreshed =  lab_athreshed
    finalbThreshed =  lab_bthreshed

    if self.options['debugging']:
      self.post('lab a', lab_split[1])
      self.post('lab b', lab_split[2])
      self.post('finalaThreshed', finalaThreshed)
      self.post('finalbThreshed', finalbThreshed)

    blurreda = cv2.medianBlur(finalaThreshed, self.options['blur_size_cutout'])
    blurredb = cv2.medianBlur(finalbThreshed, self.options['blur_size_cover'])
    self.post('blurred_a', blurreda)
    self.post('blurred_b', blurredb)

    # TODO Maybe do eroding and dilating / morphology here?
    _, contours_cutout, __ = cv2.findContours(blurreda.copy(),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_size = mat.shape[0] * mat.shape[1]

    big_contours = []
    for contour in contours_cutout:
      area = cv2.contourArea(contour)
      if area >= self.options['min_area_percent'] * image_size:
        big_contours.append((contour, area))

    candidate_cutouts = sorted(big_contours, key=lambda x: -x[1])[:2]

    _, contours, __ = cv2.findContours(blurredb.copy(),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if self.options['debugging']:
      contoursMat = mat.copy()
      cv2.drawContours(contoursMat, contours, -1, green, 5)
      self.post("All contours", contoursMat)

    contourAreas = []
    for contour in contours:
      contourArea = cv2.contourArea(contour)
      if contourArea >= self.options['min_area_percent'] * image_size:
        contourAreas.append([contour, contourArea])
    contourAreas = sorted(contourAreas, key=lambda x: -x[1])[:CONTOUR_HEURISTIC_LIMIT]

    contour_info = namedtuple("contour_info", ["contour", "area", "rectangularity", "rect_area_prod", "center", "angle", "rotrect"])

    contourScores = []
    for c, a in contourAreas:
      rotrect = cv2.minAreaRect(c)
      center, (h, w), angle = rotrect

      rectangularity = a / (w*h)
      if rectangularity >= self.options['min_rectangularity']:
        contourScores.append(contour_info(c, a, rectangularity, rectangularity * a, center, angle, rotrect))
    contourScores = sorted(contourScores, key=lambda x: -x.rect_area_prod)[:CONTOUR_SCALED_HEURISTIC_LIMIT]

    results_g1 = shm.bin1.get()
    results_g2 = shm.bin2.get()
    results_g1.p = 0
    results_g2.p = 0
    bin_results = [results_g1, results_g2]

    if contourScores:
      filteredContoursMat = mat.copy()
      binContour = max(contourScores, key=lambda x: x.center[1]) # Zero is top-left of image

      cv2.drawContours(filteredContoursMat, [binContour.contour], -1, green, 5)
      bin_center = int(binContour.center[0]), int(binContour.center[1])
      cv2.circle(filteredContoursMat, bin_center, 5, white, -1)

      shm_angle = get_angle_from_rotated_rect(binContour.rotrect)

      draw_angled_arrow(filteredContoursMat, bin_center, shm_angle)

      self.post("Filtered contours", filteredContoursMat)

      results_g1.p = 1.0
      results_g1.x = bin_center[0]
      results_g1.y = bin_center[1]
      results_g1.angle = shm_angle
      results_g1.covered = 1

    contoursMat2 = mat.copy()
    for i, (contour, area) in enumerate(candidate_cutouts):
      moms = cv2.moments(contour)
      center = int(moms["m10"] / moms["m00"]), int(moms["m01"] / moms["m00"])

      cv2.drawContours(contoursMat2, [contour], -1, yellow, 5)
      cv2.circle(contoursMat2, center, 5, white, -1)

      # Only if not a cover, reverse sketchily.
      # TODO FIX.
      bin_results[1-i].p = 1.0
      bin_results[1-i].covered = 0
      bin_results[1-i].x = center[0]
      bin_results[1-i].y = center[1]

    self.post("Cutout contours", contoursMat2)

    self.fill_single_camera_direction(results_g1)
    self.fill_single_camera_direction(results_g2)
    shm.bin1.set(results_g1)
    shm.bin2.set(results_g2)

if __name__ == '__main__':
    Bins('downward', options)()
