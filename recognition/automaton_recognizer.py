
from ast import Invert
from re import search
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import os
import sys
import numpy as np
import time
import csv
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert
from line_chaser import chase_line, Direction

import tensorflow as tf

from tensorflow.python.keras.backend import get_session


def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)

def detection_on_image(image):

  model_path = 'models/bigModel.h5'
  model = models.load_model(model_path, backbone_name='resnet50')

  draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = preprocess_image(image)
  image, scale = resize_image(image)
  
  labels_to_names = {3: "arrow", 0: "label", 1: "state", 2: "final state"}
  cv2.imwrite("output/input_for_classifier.png", image)
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  boxes /= scale
  for box, score, label in zip(boxes[0], scores[0], labels[0]):

      if score < 0.3:
          break

      color = label_color(label)
      b = box.astype(int)
      draw_box(draw, b, color=color)
      caption = "{} {:.3f}".format(labels_to_names[label], score)
      draw_caption(draw, b, caption)

  detected_img =cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
  cv2.imwrite(f'output/classification.png', detected_img)

  detected_states = []
  detected_arrows = []
  detected_labels = []

  for i in range(len(boxes[0])):
    if(scores[0][i] >= 0.3):
      if labels_to_names[labels[0][i]] == "state" and scores[0][i] >= 0.8:
        detected_states.append({
          "id": len(detected_states),
          "bndbox": boxes[0][i],
          "final": False,
          "score": scores[0][i]
        })
      elif labels_to_names[labels[0][i]] == "final state" and scores[0][i] >= 0.8:
        detected_states.append({
          "id": len(detected_states),
          "bndbox": boxes[0][i],
          "final": True,
          "score": scores[0][i]
        })
      elif labels_to_names[labels[0][i]] == "arrow":
        detected_arrows.append({
          "id": len(detected_arrows),
          "bndbox": boxes[0][i],
          "score": scores[0][i],
          "pointingFrom": None,
          "pointingAt": None
        })
      elif labels_to_names[labels[0][i]] == "label":
        detected_labels.append({
          "id": len(detected_labels),
          "bndbox": boxes[0][i],
          "score": scores[0][i],
          "meaning": ""
        })
  return detected_states, detected_arrows, detected_labels

def do_overlap(topLeftRec1, bottomRightRec1, topLeftRec2, bottomRightRec2):
    
    # if rectangle has area 0, no overlap
    if topLeftRec1[0] == bottomRightRec1[0] or topLeftRec1[1] == bottomRightRec1[1] or topLeftRec2[0] == bottomRightRec2[0] or topLeftRec2[1] == bottomRightRec2[1]:
      return False
     
    # If one rectangle is on left side of other
    if topLeftRec1[0] > bottomRightRec2[0] or topLeftRec2[0] > bottomRightRec1[0]:
      return False
 
    # If one rectangle is above other
    if bottomRightRec1[1] < topLeftRec2[1] or bottomRightRec2[1] < topLeftRec1[1]:
      return False
 
    return True

def getAngleBetweenPoints(point1, point2):
  point2moved = (point2[0] - point1[0], point2[1] - point1[1])
  point2normalized = point2moved / np.linalg.norm(point2moved)

  if point1[0] < point2[0]:
    return 360 - np.arccos(np.dot(point2normalized,[0,1]))*180 / np.pi
  else:
    return np.arccos(np.dot(point2normalized,[0,1]))*180 / np.pi

def getBoxCenter(bndbox):
  return ((bndbox[0] + bndbox[2]) / 2, (bndbox[1] + bndbox[3]) /2)

def findArrowConnection(image, arrow, detected_states):

  angle = getAngleBetweenPoints(getBoxCenter(arrow["bndbox"]), getBoxCenter(detected_states[arrow["pointingAt"]]["bndbox"]))
  line_chaser_results = []
  if(angle < 22.5 or angle > 337.5):
    direction = Direction.NORTH
  
  elif(22.5 < angle < 67.5):
    direction = Direction.NORTH_EAST

  elif(67.5 < angle < 112.5):
    direction = Direction.EAST

  elif(112.5 < angle < 157.5):
    direction = Direction.SOUTH_EAST

  elif(157.5 < angle < 202.5):
    direction = Direction.SOUTH

  elif(202.5 < angle < 247.5):
    direction = Direction.SOUTH_WEST

  elif(247.5 < angle < 292.5):
    direction = Direction.WEST

  elif(292.5 < angle < 337.5):
    direction = Direction.NORTH_WEST
  else:
    print("this should not happen")
  arrow_box_radius = 0


  for i in range(int(arrow["bndbox"][0]-arrow_box_radius), int(arrow["bndbox"][2]+arrow_box_radius)):
    for j in range(int(arrow["bndbox"][1]-arrow_box_radius),int(arrow["bndbox"][3]+arrow_box_radius)):
      if image[j,i] != 0:
        line_chaser_results.append(chase_line(image.copy(), (i,j), direction))
     



  if(line_chaser_results != []):
    endPoint = (max(line_chaser_results,key=lambda item:item[1]))
    print(endPoint)
    radius = 20
    for state in detected_states:
      search_box = {
        "minX": state["bndbox"][0] - radius,
        "minY": state["bndbox"][1] - radius,
        "maxX": state["bndbox"][2] + radius,
        "maxY": state["bndbox"][3] + radius
      }

      if(search_box["minX"] < endPoint[0][0] < search_box["maxX"] and search_box["minY"] < endPoint[0][1] < search_box["maxY"]):
        
        return state["id"]
    return "start"

def main():
  keras.backend.set_session(get_session())

  input_file_path = sys.argv[1]

  image = cv2.imread(input_file_path)
  
  image = np.pad(image, ((50,50),(50,50),(0,0)),constant_values=(255))

  detected_states, detected_arrows, detected_labels = detection_on_image(image)

  state_association_radius = 20


  skeleton_image = skeletonize(invert(image))
  skeleton_image=cv2.cvtColor(skeleton_image,cv2.COLOR_BGR2GRAY)
  skeleton_image[skeleton_image>128] = 255
  delete_radius = 5
  for state in detected_states:
    state_center = (int((state["bndbox"][0] + state["bndbox"][2])/2), int((state["bndbox"][1] + state["bndbox"][3])/2))
    print(state_center)
    axes_lengths = (int((state["bndbox"][2] - state["bndbox"][0])/2) + delete_radius,int((state["bndbox"][3] - state["bndbox"][1])/2) + delete_radius)
    print(axes_lengths)
    cv2.ellipse(skeleton_image, state_center, axes_lengths, 0, 0, 360, (0,0,0), -1)
    cv2.putText(skeleton_image, str(state["id"]), (int(state["bndbox"][0] + 100),int(state["bndbox"][1] + 100)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

  
  cv2.imwrite("output/input_line_chaser.png", skeleton_image)
  
  for state in detected_states:
    search_box = {
      "minX": state["bndbox"][0] - state_association_radius,
      "minY": state["bndbox"][1] - state_association_radius,
      "maxX": state["bndbox"][2] + state_association_radius,
      "maxY": state["bndbox"][3] + state_association_radius
    }

    for arrow in detected_arrows:
      
      if do_overlap((search_box["minX"], search_box["minY"]),(search_box["maxX"], search_box["maxY"]),
        (arrow["bndbox"][0], arrow["bndbox"][1]), (arrow["bndbox"][2],arrow["bndbox"][3])):
        arrow["pointingAt"] = state["id"]
        arrow["pointingFrom"] = findArrowConnection(skeleton_image,arrow,detected_states)
        print(arrow)

    

      
if __name__ == "__main__":
    main()