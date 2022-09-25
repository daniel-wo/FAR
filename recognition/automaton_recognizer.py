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
from skimage import data, filters
from skimage.util import invert
from line_chaser import chase_line, Direction
from PIL import Image, ImageDraw
import math
import tensorflow as tf

from tensorflow.python.keras.backend import get_session


# Parameters for classification and recognition purposes
state_confidence_treshold = 0.8
label_confidence_treshold = 0.4
arrow_confidence_treshold = 0.4
state_association_radius = 75
delete_radius = 5
arrow_end_to_state_radius = 50
arrow_line_chaser_box_radius = 10
arrow_min_length = 50

def preprocessImageForRecognition(image):

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  #gauss = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)

  binary = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,15,11)

  cv2.imwrite(f"output/preprocessed.png", binary)

  return binary

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


# Evaluates the recognition model for states, final states, labels and arrowheads on the input image and returns dictionaries containing found entities
def detection_on_image(image):

  preprocessImageForRecognition(image)

  model_path = 'models/augment13epoch.h5'
  model = models.load_model(model_path, backbone_name='resnet50')


  draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  print(image.shape)
  image = preprocessImageForRecognition(image)
  cv2.imwrite(f"output/test1.png", image)

  image = np.stack((image,image,image), axis=2)
  cv2.imwrite(f"output/test2.png", image)

  print(image.shape)
  image = preprocess_image(image)
  image, scale = resize_image(image)
  
  labels_to_names = {3: "arrow", 0: "label", 1: "state", 2: "final state"}
  cv2.imwrite(f"output/input_for_classifier.png", image)
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  boxes /= scale

  for box, score, label in zip(boxes[0], scores[0], labels[0]):

      if score < 0.4:
          break

      if((label == 0) and score < label_confidence_treshold):
        continue

      if((label == 3) and score < arrow_confidence_treshold):
        continue

      if((label == 1 or label == 2) and score < state_confidence_treshold):
        continue
        
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
    if(scores[0][i] >= 0.4):
      if labels_to_names[labels[0][i]] == "state" and scores[0][i] >= state_confidence_treshold:
        
        for j in range(len(boxes[0])):
            if(labels[0][j] == 2 
              and do_overlap((boxes[0][i][0],boxes[0][i][1]),(boxes[0][i][2],boxes[0][i][3]), (boxes[0][j][0],boxes[0][j][1]),(boxes[0][j][2],boxes[0][j][3]))
              and scores[0][j] >= state_confidence_treshold):
              break
        else:
          detected_states.append({
            "id": len(detected_states),
            "bndbox": boxes[0][i],
            "final": False,
            "score": scores[0][i],
            "label": None
          })
      elif labels_to_names[labels[0][i]] == "final state" and scores[0][i] >= state_confidence_treshold:
        detected_states.append({
          "id": len(detected_states),
          "bndbox": boxes[0][i],
          "final": True,
          "score": scores[0][i],
          "label": None
        })
      elif labels_to_names[labels[0][i]] == "arrow" and scores[0][i] >= arrow_confidence_treshold:
        detected_arrows.append({
          "id": len(detected_arrows),
          "bndbox": boxes[0][i],
          "score": scores[0][i],
          "pointingFrom": None,
          "pointingAt": None,
          "label": None
        })
      elif labels_to_names[labels[0][i]] == "label" and scores[0][i] >= label_confidence_treshold:
        detected_labels.append({
          "id": len(detected_labels),
          "bndbox": boxes[0][i],
          "score": scores[0][i],
          "meaning": "",
          "isStateLabel": False,
          "closestTransition": None,
          "closestTransitionDistance": math.inf
        })
  return detected_states, detected_arrows, detected_labels

# Checks if 2 rectangles overlap with each other and returns true if that's the case
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

# Returns the angle between point 1 and point 2 taking the vertical axis of point 1 as basis
def getAngleBetweenPoints(point1, point2):
  point2moved = (point2[0] - point1[0], point2[1] - point1[1])
  point2normalized = point2moved / np.linalg.norm(point2moved)

  if point1[0] < point2[0]:
    return 360 - np.arccos(np.dot(point2normalized,[0,1]))*180 / np.pi
  else:
    return np.arccos(np.dot(point2normalized,[0,1]))*180 / np.pi

# Returns bounding box center coordinates
def getBoxCenter(bndbox):
  return (int((bndbox[0] + bndbox[2]) / 2), int((bndbox[1] + bndbox[3]) /2))

def assignLabelsToStates(detected_labels, detected_states):
  for state in detected_states:
    for label in detected_labels:
      if do_overlap((state["bndbox"][0],state["bndbox"][1]),(state["bndbox"][2],state["bndbox"][3]),(label["bndbox"][0],label["bndbox"][1]),(label["bndbox"][2],label["bndbox"][3])):
        label["isStateLabel"] = True
        state["label"] = label["id"]

  return detected_labels

# Takes and arrowhead and tries to find a state from where the arrow came from by tracing the shaft with the line chaser algorithm
def findArrowConnection(image, arrow, detected_states, output_image, detected_labels):

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


  for i in range(int(arrow["bndbox"][0]-arrow_line_chaser_box_radius), int(arrow["bndbox"][2]+arrow_line_chaser_box_radius)):
    for j in range(int(arrow["bndbox"][1]-arrow_line_chaser_box_radius),int(arrow["bndbox"][3]+arrow_line_chaser_box_radius)):
      if image[j,i] != 0:
        line_chaser_results.append(chase_line(image.copy(), (i,j), direction, output_image, detected_labels, arrow["id"]))
    

  line_chaser_results = list(filter(lambda r: r[1] > arrow_min_length, line_chaser_results))

  if(line_chaser_results != []):
    endPoint = (max(line_chaser_results,key=lambda item:item[1]))
    for state in detected_states:
      search_box = {
        "minX": state["bndbox"][0] - arrow_end_to_state_radius,
        "minY": state["bndbox"][1] - arrow_end_to_state_radius,
        "maxX": state["bndbox"][2] + arrow_end_to_state_radius,
        "maxY": state["bndbox"][3] + arrow_end_to_state_radius
      }

      if(search_box["minX"] < endPoint[0][0] < search_box["maxX"] and search_box["minY"] < endPoint[0][1] < search_box["maxY"]):
        
        return state["id"]
    return "start"

def getLabelMeanings(detected_labels, image,i):

  for label in detected_labels:
    padded_image = np.pad(image[int(label["bndbox"][1] - 10) : int(label["bndbox"][3] + 10),int(label["bndbox"][0] - 10) : int(label["bndbox"][2] + 10),:], ((10,10),(10,10),(0,0)),constant_values=(255))
    cv2.imwrite(f"output/{label['id']}.png", padded_image)



def main(input_file_path, i = 0):

  if not os.path.exists(f"output"):
    os.mkdir(f"output")
  keras.backend.set_session(get_session())


  image = cv2.imread(input_file_path)
  
  image = np.pad(image, ((50,50),(50,50),(0,0)),constant_values=(255))

  detected_states, detected_arrows, detected_labels = detection_on_image(image)

  detected_labels = assignLabelsToStates(detected_labels, detected_states)
  
  getLabelMeanings(detected_labels, image, i)


  #Create binary image
  binary = preprocessImageForRecognition(image)
  #Convert to boolean array
  binary = binary > 128
  #Skeletonize image
  skeleton_image = skeletonize(invert(binary))

  #Convert back to opencv format
  skeleton_image = skeleton_image*255
  skeleton_image = skeleton_image.astype(np.uint8)

  cv2.imwrite(f"output/preprocessed2.png", skeleton_image)

  for state in detected_states:
    state_center = (int((state["bndbox"][0] + state["bndbox"][2])/2), int((state["bndbox"][1] + state["bndbox"][3])/2))
    axes_lengths = (int((state["bndbox"][2] - state["bndbox"][0])/2) + delete_radius,int((state["bndbox"][3] - state["bndbox"][1])/2) + delete_radius)
    cv2.ellipse(skeleton_image, state_center, axes_lengths, 0, 0, 360, (0,0,0), -1)
    cv2.putText(skeleton_image, str(state["id"]), getBoxCenter(state["bndbox"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

  
  cv2.imwrite(f"output/input_line_chaser.png", skeleton_image)
  
  line_chaser_output_image = Image.new(mode = "RGB", size=(image.shape[1],image.shape[0]), color="#ffffff")

  if os.path.exists(f'output/transitions.txt'):
    with open(f'output/transitions.txt', 'w') as f:
      f.write("")

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
        arrow["pointingFrom"] = findArrowConnection(skeleton_image,arrow,detected_states, line_chaser_output_image, detected_labels)
        
        with open(f'output/transitions.txt', 'a') as f:
          f.write(str(arrow))
          f.write('\n')

  for label in detected_labels:
    if not label["isStateLabel"]:
      detected_arrows[label["closestTransition"]]["label"] = label["id"]

  print("Transitions:")
  for arrow in detected_arrows:
    print(arrow)
  print("States:")
  for state in detected_states:
    print(state)
  print("Labels:")
  for label in detected_labels:
    print(label)
  line_chaser_output_image.save(f"output/line_chaser_output.png")

    
if __name__ == "__main__":
  input_file_path = sys.argv[1]
  main(input_file_path)
      
#for i in range(0,50):
#  main(f"../datasets/automata/images/{i}.png", i)