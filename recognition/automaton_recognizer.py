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
import torch
from torchvision import transforms


from tensorflow.python.keras.backend import get_session

import line_chaser


# Parameters for classification and recognition purposes
state_confidence_treshold = 0.75
arrow_confidence_treshold = 0.4
arrowhead_to_state_association_radius = 30
delete_radius = 5
arrowshaft_end_to_state_association_radius = 30
arrowhead_line_chaser_starting_box_radius = 0
arrow_min_length = 20
label_min_size = 20
label_max_size = 500

def preprocessImageForRecognition(image):

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  #spgauss = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)

  binary = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,15,11)

  cv2.imwrite(f"output/preprocessed_image.png", binary)

  return binary

def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


# Evaluates the recognition model for states, final states, labels and arrowheads on the input image and returns dictionaries containing found entities
def element_detection_on_image(image):

  preprocessImageForRecognition(image)

  element_model_path = 'models/9epoch.h5'
  model = models.load_model(element_model_path, backbone_name='resnet50')

  draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  image = preprocessImageForRecognition(image)

  image = np.stack((image,image,image), axis=2)

  image = preprocess_image(image)
  image, scale = resize_image(image)
  
  with open('./classes.csv', mode='r') as infile:
            reader = csv.reader(infile)
            labels_to_names = {int(rows[1]):rows[0] for rows in reader}
  cv2.imwrite(f"output/input_for_classifier.png", image)
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  boxes /= scale

  for box, score, label in zip(boxes[0], scores[0], labels[0]):

      if score < min(arrow_confidence_treshold,state_confidence_treshold):
        break
    
      if((labels_to_names[label] == "arrow") and score < arrow_confidence_treshold):
        continue

      if((labels_to_names[label] == "state" or labels_to_names[label] == "final state") and score < state_confidence_treshold):
        continue
        
      color = label_color(label)
      b = box.astype(int)
      draw_box(draw, b, color=color)
      caption = "{} {:.3f}".format(labels_to_names[label], score)
      draw_caption(draw, b, caption)

  detected_img =cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
  cv2.imwrite(f'output/classified_image.png', detected_img)


  detected_states = []
  detected_arrows = []

  for i in range(len(boxes[0])):
    if(scores[0][i] >= 0.4):
      if labels_to_names[labels[0][i]] == "state" and scores[0][i] >= state_confidence_treshold:
        # Do not append this state if the state was already recognized as final state
        for j in range(len(boxes[0])):
            if(labels[0][j] != -1
              and labels_to_names[labels[0][j]] == "final_state"
              and do_overlap((boxes[0][i][0],boxes[0][i][1]),(boxes[0][i][2],boxes[0][i][3]), (boxes[0][j][0],boxes[0][j][1]),(boxes[0][j][2],boxes[0][j][3]))
              and scores[0][j] >= state_confidence_treshold):
              continue
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
          "label": []
        })
    
  return detected_states, detected_arrows

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

def assignLabels(detected_states, detected_arrows, detected_labels):
  for state in detected_states:
    for label in detected_labels:
      if do_overlap((state["bndbox"][0],state["bndbox"][1]),(state["bndbox"][2],state["bndbox"][3]),(label["bndbox"][0],label["bndbox"][1]),(label["bndbox"][2],label["bndbox"][3])) and label["isStateLabel"]:
        state["label"] = label["meaning"]

  for arrow in detected_arrows:
    for label in detected_labels:
      if not label["isStateLabel"] and arrow["id"] == label["closestTransition"]:
        arrow["label"].append(label["meaning"])
  return detected_labels

# Takes and arrowhead and tries to find a state from where the arrow came from by tracing the shaft with the line chaser algorithm
def findArrowConnection(image, arrow, detected_states, output_image, detected_labels = []):

  line_chaser_results = []
  #Find the starting direction by determining the relative angle between the state center and the arrow head center
  angle = getAngleBetweenPoints(getBoxCenter(arrow["bndbox"]), getBoxCenter(detected_states[arrow["pointingAt"]]["bndbox"]))
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


  for i in range(int(arrow["bndbox"][0]-arrowhead_line_chaser_starting_box_radius), int(arrow["bndbox"][2]+arrowhead_line_chaser_starting_box_radius)):
    for j in range(int(arrow["bndbox"][1]-arrowhead_line_chaser_starting_box_radius),int(arrow["bndbox"][3]+arrowhead_line_chaser_starting_box_radius)):
      if image[j,i] != 0:
        line_chaser_results.append(((i,j),chase_line(image.copy(), (i,j), direction, output_image, detected_labels, arrow["id"])))
    
  line_chaser_results = list(filter(lambda r: r[1][1] > arrow_min_length, line_chaser_results))

  if(line_chaser_results != []):
    maximum_length_line = (max(line_chaser_results,key=lambda item:item[1][1]))
    for state in detected_states:
      search_box = {
        "minX": state["bndbox"][0] - arrowshaft_end_to_state_association_radius,
        "minY": state["bndbox"][1] - arrowshaft_end_to_state_association_radius,
        "maxX": state["bndbox"][2] + arrowshaft_end_to_state_association_radius,
        "maxY": state["bndbox"][3] + arrowshaft_end_to_state_association_radius
      }
      chase_line(image, maximum_length_line[0], direction, output_image, detected_labels, arrow["id"])

      if(search_box["minX"] < maximum_length_line[1][0][0] < search_box["maxX"] and search_box["minY"] < maximum_length_line[1][0][1] < search_box["maxY"]):
        return state["id"]
      
  return "start"

def findLabels(image, detected_states):

  #Get labels as the left over connected components after making deletions
  _, _, component_bboxes, _ = cv2.connectedComponentsWithStats(image)
  #Filter out everything that exceeds label max size or does not reach label min size
  component_bboxes = list(filter(lambda r: label_max_size > r[2] > label_min_size or label_max_size > r[3] > label_min_size, component_bboxes))

  detected_labels = []
  #For each label extract the original image excerpt and classify it
  for i, bbox in enumerate(component_bboxes):

    #Read and pad original image
    image = cv2.imread(input_file_path)
    image = np.pad(image, ((50,50),(50,50),(0,0)),constant_values=(255))

    gray = cv2.cvtColor(invert(image), cv2.COLOR_BGR2GRAY)

    #Extract label excerpt
    label_image = Image.fromarray(squarify(gray[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]],0))

    #Resize and save for visualization
    label_image = label_image.resize((28,28))
    label_image.save(f"output/labels/{i}.png")

    #Flip and rotate image to convert it to EMNIST format
    image = np.rot90(np.fliplr(np.array(label_image))).copy()

    #Normalize, convert to tensor and evaluate the model on the label image        
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    image = normalize_transform(image)
    model = torch.load("models/epoch19.pt", map_location=torch.device('cpu'))
    model.eval()              # turn the model to evaluate mode
    with torch.no_grad():     # does not calculate gradient
      class_index = model(image[None,...]).argmax()
    #Get character mapping
    with open('./emnist-balanced-mapping.csv', mode='r') as infile:
            reader = csv.reader(infile)
            mapping = {int(rows[0]):int(rows[1]) for rows in reader}

    bndbox = [bbox[0],bbox[1],bbox[0] + bbox[2], bbox[1] + bbox[3]]
    detected_labels.append({
      "id": i,
      "meaning": chr(mapping[class_index.item()]),
      "bndbox": bndbox,
      "isStateLabel": False,
      "closestTransitionDistance": math.inf
    })
  
  offset = len(detected_labels)
  for i, state in enumerate(detected_states):
    id = offset + i
    #Read and pad original image
    image = cv2.imread(input_file_path)
    image = np.pad(image, ((50,50),(50,50),(0,0)),constant_values=(255))

    gray = cv2.cvtColor(invert(image), cv2.COLOR_BGR2GRAY)

    #Extract label excerpt
    x_margin = (state["bndbox"][2] - state["bndbox"][0])*0.2
    y_margin = (state["bndbox"][3] - state["bndbox"][1])*0.2

    label_image = Image.fromarray(squarify(gray[
      int(state["bndbox"][1]+y_margin):
      int(state["bndbox"][3]-y_margin), 
      int(state["bndbox"][0]+x_margin): 
      int(state["bndbox"][2]-x_margin)],0))

    #Resize and save for visualization
    label_image = label_image.resize((28,28))
    label_image.save(f"output/labels/{id}.png")

    #Flip and rotate image to convert it to EMNIST format
    image = np.rot90(np.fliplr(np.array(label_image))).copy()

    #Normalize, convert to tensor and evaluate the model on the label image        
    normalize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    image = normalize_transform(image)
    model = torch.load("models/epoch19.pt", map_location=torch.device('cpu'))
    model.eval()              # turn the model to evaluate mode
    with torch.no_grad():     # does not calculate gradient
      class_index = model(image[None,...]).argmax()
    #Get character mapping
    with open('./emnist-balanced-mapping.csv', mode='r') as infile:
            reader = csv.reader(infile)
            mapping = {int(rows[0]):int(rows[1]) for rows in reader}

    detected_labels.append({
      "id": id,
      "meaning": chr(mapping[class_index.item()]),
      "bndbox": state["bndbox"],
      "isStateLabel": True,
    })  
  return detected_labels


  
def squarify(M,val):
    (a,b)=M.shape
    
    if a>b:
        padding=((3,3),(3+int((a-b)/2),3+int((a-b)/2)))
    else:
        padding=((3+int((b-a)/2),3+int((b-a)/2)),(3,3))
    return np.pad(M,padding,mode='constant',constant_values=val) 

def main(input_file_path, i = 0):

  if not os.path.exists(f"output"):
    os.mkdir(f"output")
  if not os.path.exists(f"output/labels"):
    os.mkdir(f"output/labels")
  keras.backend.set_session(get_session())


  image = cv2.imread(input_file_path)
  
  image = np.pad(image, ((50,50),(50,50),(0,0)),constant_values=(255))


  detected_states, detected_arrows = element_detection_on_image(image)

  #Create binary image
  binary = preprocessImageForRecognition(image)
  #Convert to boolean array
  binary = binary > 128
  #Skeletonize image
  skeleton_image = skeletonize(invert(binary))

  #Convert back to opencv format
  skeleton_image = skeleton_image*255
  skeleton_image = skeleton_image.astype(np.uint8)

  #Save skeletonized image
  cv2.imwrite(f"output/skeletonized_image.png", skeleton_image)

  #Delete states to allow line chaser to terminate
  for state in detected_states:
    state_center = (int((state["bndbox"][0] + state["bndbox"][2])/2), int((state["bndbox"][1] + state["bndbox"][3])/2))
    axes_lengths = (int((state["bndbox"][2] - state["bndbox"][0])/2) + delete_radius,int((state["bndbox"][3] - state["bndbox"][1])/2) + delete_radius)
    cv2.ellipse(skeleton_image, state_center, axes_lengths, 0, 0, 360, (0,0,0), -1)
    cv2.putText(skeleton_image, str(state["id"]), getBoxCenter(state["bndbox"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
  skeleton_image_original = skeleton_image.copy()

  #Save line chaser input
  cv2.imwrite(f"output/input_for_line_chaser.png", skeleton_image)
  
  #Create empty image to visualize line chaser pathing
  line_chaser_output_image = Image.new(mode = "RGB", size=(image.shape[1],image.shape[0]), color="#ffffff")

  #Create file to save transitions in
  if os.path.exists(f'output/transitions.txt'):
    with open(f'output/transitions.txt', 'w') as f:
      f.write("")

  #Find arrow connections by asssociating arrow heads to states (pointingAt) and then chasing their shaft with the line chaser (pointingFrom)
  for state in detected_states:
    #Find arrowheads that are close to states by defining a search box
    search_box = {
      "minX": state["bndbox"][0] - arrowhead_to_state_association_radius,
      "minY": state["bndbox"][1] - arrowhead_to_state_association_radius,
      "maxX": state["bndbox"][2] + arrowhead_to_state_association_radius,
      "maxY": state["bndbox"][3] + arrowhead_to_state_association_radius
    }

    for arrow in detected_arrows:
      #Check for each arrow if it overlaps with the search box and if so set the pointingAt and start the line chaser to find the pointingFrom
      if do_overlap((search_box["minX"], search_box["minY"]),(search_box["maxX"], search_box["maxY"]),
        (arrow["bndbox"][0], arrow["bndbox"][1]), (arrow["bndbox"][2],arrow["bndbox"][3])):
        arrow["pointingAt"] = state["id"]
        arrow["pointingFrom"] = findArrowConnection(skeleton_image,arrow,detected_states, line_chaser_output_image)

       
       
  #Save the line chaser output for visualization of pathing
  line_chaser_output_image.save(f"output/line_chaser_pathing.png")


  #Delete states and arrows from image (after line chaser deleted arrow shafts) to find labels
  for state in detected_states:
    skeleton_image[int(state["bndbox"][1]): int(state["bndbox"][3]),int(state["bndbox"][0]): int(state["bndbox"][2])] = 0
  for arrow in detected_arrows:
    skeleton_image[int(arrow["bndbox"][1]): int(arrow["bndbox"][3]),int(arrow["bndbox"][0]): int(arrow["bndbox"][2])] = 0

  Image.fromarray(skeleton_image).save(f"output/image_after_deletions_for_label_recog.png")
  
  detected_labels = findLabels(skeleton_image, detected_states)
  

  line_chaser_output_image = Image.new(mode = "RGB", size=(image.shape[1],image.shape[0]), color="#ffffff")
  for state in detected_states:
    #Find arrowheads that are close to states by defining a search box
    search_box = {
      "minX": state["bndbox"][0] - arrowhead_to_state_association_radius,
      "minY": state["bndbox"][1] - arrowhead_to_state_association_radius,
      "maxX": state["bndbox"][2] + arrowhead_to_state_association_radius,
      "maxY": state["bndbox"][3] + arrowhead_to_state_association_radius
    }

    for arrow in detected_arrows:
      #Check for each arrow if it overlaps with the search box and if so set the pointingAt and start the line chaser to find the pointingFrom
      if do_overlap((search_box["minX"], search_box["minY"]),(search_box["maxX"], search_box["maxY"]),
        (arrow["bndbox"][0], arrow["bndbox"][1]), (arrow["bndbox"][2],arrow["bndbox"][3])):
        findArrowConnection(skeleton_image_original,arrow,detected_states, line_chaser_output_image, detected_labels)

       
        #Write found transitions to output file
        with open(f'output/transitions.txt', 'a') as f:
          f.write(str(arrow))
          f.write('\n')
  assignLabels(detected_states,detected_arrows,detected_labels)
  
  

  #Print results to console
  print("Transitions:")
  for arrow in detected_arrows:
    print(arrow)
  print("States:")
  for state in detected_states:
    print(state)
  print("Labels:")
  for label in detected_labels:
    print(label)
    cv2.putText(skeleton_image_original, str(label["id"]), getBoxCenter(label["bndbox"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.rectangle(skeleton_image_original, (int(label["bndbox"][0]), int(label["bndbox"][1])),(int(label["bndbox"][2]), int(label["bndbox"][3])),(255, 0, 0))
  cv2.imwrite("output/labelIds.png", skeleton_image_original)
    
if __name__ == "__main__":
  input_file_path = sys.argv[1]
  main(input_file_path)
  
#for i in range(0,50):
#  main(f"../datasets/automata/images/{i}.png", i)