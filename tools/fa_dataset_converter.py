from __future__ import annotations
import csv
from email.mime import base
from inspect import trace
import random
import numpy as np
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from dict2xml import dict2xml
from PIL import Image, ImageChops
import cv2
from os.path import exists
import albumentations as A

#Returns traces as lists of coordinates for drawing purposes and calculates an image size for the drawing
def get_traces_data(inkml_file_abs_path):

    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
  
    root = tree.getroot()
    
    'Stores traces_all with their corresponding id'
    traces_all = [{'id': trace_tag.get('id'),
                   'coords': [[round(float(axis_coord))/10 if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                               for axis_coord in coord[1:].split(' ')[0:2]] if coord.startswith(' ')
                              else [round(float(axis_coord))/10 if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                                    for axis_coord in coord.split(' ')[0:2]]
                              for coord in (trace_tag.text).replace('\n', '').split(',')]}
                  for trace_tag in root.findall('trace')]

    image_width, image_height = findImageSize(traces_all)

    'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find('traceGroup')
    
    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall('traceGroup'):

            label = traceGroup.find('annotation').text

            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall('traceView'):

                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

                'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)

            traces_data.append({'label': label, 'trace_group': traces_curr})

    else:
        'Consider Validation data that has no labels'
        [traces_data.append({'trace_group': [trace['coords']]})
         for trace in traces_all]

    return traces_data, image_width, image_height


#Gets the bounding box of a set of traces
def getBoundingBoxOfTraces(object, root):
    traceIds = object.findall('traceView')
    traces = []
    for traceId in traceIds:
        for trace in root.findall('trace'):
            if trace.attrib['id'] == traceId.attrib['traceDataRef']:
                traces.append(trace)

    xCoords = []
    yCoords = []
    for trace in traces:
        for coord in (" " + trace.text).replace('\n', '').split(','):
            xCoords.append(int(coord.split(' ')[1])/10)
            yCoords.append(int(coord.split(' ')[2])/10)
            
    
    
    return {
        "xmin": int(np.min(xCoords)),
        "ymin": int(np.min(yCoords)),
        "xmax": int(np.max(xCoords)),
        "ymax": int(np.max(yCoords)),
    }


#Extracts annotations from inkml files
def extractAnnotations(inkml_file_abs_path, size, filename):
    
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()

    annotations =   {
        "folder": "datasets/automata",
        "filename": filename,
        "path": f"\home\\niemand\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection-/dataset/images/{filename}.png",
        "size": {"width": size[0], "height": size[1], "depth": 1},
        "segmented": 0,
        "object": []
    }

    for object in root.findall('symbols/traceGroup'):
        name = object.findall('annotation')[0].text
        if(name != "arrow" and name != "label"):
            bndbox = getBoundingBoxOfTraces(object, root)
            annotations["object"].append(
                {
                    "name": name,
                    "bndbox": bndbox
                }
            )
        elif(name != "label"):
            bndbox = getBoundingBoxOfTraces(object.find('head'),root)
            isFalselyAnnotated = False
            for traceView in object.findall('head/traceView'):
                if 'to' in traceView.attrib:
                    isFalselyAnnotated = True
            if(len(object.findall('traceView')) < 2 or isFalselyAnnotated):
                continue
            annotations["object"].append(
                {
                    "name": name,
                    "bndbox": bndbox
                }
            )
        elif(name == "label"):
            bndbox = getBoundingBoxOfTraces(object, root)
            meaning =  object.findall('annotation')[1].text
            annotations["object"].append(
                {
                    "name": name,
                    "bndbox": bndbox
                }
            )

    if(exists(f'../additional_annotations/{filename}.xml')):
        add_tree = ET.parse(f'../additional_annotations/{filename}.xml')
        add_root = add_tree.getroot()
        for arrow in add_root.findall('object'):
            xml_bndbox = arrow.find('bndbox')
            annotations["object"].append(
                {
                    "name": arrow.find('name').text,
                    "bndbox": {
                        "xmin": int(xml_bndbox.find('xmin').text) -50,
                        "ymin": int(xml_bndbox.find('ymin').text) -50,
                        "xmax": int(xml_bndbox.find('xmax').text) -50,
                        "ymax": int(xml_bndbox.find('ymax').text) -50
                    }
                }
            )

    return annotations

#Finds max coordinates from a set of traces
def findImageSize(all_traces):
    maxY = 0
    maxX = 0
    for trace in all_traces:
        for coord in trace['coords']:
            if coord[0] > maxX:
                maxX = coord[0]
            if coord[1] > maxY:
                maxY = coord[1]
    return int(maxX), int(maxY)


#Converts inkml files to a png file with a corresponding annotation xml file in pascal voc format
def inkml2img(input_path, filename):
    
    traces, width, height = get_traces_data(input_path)
    image = np.zeros((int(height),int(width),1))

    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i,j,0] = 255

    for elem in traces:
        ls = elem['trace_group']
        for subls in ls:
            data = np.array(subls)
            x, y = zip(*data)
            for i in range(len(x)-1):
                cv2.line(image,(int(x[i]),int(y[i])), (int(x[i+1]),int(y[i+1])), color=0, thickness=2)

    annotations = extractAnnotations(input_path, (width,height), filename)

    #for object in annotations["object"]:
     #   cv2.rectangle(image, (object["bndbox"]["xmin"],object["bndbox"]["ymin"]), (object["bndbox"]["xmax"],object["bndbox"]["ymax"]), (0,255,0), 2)

    xml = dict2xml(annotations)
    xmlfile = open(f"../datasets/automata/annotations/{filename}.xml", "w+")
    xmlfile.write(xml)
    xmlfile.close()
    
    cv2.imwrite(f"../datasets/automata/images/{filename}.png",image)

    # Augment image and save result as well

    for i in range(1,4):
        augmented_image, augmented_annotations = createAugmentations(image, extractAnnotations(input_path, (width,height), filename), filename+300*i, i)
        
        xml = dict2xml(augmented_annotations)
        xmlfile = open(f"../datasets/automata/annotations/{(filename+300*i)}.xml", "w+")
        xmlfile.write(xml)
        xmlfile.close()

        #for object in augmented_annotations["object"]:
         #   cv2.rectangle(augmented_image, (object["bndbox"]["xmin"],object["bndbox"]["ymin"]), (object["bndbox"]["xmax"],object["bndbox"]["ymax"]), (0,255,0), 2)

        cv2.imwrite(f"../datasets/automata/images/{(filename + 300*i)}.png",augmented_image)

def squarify(M,val):
    (a,b,c)=M.shape
    
    if a>b:
        padding=((0,0),(0,a-b),(0,0))
    else:
        padding=((0,b-a),(0,0),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

def trim(img):
    mask = img!=255
    mask = mask.any(2)
    mask0,mask1 = mask.any(0),mask.any(1)
    colstart, colend = mask0.argmax(), len(mask0)-mask0[::-1].argmax()+1
    rowstart, rowend = mask1.argmax(), len(mask1)-mask1[::-1].argmax()+1
    return img[rowstart:rowend, colstart:colend]

#Creates augmentations of image
def createAugmentations(image, annotations, filename, iteration):

    padding=((0,20),(0,20),(0,0))
    image = np.pad(image,padding,mode='constant',constant_values=255)

    if(iteration == 1):
        transform1 = A.Compose([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomScale(always_apply=True, scale_limit=(-0.5,-0.1)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    if(iteration == 2):
        transform1 = A.Compose([
            A.VerticalFlip(p=1),
            A.RandomScale(always_apply=True, scale_limit=(-0.5,-0.1)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    if(iteration == 3):
        transform1 = A.Compose([
            A.HorizontalFlip(p=1),
            A.RandomScale(always_apply=True, scale_limit=(-0.5,-0.1)),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    bndbox_labels = []
    bndboxes = []

    for annotation in annotations["object"]:
        bndbox_labels.append(annotation["name"])
        bndboxes.append(
            [
                annotation["bndbox"]["xmin"],
                annotation["bndbox"]["ymin"],
                annotation["bndbox"]["xmax"],
                annotation["bndbox"]["ymax"]
            ]
        )
    transformed = transform1(image = image, bboxes = bndboxes, class_labels = bndbox_labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    transformed_annotations = annotations.copy()

    transformed_annotations["filename"] = filename
    transformed_annotations["path"] = f"\home\\niemand\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection-/dataset/images/{filename}.png",

    transformed_annotations["size"] = {"width": transformed_image.shape[0], "height": transformed_image.shape[1], "depth": 1}
    i = 0
    for bndbox in transformed_bboxes:
        transformed_annotations["object"][i]["bndbox"] = {
            "xmin": int(bndbox[0]),
            "ymin": int(bndbox[1]),
            "xmax": int(bndbox[2]),
            "ymax": int(bndbox[3])
        }
        i += 1
    
    return transformed_image, transformed_annotations

def createTextDetectionDataset(number_of_samples):
    for i in range(0,number_of_samples):

        # Get random automaton picture and corresponding annotations
        base_image_number = random.randint(0,299)
        base_image = Image.open(f"../datasets/automata/images/{base_image_number}.png")
        np_base_image = np.pad(np.array(base_image), ((50,50),(50,50)),mode="constant", constant_values="255")
        base_annotations = extractAnnotations(f"../FA_database_1.1/{base_image_number}.inkml", (np_base_image.shape[1]+100,np_base_image.shape[0]+100), base_image_number)

        with open('../datasets/characters/english.csv', mode='r') as infile:
            reader = csv.reader(infile)
            char_dict = {rows[0]:rows[1] for rows in reader}
        # Replace old labels with random letters from the character dataset

        indices_to_delete = []
        for j in range(0,len(base_annotations["object"])):
            if base_annotations["object"][j]["name"]  != "label":
                indices_to_delete.append(j)
        
        indices_to_delete.reverse()
        for j in indices_to_delete:
            base_annotations["object"].pop(j)

        for object in base_annotations["object"]:
            if object["name"] == "label":
                old_label_bndbox = {
                    "xmin": object["bndbox"]["xmin"] + 50,
                    "ymin": object["bndbox"]["ymin"] + 50,
                    "xmax": object["bndbox"]["xmax"] + 50,
                    "ymax": object["bndbox"]["ymax"] + 50,
                }
                
                # Delete old label by coloring it white
                np_base_image[old_label_bndbox["ymin"]:old_label_bndbox["ymax"], old_label_bndbox["xmin"]:old_label_bndbox["xmax"]] = 255
                np_base_image[old_label_bndbox["ymin"]-3:old_label_bndbox["ymax"]+3, old_label_bndbox["xmin"]-3:old_label_bndbox["xmax"]+3] = 255
                # Insert new letter and correct bndbox
                random_char = random.randint(1,62)
                random_char_variant = random.randint(1,55)
                char_image_name = f"Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png"
                char_image_size = trim(np.array(Image.open(f"../datasets/characters/{char_image_name}"))).shape

                letter_image = Image.fromarray(trim(np.array(Image.open(f"../datasets/characters/{char_image_name}")))).resize((
                    int(char_image_size[1] * 0.12), int(char_image_size[0] * 0.12)
                ))
                automaton_image = Image.fromarray(np_base_image)
                automaton_image.paste(letter_image,
                    (int(old_label_bndbox["xmin"] + (old_label_bndbox["xmax"] - old_label_bndbox["xmin"])/2 - char_image_size[1]*0.05), 
                    int(old_label_bndbox["ymin"] + (old_label_bndbox["ymax"] - old_label_bndbox["ymin"])/2 - char_image_size[0]*0.05))
                )

                object["bndbox"] = {
                    "xmin": int(old_label_bndbox["xmin"] + (old_label_bndbox["xmax"] - old_label_bndbox["xmin"])/2 - char_image_size[1]*0.05), 
                    "ymin": int(old_label_bndbox["ymin"] + (old_label_bndbox["ymax"] - old_label_bndbox["ymin"])/2 - char_image_size[0]*0.05),
                    "xmax": int(old_label_bndbox["xmin"] + (old_label_bndbox["xmax"] - old_label_bndbox["xmin"])/2 - char_image_size[1]*0.05 + char_image_size[1]*0.12),
                    "ymax": int(old_label_bndbox["ymin"] + (old_label_bndbox["ymax"] - old_label_bndbox["ymin"])/2 - char_image_size[0]*0.05 + char_image_size[0]*0.12) 
                }
                object["name"] = char_dict[char_image_name]
                np_base_image = np.array(automaton_image)

        xml = dict2xml(base_annotations)
        xmlfile = open(f"../datasets/text_detection_automata/annotations/{i}.xml", "w+")
        xmlfile.write(xml)
        xmlfile.close()
        automaton_image = np.array(automaton_image)
        for object in base_annotations["object"]:
            cv2.rectangle(automaton_image, (object["bndbox"]["xmin"],object["bndbox"]["ymin"]), (object["bndbox"]["xmax"],object["bndbox"]["ymax"]), (0,255,0), 2)

        cv2.imwrite(f"../datasets/text_detection_automata/images/{i}.png", automaton_image)
for i in range(0,300):
    inkml2img(f"../FA_database_1.1/{i}.inkml", i)

createTextDetectionDataset(500)


