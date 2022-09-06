from inspect import trace
import numpy as np
from skimage.draw import line
from skimage.morphology import thin
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from dict2xml import dict2xml
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
        "filename": filename*2,
        "path": f"\home\\niemand\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection-/dataset/images/{filename}.jpg",
        "size": {"width": size[0], "height": size[1], "depth": 1},
        "segmented": 0,
        "object": []
    }

    for object in root.findall('symbols/traceGroup'):
        name = object.findall('annotation')[0].text
        if(name != "arrow"):
            bndbox = getBoundingBoxOfTraces(object, root)
            annotations["object"].append(
                {
                    "name": name,
                    "bndbox": bndbox
                }
            )
        else:
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
    if(exists(f'../additional_annotations/{filename}.xml')):
        add_tree = ET.parse(f'../additional_annotations/{filename}.xml')
        add_root = add_tree.getroot()
        for arrow in add_root.findall('object'):
            xml_bndbox = arrow.find('bndbox')
            annotations["object"].append(
                {
                    "name": arrow.find('name').text,
                    "bndbox": {
                        "xmin": int(xml_bndbox.find('xmin').text) -60,
                        "ymin": int(xml_bndbox.find('ymin').text) -60,
                        "xmax": int(xml_bndbox.find('xmax').text) -60,
                        "ymax": int(xml_bndbox.find('ymax').text) -60
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

    file = filename * 2

    for object in annotations["object"]:
      cv2.rectangle(image, (object["bndbox"]["xmin"],object["bndbox"]["ymin"]), (object["bndbox"]["xmax"],object["bndbox"]["ymax"]), (0,255,0), 2)

    xml = dict2xml(annotations)
    xmlfile = open(f"../datasets/automata/annotations/{file}.xml", "w+")
    xmlfile.write(xml)
    xmlfile.close()
    
    cv2.imwrite(f"../datasets/automata/images/{file}.png",image)

    # Augment image and save result as well
    augmented_image, augmented_annotations = createAugmentations(image, annotations, file)
    
    xml = dict2xml(augmented_annotations)
    xmlfile = open(f"../datasets/automata/annotations/{(file + 1)}.xml", "w+")
    xmlfile.write(xml)
    xmlfile.close()

    for object in augmented_annotations["object"]:
      cv2.rectangle(augmented_image, (object["bndbox"]["xmin"],object["bndbox"]["ymin"]), (object["bndbox"]["xmax"],object["bndbox"]["ymax"]), (0,255,0), 2)

    


    cv2.imwrite(f"../datasets/automata/images/{(file + 1)}.png",augmented_image)

    
#Creates augmentations of image
def createAugmentations(image, annotations, filename):

    transform1 = A.Compose([
        A.Flip(always_apply=True),
        A.RandomScale(always_apply=True, scale_limit=(-0.5,-0.1))
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    transform2 = A.Compose([
        A.Rotate(always_apply=True),
        A.RandomScale(always_apply=True, scale_limit=(-0.5,-0.1))
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

    transformed_annotations["filename"] = filename+1
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


for i in range(0,300):
    inkml2img(f"../FA_database_1.1/{i}.inkml", i)


