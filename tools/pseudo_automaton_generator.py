from PIL import Image, ImageDraw
import cv2
import random
from dict2xml import dict2xml
import csv

def makeWhiteTransparent(img):
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 or item[1] == 255 or item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return img

#Create dictionary for char lookup
with open('datasets/handwritten_characters/english.csv', mode='r') as infile:
    reader = csv.reader(infile)
    char_dict = {rows[0]:rows[1] for rows in reader}

annotations = []
number_of_samples = 1

#Create number_of_samples training images
for instance in range(number_of_samples):
        
    #Choose initial parameters
    automaton = Image.new(mode = "L", size=(1024,768), color=255)
    automaton.convert("RGBA")

    state_positions = []
    state_number = random.randint(2,8)

    #Scale states depending on how many there are
    scale_factor = int((9-state_number)/5 + 1)

    state_size = 100*scale_factor
    state_padding = 100*scale_factor

    #Append annotation metadata
    annotations.append(
        {
            "folder": "datasets/pseudo_automata",
            "filename": instance,
            "path": f"\home\\niemand\Keras-retinanet-Training-on-custom-datasets-for-Object-Detection-/{instance}.jpg",
            "source": {"database": "Unknown"},
            "size": {"width": 1024, "height": 768, "depth": 1},
            "segmented": 0,
            "object": []
        }   
    )

    #Choose some state positions
    for i in range(state_number):

        threshold = 0
        seeking = True
        while(seeking):
            seeking = False
            x = random.randint(0,1024-state_size)
            y = random.randint(0,768-state_size)
            threshold += 1
            for state in state_positions:
                if(not  (x > state[0]+state_size+state_padding) | 
                        (x+state_size+state_padding < state[0]) | 
                        (y > state[1]+state_size+state_padding) | 
                        (y+state_size+state_padding < state[1])):
                    seeking = True
                    break
            if(threshold > 100): break

        state_positions.append((x,y))
        
    #Draw some transitions + chars between the states
    for i in range(len(state_positions)-1):

        #Draw transitions sometimes
        if(random.randint(1,10) > 5):
            draw = ImageDraw.Draw(automaton)
            draw.line(
                xy=[(state_positions[i][0]+state_size/2,state_positions[i][1]+state_size/2),state_positions[i+1][0]+state_size/2,state_positions[i+1][1]+state_size/2], 
                width=scale_factor*3
            )

        
        #Choose a char and scale it down
        random_char = random.randint(1,62)
        random_char_variant = random.randint(1,55)
        char_image_name = f"Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png"
        char_image_path = f"datasets/handwritten_characters/{char_image_name}"

        char_scale_factor = 0.6

        #Calculate the position of the chars at the transitions
        initial_label_position = (int((state_positions[i][0] + state_positions[i+1][0])/2),int((state_positions[i][1] + state_positions[i+1][1])/2))
        (char_x,char_y) = initial_label_position

        #Insert the char
        automaton.paste(
            im = makeWhiteTransparent(
                Image.open(char_image_path).resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor)))),
            box = (char_x,char_y),
            mask = makeWhiteTransparent(
                Image.open(char_image_path).resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor))))
        )
        #Append annotation
        annotations[instance]["object"].append(
            {   
                "name": 'char',
                "bndbox": {"xmin": char_x, "ymin": char_y, "xmax": char_x+state_size*char_scale_factor, "ymax": char_y+state_size*char_scale_factor}
            }
        )

        #Add multiple chars with commas TODO
        '''
        for j in range(2,random.randint(2,3)):

            random_char = random.randint(1,62)
            random_char_variant = random.randint(1,55)
            char_scale_factor = 0.6

            automaton.paste(
                im = makeWhiteTransparent(
                    Image.open(f"datasets/handwritten_characters/Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png")
                        .resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor)))),
                box = (int(initial_label_position[0] + j*state_size*char_scale_factor/4), initial_label_position[1]),
                mask = makeWhiteTransparent(
                    Image.open(f"datasets/handwritten_characters/Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png")
                        .resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor))))
            )

            annotations[instance]["object"].append({"name": "char", "bndbox": {"xmin": x, "ymin": y, "xmax": x+state_size*char_scale_factor, "ymax": y+state_size*char_scale_factor}})
        '''

        
    #Place circles and chars inside them
    for (x,y) in state_positions:
        #Choose a circle
        random_circle = random.randint(1,919)
        #Insert it
        automaton.paste(Image.open(f"datasets/circles/{random_circle}.png").resize((state_size,state_size)), (x,y))
        #Append annotation
        annotations[instance]["object"].append({"name": "state", "bndbox": {"xmin": x, "ymin": y, "xmax": x+state_size, "ymax": y+state_size}})


        #Add chars as labels inside the states sometimes
        if(random.randint(1,10)>4):
            #Choose a char and scale it
            random_char = random.randint(1,62)
            random_char_variant = random.randint(1,55)
            char_image_name = f"Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png"
            char_image_path = f"datasets/handwritten_characters/{char_image_name}"
            char_scale_factor = 0.8

            #Calculate the label position
            label_position = (int(x + state_size * (1-char_scale_factor)/2),int(y + state_size * (1-char_scale_factor)/2))
            (char_x,char_y) = label_position

            #Insert char
            automaton.paste(
                im = makeWhiteTransparent(
                    Image.open(char_image_path).resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor)))),
                box = (char_x,char_y),
                mask = makeWhiteTransparent(
                    Image.open(char_image_path).resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor))))
            )
            #Annotate it
            annotations[instance]["object"].append(
                {
                    "name": 'char',
                    "bndbox": {"xmin": char_x, "ymin": char_y, "xmax": char_x+state_size*char_scale_factor, "ymax": char_y+state_size*char_scale_factor}
                }
            )

    #Save image file
    automaton.convert("L")
    automaton.save(f"datasets/pseudo_automata/images/{instance}.jpg")

    #Save annotation file
    xml = dict2xml(annotations[instance])
    xmlfile = open(f"datasets/pseudo_automata/annotations/{instance}.xml", "w+")
    xmlfile.write(xml)
    xmlfile.close()
