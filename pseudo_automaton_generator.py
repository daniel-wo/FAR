from PIL import Image, ImageDraw
import cv2
import random

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

#Choose initial parameters
automaton = Image.new(mode = "L", size=(1024,768), color=255)
automaton.convert("RGBA")

state_positions = []
state_number = random.randint(2,8)

scale_factor = int((9-state_number)/5 + 1)

state_size = 100*scale_factor
state_padding = 100*scale_factor

transition_label_positions = []


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

    
        
    if(random.randint(1,10) > 5):
        draw = ImageDraw.Draw(automaton)
        draw.line(
            xy=[(state_positions[i][0]+state_size/2,state_positions[i][1]+state_size/2),state_positions[i+1][0]+state_size/2,state_positions[i+1][1]+state_size/2], 
            width=scale_factor*3
        )

    initial_label_position = (int((state_positions[i][0] + state_positions[i+1][0])/2),int((state_positions[i][1] + state_positions[i+1][1])/2))

    random_char = random.randint(1,62)
    random_char_variant = random.randint(1,55)
    char_scale_factor = 0.6

    automaton.paste(
        im = makeWhiteTransparent(
            Image.open(f"datasets/handwritten_characters/Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png")
                .resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor)))),
        box = (int(initial_label_position[0]), initial_label_position[1]),
        mask = makeWhiteTransparent(
            Image.open(f"datasets/handwritten_characters/Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png")
                .resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor))))
    )
    
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

    
#Place circles and chars
for (x,y) in state_positions:
    random_circle = random.randint(1,919)
    automaton.paste(Image.open(f"datasets/circles/{random_circle}.png").resize((state_size,state_size)), (x,y))


    if(random.randint(1,10)>4):
        random_char = random.randint(1,62)
        random_char_variant = random.randint(1,55)
        char_scale_factor = 0.8
        automaton.paste(
            im = makeWhiteTransparent(
                Image.open(f"datasets/handwritten_characters/Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png")
                    .resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor)))),
            box = (int(x + state_size * (1-char_scale_factor)/2),int(y + state_size * (1-char_scale_factor)/2)),
            mask = makeWhiteTransparent(
                Image.open(f"datasets/handwritten_characters/Img/img{str(random_char).zfill(3)}-{str(random_char_variant).zfill(3)}.png")
                    .resize((int(state_size*char_scale_factor),int(state_size*char_scale_factor))))
        )


automaton.save("automaton.png")
