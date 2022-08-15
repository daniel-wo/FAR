import numpy as np
from PIL import Image, ImageDraw
import cv2
from enum import IntEnum

class Direction(IntEnum):
  NORTH = 0
  NORTH_EAST = 1
  EAST = 2
  SOUTH_EAST = 3
  SOUTH = 4
  SOUTH_WEST = 5
  WEST = 6
  NORTH_WEST = 7


# image = Image.new(mode = "L", size=(1024,768), color=0)
# draw = ImageDraw.Draw(image)
# draw.line(xy=[(0,0),(50,50),(70,20), (80,20),(100,30),(100,600),(0,517)], width=1, fill=255)
# image.save("test.png")
# image_matrix = np.array(image)



def chase_line(image, starting_position, initial_direction):

  image_matrix = np.array(image)
  height, width = image_matrix.shape

  steps = 0

  image_matrix = np.swapaxes(image,0,1)

  output_image = Image.new(mode = "RGB", size=(width,height), color="#ffffff")
  draw = ImageDraw.Draw(output_image)

  

  current_position = starting_position
  current_direction = initial_direction
  window_value_sum = 1

  while(window_value_sum > 0):

    (x,y) = current_position

    #print(f"current position:{(x,y)}, current direction:{current_direction}")

    all_jump_windows = []
    for i in range(0,3):
      for j in range(0,3):
        if(i != 1 or j != 1):
          x_window = x-3 + j*2
          y_window = y-3 + i*2
          all_jump_windows.append(image_matrix[x_window : x_window+3,y_window : y_window+3])


    all_jump_windows_ordered = []
    all_jump_windows_ordered.append(all_jump_windows[1])
    all_jump_windows_ordered.append(all_jump_windows[2])
    all_jump_windows_ordered.append(all_jump_windows[4])
    all_jump_windows_ordered.append(all_jump_windows[7])
    all_jump_windows_ordered.append(all_jump_windows[6])
    all_jump_windows_ordered.append(all_jump_windows[5])
    all_jump_windows_ordered.append(all_jump_windows[3])
    all_jump_windows_ordered.append(all_jump_windows[0])


    directional_jump_windows = []

    for i in range(0,5):
      directional_jump_windows.append(all_jump_windows_ordered[(current_direction -2 + i) % 8])

    directional_window_values = []
    

   


    directional_window_values.append(np.sum(directional_jump_windows[0])*0.5)
    directional_window_values.append(np.sum(directional_jump_windows[1])*0.75)
    directional_window_values.append(np.sum(directional_jump_windows[2]))
    directional_window_values.append(np.sum(directional_jump_windows[3])*0.75)
    directional_window_values.append(np.sum(directional_jump_windows[4])*0.5)

    #for i in range(0,5):
      #print(directional_jump_windows[i].transpose())

    window_value_sum = np.sum(directional_window_values)

    direction_adjust = np.argmax(directional_window_values) - 2

    current_direction = (current_direction + direction_adjust)%8

    
    if(current_direction == Direction.NORTH):
      current_position = (x,y-1)
      image_matrix[x-1: x+2 ,y: y+3] = 0
    if(current_direction == Direction.NORTH_EAST):
      current_position = (x+1,y-1)
      image_matrix[x-2: x+1 ,y: y+3] = 0
    if(current_direction == Direction.EAST):
      current_position = (x+1,y)
      image_matrix[x-2: x+1 ,y-1: y+2] = 0
    if(current_direction == Direction.SOUTH_EAST):
      current_position = (x+1,y+1)
      image_matrix[x-2: x+1 ,y-2: y+1] = 0
    if(current_direction == Direction.SOUTH):
      current_position = (x,y+1)
      image_matrix[x-1: x+2 ,y-2: y+1] = 0
    if(current_direction == Direction.SOUTH_WEST):
      current_position = (x-1,y+1)
      image_matrix[x: x+3 ,y-2: y+1] = 0
    if(current_direction == Direction.WEST):
      current_position = (x-1,y)
      image_matrix[x: x+3 ,y-1: y+2] = 0
    if(current_direction == Direction.NORTH_WEST):
      current_position = (x-1,y-1)
      image_matrix[x: x+3 ,y: y+3] = 0

    draw.point(current_position, "#ff0000")
    steps +=1
  output_image.save("output.png")
  return current_position, steps


