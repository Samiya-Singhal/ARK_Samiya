import numpy as np    
from PIL import Image
import matplotlib.pyplot as plt

left = Image.open("left.png").convert("L")
leftarr= np.array(left)
right = Image.open("right.png").convert("L")
rightarr= np.array(right)

h, w = leftarr.shape
depth_map = np.zeros((h, w), dtype=np.float32)

block_size = 5
max_shift = 64
margin = block_size // 2

for y in range(margin, h - margin):
    for x in range(margin + max_shift, w - margin):
        best_shift = 0
        min_diff = float('inf')
        for d in range(max_shift):
            if x - margin - d < 0:  
                continue
            left_block = leftarr[y-margin:y+margin+1, x-margin:x+margin+1]
            right_block = rightarr[y-margin:y+margin+1, x-margin-d:x+margin+1-d]
            diff = np.sum((left_block - right_block) ** 2)
            if diff < min_diff:
                min_diff = diff
                best_shift = d
        depth_map[y, x] = best_shift


depth_normalized = (depth_map / 64) * 255

def simple_colormap(gray_img):
    height, width = gray_img.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            val = gray_img[y, x]
            r, g, b = 0, 0, 0

            if val < 85:         
                r = 0
                g = (val * 3)
                b = 255 
            elif ((val < 170) and (val>=85)) :     
                r = ((val - 85) * 3)
                g = 255
                b = 255-r
            else:                 
                r = 255
                g = 255 - ((val - 170) * 3)
                b = 0

            color_img[y, x] = [b, g, r]  
    return color_img
depth_colored= simple_colormap(depth_normalized)
plt.imshow(depth_colored)
plt.title("depth_colored")
plt.show()



