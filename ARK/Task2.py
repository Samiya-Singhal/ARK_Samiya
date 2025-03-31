from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

pi_image = Image.open("pi_image.png").convert("L")
pi_array = np.array(pi_image)

corrected_values = pi_array/ (10 * math.pi)
sorted_values = np.sort(corrected_values.flatten())[::-1] 

filter_1 = sorted_values[:4]
print(filter_1)
filter_2x2= filter_1.reshape(2,2)
print("Generated Filter:\n", filter_2x2)

distorted_portrait = Image.open("artwork_picasso.png").convert("L")
portrait_array = np.array(distorted_portrait)

def apply_filter(image, filter_matrix, operation):
    filtered_image = image.copy().astype(np.uint8)
    rows, cols = image.shape
    filter_size = filter_matrix.shape[0]

    for i in range(0, rows - filter_size + 1, filter_size):
        for j in range(0, cols - filter_size + 1, filter_size):
            sub_matrix = image[i:i+filter_size, j:j+filter_size].astype(np.uint8)
            filter_matrix = filter_matrix.astype(np.uint8)

            if operation == "OR":
                filtered_image[i:i+filter_size, j:j+filter_size] = sub_matrix | filter_matrix
            elif operation == "AND":
                filtered_image[i:i+filter_size, j:j+filter_size] = sub_matrix & filter_matrix
            elif operation == "XOR":
                filtered_image[i:i+filter_size, j:j+filter_size] = sub_matrix ^ filter_matrix

    return filtered_image

restored_portrait = apply_filter(portrait_array, filter_2x2, "OR")

plt.imshow(restored_portrait, cmap="gray")
plt.title("Restored Portrait")
plt.show()

template = Image.fromarray(restored_portrait).resize((100, 100))

def template_match(image, template):
    img_h, img_w = image.shape
    temp_h, temp_w = template.shape
    min_diff = float("inf")
    best_x, best_y = 0, 0

    for x in range(img_h - temp_h):
        for y in range(img_w - temp_w):
            sub_image = image[x:x+temp_h, y:y+temp_w]
            diff = np.sum(np.abs(sub_image - template))

            if diff < min_diff:
                min_diff = diff
                best_x, best_y = x, y

    return best_x, best_y

collage = Image.open("collage.png").convert("L")
collage_array = np.array(collage)

x, y = template_match(collage_array, np.array(template))
print(f"Template found at: ({x}, {y})")

password = math.floor((x + y) * math.pi)
print("Password:", password)
