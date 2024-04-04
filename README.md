## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
<br>
Import numpy module as np and pandas as pd.

### Step2:
<br>
Assign the values to variables in the program.

### Step3:
<br>
Get the values from the user appropriately.

### Step4:
<br>
Continue the program by implementing the codes of required topics.

### Step5:
<br>
Thus the program is executed in google colab.

## Program:
```python
Developed By: SASIRAJKUMAR T J
Register Number:212222230137
```
i)Image Translation
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread("cat.jpg")

# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Disable x & y axis
plt.axis('off')

# Show the image
plt.imshow(input_image)
plt.show()

# Get the image shape
rows, cols, dim = input_image.shape

# Transformation matrix for translation
M = np.float32([[1, 0, 100],
                [0, 1, 200],
                [0, 0, 1]])  # Fixed the missing '0' and added correct dimensions

# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))

# Disable x & y axis
plt.axis('off')

# Show the resulting image
plt.imshow(translated_image)
plt.show()
```

ii) Image Scaling
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis

# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)
```


iii)Image shearing
```import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '3nat.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)
```

iv)Image Reflection
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '4 nature.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)

# Display original and reflected images
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)
```

v)Image Rotation
```
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nat5.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)


```



vi)Image Cropping
```

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = '6nat.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)




```
## Output:
### i)Image Translation
![Screenshot 2024-04-04 081417](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/50ee5a71-e15e-43a7-9053-54413b0addd0)
![Screenshot 2024-04-04 081426](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/b6e99d33-c891-49c8-82d4-1668cea8afec)







### ii) Image Scaling
![Screenshot 2024-04-04 081820](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/9324e822-6747-47b7-8dc4-8edab585aad0)
![Screenshot 2024-04-04 081832](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/3f23949d-0046-43a6-95f2-30101a410c7a)






### iii)Image shearing
![Screenshot 2024-04-04 082129](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/3779677c-d489-4031-927b-0e30fe005b98)
![Screenshot 2024-04-04 082135](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/efb32e97-2566-48bd-9374-bfd09ba36d8b)





### iv)Image Reflection
![Screenshot 2024-04-04 082522](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/486df938-6d74-4858-b2a3-2d6a0d67687a)
![Screenshot 2024-04-04 082527](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/9749bef5-99fa-48f4-ad7e-594c6821bcd9)
![Screenshot 2024-04-04 082534](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/7d747341-8383-4606-87e9-1e05e4009de2)
![Screenshot 2024-04-04 082540](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/8f1fd772-7816-4ae6-b1f4-ae46c3ea1587)





### v)Image Rotation
![Screenshot 2024-04-04 082509](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/92fdcfc8-568b-40ac-b07a-9b926f8ca994)
![Screenshot 2024-04-04 082515](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/abe198fa-79f6-4747-83b9-73025f063f31)



### vi)Image Cropping
![Screenshot 2024-04-04 082907](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/2982b6b8-e592-4abd-b875-2e4602bb99e0)
![Screenshot 2024-04-04 082913](https://github.com/SASIRAJ27/IMAGE-TRANSFORMATIONS/assets/113497176/66fe9e4d-75ca-44ab-8efd-079982fcea42)




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
