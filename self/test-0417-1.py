import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 1.颜色阈值+ 区域掩模
# Read in the image
img_path = "/data/ldp/zjf/code/Ultra-Fast-Lane-Detection/img/1492626274615008344/20.jpg"
image = mpimg.imread(img_path)

# Grab the x and y sizes and make two copies of the image
# With one copy we'll extract only the pixels that meet our selection,
# then we'll paint those pixels red in the original image to see our selection
# overlaid on the original.
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

# Define our color criteria
red_threshold = 220
green_threshold = 220
blue_threshold = 220
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define a triangle region of interest (Note: if you run this code,
P1 = [0, 360]
P2 = [320, 240]
P3 = [1040, 240]
P4 = [xsize - 1, 360]
fit_1 = np.polyfit((P1[0], P2[0]), (P1[1], P2[1]), 1)
fit_2 = np.polyfit((P3[0], P4[0]), (P3[1], P4[1]), 1)

left_bottom = [0, ysize - 1]
right_bottom = [xsize - 1, ysize - 1]
apex = [650, 400]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                   (image[:, :, 1] < rgb_threshold[1]) | \
                   (image[:, :, 2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX * fit_1[0] + fit_1[1])) & \
                    (YY > (XX * fit_2[0] + fit_2[1])) & \
                    (YY > 240) & \
                    (YY < ysize - 1)
# region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
#                     (YY > (XX * fit_right[0] + fit_right[1])) & \
#                     (YY < (XX * fit_bottom[0] + fit_bottom[1]))
# Mask color selection
color_select[color_thresholds] = [0, 0, 0]
# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display our two output images
# plt.imshow(color_select)
plt.imshow(line_image)

# uncomment if plot does not display
plt.show()
