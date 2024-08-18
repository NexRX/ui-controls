import cv2
import numpy as np
import time

# Load the source and template images
source_image_path = '__fixtures__/screen-2k.png'  # Replace with your source image path
template_image_path = '__fixtures__/btn.png'  # Replace with your template image path

# Read the images
source_img = cv2.imread(source_image_path)
template_img = cv2.imread(template_image_path)

# Convert images to grayscale
source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

# Start timing
start_time = time.time()

# Perform template matching using cv2.matchTemplate
result = cv2.matchTemplate(source_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Get the best match position
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# End timing
# End timing
end_time = time.time()
time_taken_ms = (end_time - start_time) * 1000

# Define the bounding box based on the template size
template_height, template_width = template_gray.shape
top_left = max_loc
bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

# Draw the bounding box on the source image
cv2.rectangle(source_img, top_left, bottom_right, (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Template', source_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result image with bounding box
output_image_path = 'matched_result.jpg'  # Path where the result will be saved
cv2.imwrite(output_image_path, source_img)

print(f'Template matching result saved to {output_image_path}')
print(f'Time taken for template matching: {time_taken_ms:.4f} seconds')
