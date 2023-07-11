import streamlit as st
import cv2
import numpy as np
from PIL import*

st.title('Image processing App')

# Load the image
image_file = st.file_uploader('Choose an image', type=['jpg', 'png', 'jpeg'])
if image_file is not None:
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, width=400)
# Apply a processing technique

if st.checkbox('Colorspace'):
    COLORS  =  st.radio('type', ['RGB', 'HSV', 'LAB', 'LUV','Color threshold'])
    if COLORS == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif  COLORS == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    elif COLORS == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif COLORS == 'LUV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    elif COLORS =='Color threshold':
        # Split the image into its channels
        l, a, b = cv2.split(image)

        # Perform histogram equalization on the L channel
        l_equalized = cv2.equalizeHist(l)

        # Merge the equalized L channel back with the original A and B channels
        image_balanced = cv2.merge((l_equalized, a, b))

        # Convert the image back to the BGR colorspace
        image = cv2.cvtColor(image_balanced, cv2.COLOR_LAB2BGR)

    st.image(image, width=400)

if st.checkbox('Grayscale'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(image, width=400)

if st.checkbox("Dupli"):
    image=np.copy(image)
    st.image(image, width=400)

if st.checkbox('Channels'):
    channel = st.radio('type', ['Blue', 'Green', 'Red'])
    image = cv2.split(image)
    if channel == 'Red':
        image= image[0]

    elif channel == 'Green':
        image = image[1]

    elif channel == 'Blue':
        image= image[2]

    st.image(image, width=400)

# if st.checkbox("Histograms Equalization"):
#     equ = cv2.equalizeHist(iimage)
#     image = np.hstack((image, equ)
#     st.image(image, width=400)
if st.checkbox('CLAHE'):
    # Create a CLAHE object with a clip limit of 2.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the grayscale image
    image = clahe.apply(image)
    st.image(image, width=400)


if st.checkbox('Blur'):
    blurs =st.radio('type', ['blur', 'Gaussian', 'median'])
    ksize = st.slider('Kernel size', 3, 21, 5)
    if blurs=='blur':
         image = cv2.blur(image, (ksize, ksize))
    elif blurs =='Gaussian':
        image= cv2.GaussianBlur(image, (5, 5), 0)

    elif blurs == 'median':
        image = cv2.medianBlur(image, 5)
    st.image(image, width=400)
if st.checkbox('Filter'):
    filters = st.radio('type', ['Median', 'Gaussian', 'bilateral', 'Sobel', 'Salt and Pepper', 'Despeckle'])
    if filters == "Median":
        image = cv2.medianBlur(image, 5)
    elif filters == 'Gaussian':
        image = cv2.GaussianBlur(image, (5, 5), 0)
    elif filters == "bilateral":
        image = cv2.bilateralFilter(image, 15, 75, 75)

    elif filters =='Sobel':
        image = cv2.Sobel(image, cv2.CV_64F, 1, 0, 3)

    elif filters == 'Salt and Pepper':
        # Generate random pixel values
        rows, cols, channels = image.shape
        noise = np.random.randint(0, 2, (rows, cols, channels), dtype=np.uint8)

        # Set the pixels to either 0 (black) or 255 (white)
        noise[noise == 0] = 0
        noise[noise == 1] = 255

        # Add the noise to the image
        image = cv2.add(image, noise)
    if filters == 'Despeckle':
        # Define the kernel size
        kernel_size = 3

        # Create a kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)


    st.image(image, width=400)
if st.checkbox('Detection'):
    detections = st.radio('type', ['Coutour','Circle', 'Line'])

    if detections == 'Contour':
        image = cv2.Canny(image, 100, 200)

    if detections == 'Circle':
        image = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=0, maxRadius=0)

    if detections == 'Line':
        image = cv2.HoughLinesP(image, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    st.image(image, width=400)
if st.checkbox('CONTOUR'):
    image = cv2.Canny(image, 100, 200)

    st.image(image, width=400)
if st.checkbox('Transform'):
    transformations = st.radio('type', ['Rotate 90 Degrees Right','Rotate 90 Degrees Left','Flip Horizontally', 'Flip Vertically', 'Flip Z', 'transpose', 'translate', 'Zoom'])
    if transformations == "Rotate 90 Degrees Right":
        # Rotate the image 90 degrees to the right
        st.title("Replaces the image or selection with a x-mirror image of the original.")
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if transformations == 'Rotate 90 Degrees Left':
        st.title("Rotates the entire image or stack counter-clockwise 90◦")
        # Rotate the image 90 degrees to the left
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if transformations == 'Flip Horizontally':
        st.title("Replaces the image or selection with a x-mirror image of the original.")
        image = cv2.flip(image, 0)
    if transformations == 'Flip Vertically':
        st.title("Turns the image or selection upside down (y-mirror).")
        image = cv2.flip(image, 1)

    if transformations == 'Flip Z':
        st.title("Reverses the order of the slices in a stack (z-mirror).")
        image = cv2.flip(image, -1)
    if transformations == 'transpose':
        image = cv2.transpose(image)

    if transformations == 'translate':
        M = np.float32([[1, 0, 100], [0, 1, 50]])

        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    if transformations == 'Zoom':
        # Define the scaling factor
        scale_percent = 200  # The image will be 200% of the original size

        # Calculate the dimensions of the output image
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)

    st.image(image, width=400)

if st.checkbox('Plot Profile'):
    # Get the size of the image
    height, width = image.shape[:2]

    # Select the line position (x-coordinate)
    line_pos = width // 2
    # Draw the line on the image
    image = cv2.line(image, (line_pos, 0), (line_pos, height), (0, 0, 255), 1)
    st.bar_chart(image, width=600)

if st.checkbox('MOrpho'):
    kernel = np.ones((3, 3), np.uint8)
    morphos =st.radio('type', ['Dilate', 'Erode', 'Opening', 'Closing'])

    if morphos=='Dilate':
         st.title('Dilation is the opposite of erosion, and it increases the size of objects in the image. It can be used to fill in small gaps or holes in an image.')
         image = cv2.dilate(image, kernel)
    elif morphos =='Erode':
        st.title(' Erosion is a morphological transformation that erodes away the boundaries of objects in the image. It can be used to remove small details or noise from an image.')
        image= cv2.erode(image, kernel)

    elif morphos =='Opening':
        st.title(' Opening is a combination of erosion followed by dilation. It can be used to remove small details or noise from an image, while preserving the overall shape of the objects in the image')
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    elif morphos =='Closing':
        st.title(' Closing is the opposite of opening, and it is a combination of dilation followed by erosion. It can be used to fill in small gaps or holes in an image, while preserving the overall shape of the objects in the image.')
        image =  cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


    st.image(image, width=400)

#
# if st.checkbox("CROP"):
#     # Define the ROI dimensions
#     x, y, w, h = 50, 50, 200, 200
#     image = cv2.getRectSubPix(image, (w, h), (x + w / 2, y + h / 2))
#     st.image(image)

if st.checkbox('Histogramme'):
    image = cv2.calcHist([image], [0], None, [256], [0, 256])
    st.empty()

    st.title("Histogramm")

    st.bar_chart(image, width=600)

if st.checkbox('Enhance Contrast'):
    # Enhance the contrast of the image
    image=  cv2.equalizeHist(image)
    st.image(image, width=400)

if st.checkbox('LUT'):
    # This look-up table will invert the intensity values of the image
    lut = np.array([255 - i for i in range(256)], dtype=np.uint8)

    # Create the output image
    output = np.empty_like(image)

    # Apply the look-up table to the input image

    image = cv2.LUT(image, lut, output)


    st.image(image, width=400)

if st.checkbox('Threshold'):
    threshold_type = st.radio('Type', ['binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv','Otsu'])
    threshold = st.slider('Threshold', 0, 255, 128)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if threshold_type == 'binary':
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif threshold_type == 'binary_inv':
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    elif threshold_type == 'trunc':
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_TRUNC)
    elif threshold_type == 'tozero':
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)
    elif threshold_type == 'tozero_inv':
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO_INV)
    elif threshold_type == 'Otsu':
        _, image = cv2.threshold(gray, threshold, 255, cv2.THRESH_OTSU)
    st.image(image, width=400)
if st.checkbox('Color Balance'):
    # Split the image into its channels
    l, a, b = cv2.split(image)

    # Perform histogram equalization on the L channel
    l_equalized = cv2.equalizeHist(l)

    # Merge the equalized L channel back with the original A and B channels
    image_balanced = cv2.merge((l_equalized, a, b))

    # Convert the image back to the BGR colorspace
    image = cv2.cvtColor(image_balanced, cv2.COLOR_LAB2BGR)
    st.image(image, width=400)

if st.checkbox('Displays LUT'):
    # Create a LUT with 256 entries (for 8-bit images)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    # Fill the LUT with values according to some transformation
    for i in range(256):
        lut[i] = [i, 255 - i, i]

    # Apply the LUT to the image using cv2.LUT()
    result = cv2.LUT(image, lut)
    st.image(result, width=400)