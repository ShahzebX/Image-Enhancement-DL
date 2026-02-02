import cv2 as cv
import numpy as np

# Load models
prototxt = "models/colorization_deploy_v2.prototxt"
model = "models/colorization_release_v2.caffemodel"
pts = "models/pts_in_hull.npy"

print("Loading models...")
net = cv.dnn.readNetFromCaffe(prototxt, model)
pts_data = np.load(pts)

# Set cluster centers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts_data.transpose().reshape(2, 313, 1, 1)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load grayscale image
img = cv.imread("b&w.jpg")
if img is None:
    print("Image not found")
    exit()

# Convert to LAB
scaled = img.astype("float32") / 255.0
lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)

L = lab[:, :, 0]
L_input = cv.dnn.blobFromImage(L)

# Predict ab channels
net.setInput(L_input)
ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize and combine
ab_dec_us = cv.resize(ab_dec, (img.shape[1], img.shape[0]))
lab_out = np.concatenate((L[:, :, np.newaxis], ab_dec_us), axis=2)

# Convert back to BGR
bgr_out = cv.cvtColor(lab_out, cv.COLOR_LAB2BGR)
bgr_out = np.clip(bgr_out * 255, 0, 255).astype("uint8")

cv.imwrite("result.jpg", bgr_out)
print("Done! Saved as result.jpg")
interface = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="numpy", label="Upload Black & White Image"),
    outputs=[
        gr.Image(type="numpy", label="Original Image"),
        gr.Image(type="numpy", label="Colorized Image"),
    ],
    title="Black & White → Color Image",
)

# THIS LAUNCHES THE UI
interface.launch()

