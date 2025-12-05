import cv2
import numpy as np
import gradio as gr

# =====================================
# LOAD PRETRAINED MODEL
# =====================================

proto = "colorization_deploy_v2.prototxt"
model = "colorization_release_v2.caffemodel"
pts = "pts_in_hull.npy"
 
net = cv2.dnn.readNetFromCaffe(proto, model)

cluster_centers = np.load(pts)
cluster_centers = cluster_centers.transpose().reshape(2, 313, 1, 1)

net.getLayer(net.getLayerId("class8_ab")).blobs = [cluster_centers.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

print("Model Loaded Successfully!")

# =====================================
# COLORIZATION FUNCTION
# =====================================

def colorize_image(input_image):
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0  # scale 0-1

    h, w = img.shape[:2]

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]

    # Model expects L channel resized to 224x224
    L_resized = cv2.resize(L, (224, 224))
    L_resized = L_resized - 50  # subtract mean (standard for this model)

    # Create blob
    net_input = cv2.dnn.blobFromImage(L_resized)
    net.setInput(net_input)

    # Predict ab channels
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (w, h))  # resize to original image size

    # Merge L and ab channels
    lab_out = np.zeros((h, w, 3), dtype="float32")
    lab_out[:, :, 0] = L
    lab_out[:, :, 1:] = ab

    # Convert back to RGB
    colorized = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (colorized * 255).astype("uint8")

    return img, colorized
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

