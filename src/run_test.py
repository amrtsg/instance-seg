import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

def data_loader(img_path):
    image = cv2.imread(img_path, 1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = Image.fromarray(image)
    return np.array(image)

print("Loading image...")
image_dataset = data_loader("test/test_img.jpg")
image_dataset = np.array(image_dataset)

model = load_model('../models/segmentation_model.hdf5')

# Randomly selecting an image from the test set and making prediction
test_img = image_dataset
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

# Plotting the real image, its mask, and its predicted mask
plt.figure(figsize=(16, 12))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()