# --- IMPORTING ALL LIBRARIES AND VGG16 MODEL ---

import os
import keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
import random
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.decomposition import PCA
import pandas as pd

model = keras.applications.VGG16(weights='imagenet', include_top=True)
# model.summary()  --- if you want some details about model architecture

# CONNECTING TO THE GOOGLE DRIVE

from google.colab import drive
drive.mount('/content/MyDrive', force_remount=True)
os.chdir('/content/MyDrive/MyDrive')

# UNZIPPING COLLECTION OF SEM IMAGES

!unzip sem_images.zip

# READING SEM IMAGES FROM THE FOLDER

file_path = os.listdir('contour')
print(len(file_path))

labels = []
for im in file_path:
    impath = os.path.abspath(im)
    c_prev = os.path.split(impath)[-1]
    c = c_prev[:-4]
    labels.append(c)

import re

regex = re.compile(r'(\d+)_?([a-z]+)?')
new_labels = []

for label in labels:
  find = regex.search(label)
  new_labels.append(find.group(1))

len(new_labels)

os.chdir('/content/MyDrive/MyDrive')

df = pd.read_excel('shapes.xlsx')

df.shape

shapes = []
for el in new_labels:
  for i in range(len(df)):
    if el == str(df.loc[i, 'number']):
      shapes.append(df.loc[i, 'shape'])

shapes

df_shape = pd.DataFrame(shapes, columns=['shape'])

df_shape

# (ADDITIONAL) IMAGE GENERATION BY FLIPPING

os.chdir("/content/MyDrive/MyDrive/sem_images")
for image in file_path:
    path = os.path.abspath(image)
    c_prev = os.path.split(path)[-1]
    c = c_prev[:-4]
    im = cv2.imread(image)
    im_ud = cv2.flip(im, 0)
    im_lr = cv2.flip(im, 1)
    im_ur = cv2.flip(im, -1)
    img_ud = Image.fromarray(im_ud)
    img_lr = Image.fromarray(im_lr)
    img_ur = Image.fromarray(im_ur)
    img_ud.save(str(c) + "_ud.tif")
    img_lr.save(str(c) + "_lr.tif")
    img_ur.save(str(c) + "_ur.tif")

# DEFINE THE FUNCTION FOR READING IMAGES FROM THE FOLDER

def load_image(path):
    img = load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# (ADDITIONAL) SELECT AND SHOW ONE OF THE IMAGES

img, x = load_image("/content/MyDrive/MyDrive/sem_images/210.tif")
print("shape of x: ", x.shape)
print("data type: ", x.dtype)
plt.imshow(img)

# DEFINE FEATURE EXTRACTOR

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
feat_extractor.summary()

# (ADDITIONAL) CHECK FEATURE SPACE OF A SINGLE IMAGE

feat = feat_extractor.predict(x)

plt.figure(figsize=(16,4))
plt.plot(feat[0])

# SELECT IMAGES FROM THE FOLDER TO USE

images_path = "/content/MyDrive/MyDrive/contour"
image_extensions = ['.tif', '.png', '.jpg']   # case-insensitive (upper/lower doesn't matter)
max_num_images = 10000

images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]
if max_num_images < len(images):
    images = [images[i] for i in sorted(random.sample(xrange(len(images)), max_num_images))]

# IMAGE FEATURES EXTRACTION

import time
tic = time.clock()

features = []
for i, image_path in enumerate(images):
    if i % 500 == 0:
        toc = time.clock()
        elap = toc-tic;
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(images),elap))
        tic = time.clock()
    img, x = load_image(image_path);
    feat = feat_extractor.predict(x)[0]
    features.append(feat)

print('finished extracting features for %d images' % len(images))

# FEATURE SPACE COMPRESSION USING PCA

features = np.array(features)
pca = PCA(n_components=216)
pca.fit(features)
pca_features = pca.transform(features)

np.cumsum(pca.explained_variance_ratio_)

df = pd.DataFrame(data=pca_features, columns=['pc1', 'pc2'])

df.plot.scatter(x='pc1', y='pc2', title= "Scatter plot");
plt.show(block=True);

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
targets = ['cube', 'sphere', 'spike']
colors = ['#EBACA2', '#BED3C3', '#4A919E']
for target, color in zip(targets,colors):
    indicesToKeep = df_shape['shape'] == target
    ax.scatter(df.loc[indicesToKeep, 'pc1']
               , df.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 35)
ax.legend(targets)
ax.grid(False)
plt.savefig('PCA.eps', format='eps')

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

fig_k = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
plt.scatter(df['pc1'], df['pc2'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200)
plt.savefig('KMeans.eps', format='eps')

from scipy.cluster.vq import vq

closest, distances = vq(centroids, df)

closest

labels[236]

# GET QUERY IMAGE

# query_image_idx = int(len(images) * random.random())

n = 215
idx = images[n]

img = image.load_img(idx)
plt.imshow(img)

# IMAGE SIMILARITY OF A QUERY IMAGE TO IMAGES IN THE DATABASE

similar_idx = [ distance.cosine(pca_features[n], feat) for feat in pca_features ]

similar_idx

# SELECT THE MOST 5 SIMILAR IMAGES

idx_closest = sorted(range(len(similar_idx)), key=lambda k: similar_idx[k])[1:6]

for feat in pca_features:
  print(distance.cosine(pca_features[n], feat))

idx_closest

# OPEN 5 CLOSEST IMAGES IN THE DATASET

thumbs = []
for idx in idx_closest:
    img = image.load_img(images[idx])
    img = img.resize((int(img.width * 224 / img.height), 224))
    thumbs.append(img)

# CONCATENATE IMAGES INTO A SINGLE IMAGE

concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)

# SHOW TOP 5 CLOSEST IMAGES IN THE DATASET

plt.figure(figsize = (16,12))
plt.imshow(concat_image)

import os
import cv2
from PIL import Image

os.chdir('/content/MyDrive/MyDrive/sem_images')
im = cv2.imread('spikes.tif')
im = cv2.resize(im, (224,224))
im = Image.fromarray(im, 'RGB')
im.save('spikes.tif')
