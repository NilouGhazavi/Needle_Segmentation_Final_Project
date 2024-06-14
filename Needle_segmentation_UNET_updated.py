#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:12:44 2024

@author: niloughazavi
"""



# GPU Usage 
import tensorflow as tf
import os

# Available GPUs: 2080 Ti and 4070 Ti 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Visible GPUs:", gpus)

# If more than one GPU is available, choose one of them
if gpus:
    try:
        # Specify the GPU(s) to be used first GPU gpus[0], second GPU gpus[1]
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  
    except RuntimeError as e:       
        print(e)

        
def set_gpu_device(device_index):
    if not hasattr(set_gpu_device, "has_been_called"):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Set the visible GPU(s)
                tf.config.experimental.set_visible_devices(gpus[device_index], 'GPU')
            except RuntimeError as e:
                print(e)
        set_gpu_device.has_been_called = True

# Set GPU 0 as the active device 0 or 1 
set_gpu_device(0)

import tensorflow as tf
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)


# install opencv-python-headless if needed 
pip install opencv-python-headless


# import libraries 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt 
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.model_selection import KFold




#%% Load the data 
# Create data generators


# train_images 
train_image_dir="/Users/Nilou Ghazavi/Desktop/Nilou/PhD Courses/Spring2024/Midterm Report/trainImages"
# train_masks 
train_mask_dir="/Users/Nilou Ghazavi/Desktop/Nilou/PhD Courses/Spring2024/Midterm Report/trainMasks"
# test_images 
test_image_dir="/Users/Nilou Ghazavi/Desktop/Nilou/PhD Courses/Spring2024/Midterm Report/testImages"
# save predicted masks 
save_directory = "/Users/Nilou Ghazavi/Desktop/Nilou/PhD Courses/Spring2024/Midterm Report/predictedMasks"
# Augmented images 
augmented_image_path = "/Users/Nilou Ghazavi/Desktop/Nilou/PhD Courses/Spring2024/Midterm Report/augmented_image"
# Corresponding masks to augmented images 
augmented_mask_path="/Users/Nilou Ghazavi/Desktop/Nilou/PhD Courses/Spring2024/Midterm Report/augmented_mask"




# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(save_directory, exist_ok=True)
os.makedirs(augmented_image_path, exist_ok=True)
os.makedirs(augmented_mask_path, exist_ok=True)



# List and sort image and mask files
# images- train - sort the images 
train_image_files = sorted([os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith('.jpg')])
# masks- train 
train_mask_files = sorted([os.path.join(train_mask_dir, f) for f in os.listdir(train_mask_dir) if f.endswith('.png')])
# images- test 
test_images =sorted( [os.path.join(test_image_dir, x) for x in os.listdir(test_image_dir) if x.endswith('.jpg')])


# masks and train images 
data={
      'image':train_image_files,
      'mask':train_mask_files
      }

df=pd.DataFrame(data)


# Augmentation Technique : shift along height and width/ shear and rotation 
data_gen_args = {
    'rotation_range': 40,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}


# Apply transformation to an image based on given augmentation techniques 
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


# Process each image and mask
for index, row in df.iterrows():
    # preprocess each image and mask 
    image_path = row['image']
    mask_path = row['mask']
    image_id = os.path.basename(image_path).split('.')[0]  

    # Load the original images 
    img = load_img(image_path)  
    # Load the masks- training 
    mask = load_img(mask_path, color_mode='grayscale')  

    x_img = img_to_array(img)
    x_mask = img_to_array(mask)

    # Reshape the images to 1, W, H, channel= 1 (grayscale) 
    x_img = x_img.reshape((1,) + x_img.shape)
    x_mask = x_mask.reshape((1,) + x_mask.shape)

    # Define seeds (random number 0-10000)
    seed = np.random.randint(10000)
    # image data generation
    image_gen = image_datagen.flow(x_img, batch_size=1, seed=seed)
    mask_gen = mask_datagen.flow(x_mask, batch_size=1, seed=seed)

    # Generate and save augmented images and masks
    # 5 augmented version is generated 
    for i in range(5):  
        # augmented masks and images 
        aug_img = next(image_gen)[0].astype('uint8')
        aug_mask = next(mask_gen)[0].astype('uint8')

        # Save images with the new naming convention
        save_img(os.path.join(augmented_image_path, f'{image_id}_{i}.jpeg'), aug_img)
        save_img(os.path.join(augmented_mask_path, f'{image_id}_{i}_mask.png'), aug_mask)


# sort the augmented images masks 
# augmented images- train 
train_aug_image_files= sorted([os.path.join(augmented_image_path, f) for f in os.listdir(augmented_image_path) if f.endswith('.jpeg')])
# augmented masks- train 
train_aug_mask_files= sorted([os.path.join(augmented_mask_path, f) for f in os.listdir(augmented_mask_path) if f.endswith('.png')])





#%% Preprocessing images  

# (Normalize images )
def preprocess_image(image_dir, target_size):
    img = load_img(image_dir, color_mode='grayscale', target_size=target_size)
    img = img_to_array(img)
    img /= 255.0  
    return img

# (Normalize masks )
def preprocess_mask(mask_dir, target_size):
    mask = load_img(mask_dir, color_mode='grayscale', target_size=target_size)
    mask = img_to_array(mask)
    mask /= 255.0  
    mask = np.round(mask)  
    return mask



# use CTScanData to generate batches 
class CTScanData(Sequence):
    # image file name and mask file name 
    def __init__(self, image_filenames, mask_filenames, batch_size):
        self.image_filenames, self.mask_filenames = image_filenames, mask_filenames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return np.array([
            # image preprocessing 
            preprocess_image(file_name, input_shape[:2]) for file_name in batch_x]), np.array([
            preprocess_mask(file_name, input_shape[:2]) for file_name in batch_y])




# map from an original image to the corresponding mask 
mask_map = {os.path.splitext(os.path.basename(mask))[0].replace('_mask', ''): mask for mask in train_aug_mask_files}



# pair images with the corresponding masks for the train dataset
paired_images_masks = []
for img in train_aug_image_files:
    base_id = os.path.splitext(os.path.basename(img))[0]
    if base_id in mask_map:
        paired_images_masks.append((img, mask_map[base_id]))
    else:
        print(f"No matching mask for image ID {base_id}")



# Generate train images and the corresponding mask images (without augmentation)
train_img_files, train_mask_files = zip(*paired_images_masks)
train_gen = CTScanData(list(train_img_files), list(train_mask_files), batch_size=8)


# train- augmented images and masks (augmented images and masks)
train_aug_img_files, train_aug_mask_files = zip(*paired_images_masks)
train_aug_gen = CTScanData(list(train_aug_img_files), list(train_aug_mask_files), batch_size=8)



#%% UNET Model 

# UNet model with encoder and decoder blocks

# convolutional block 
def conv_block(input_tensor, num_filters):
   
    # first layer
    x = Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    # second layer
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    return x


# Encoder
def encoder_block(input_tensor, num_filters):

    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p


# Decoder
def decoder_block(input_tensor, concat_tensor, num_filters):

    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
    x = concatenate([x, concat_tensor], axis=-1)
    x = conv_block(x, num_filters)
    return x


# U-Net Model 
def build_unet(input_shape):
    
    inputs = Input(input_shape)

    # Encoder
    c1, p1 = encoder_block(inputs, 16)
    c2, p2 = encoder_block(p1, 32)
    c3, p3 = encoder_block(p2, 64)
    c4, p4 = encoder_block(p3, 128)

    # Bridge
    b = conv_block(p4, 256)

    # Decoder
    d1 = decoder_block(b, c4, 128)
    d2 = decoder_block(d1, c3, 64)
    d3 = decoder_block(d2, c2, 32)
    d4 = decoder_block(d3, c1, 16)

    # Output (sigmoid activation function)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model



# input shape is 512X512x1 gray scale CT images 
input_shape = (512, 512, 1)  
model = build_unet(input_shape)

# use Adam optimizer and binary cross entropy 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# checkpoint for early stopping 
checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(patience=10, verbose=1)


history = model.fit(
    # train with either original images or augmented images 
    #train_gen,
    train_aug_gen,
    epochs=50,
    callbacks=[checkpoint, early_stopping], 
    verbose=1
)




#%% Model Evaluation : Loss and Accuracy


# Plot accuracy vs epoch- training datase
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot loss vs epoch- training dataset
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()




### load the best model 
model.load_weights('best_model.h5') 


# Function to predict masks when inputing test images 
def predict_masks(model, image_paths, image_size=(512, 512), save_directory=None):
    # normalize images 
    images = np.array([preprocess_image(path, image_size) for path in image_paths])
    # predict masks for input images 
    predictions = model.predict(images) 
    # Binarize the prediction and change it back to 255 for visualization 
    predictions = (predictions > 0.5).astype(np.uint8) * 255  

    if save_directory:
        # if the directory doesn't exist 
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        for path, prediction in zip(image_paths, predictions):
            # Find the ID of each image from the file name 
            image_id = os.path.basename(path).split('.')[0]  
            # name of the files that should be saved imgae_ID
            mask_filename = f"{image_id}_mask.png"  
            img = Image.fromarray(prediction.squeeze(), 'L')  
            # the prediction should be saved in the same directory 
            img.save(os.path.join(save_directory, mask_filename))  

    return predictions


# Get the predicted masks (inpupt: Model, test image dataset, save directory)
predicted_masks = predict_masks(model, test_images, save_directory=save_directory)


# Visualize random predicted masks  ( 5 random images )
def plot_predictions(images, predictions, num=5):
    fig, axes = plt.subplots(num, 2, figsize=(10, 3 * num))
    # plot for 5 images 
    for i in range(num):
        ax = axes[i, 0]
        
        # original image 
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original Image")
        ax.axis('off')
        
        ax = axes[i, 1]
        # predicted image
        ax.imshow(predictions[i].squeeze(), cmap='gray')
        ax.set_title("Predicted Mask")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# compare the original image with the predicted mask randomly choose images from 20-30
original_images = [preprocess_image(path, (512, 512)) for path in test_images[20:30]]  
plot_predictions(original_images, predicted_masks, num=5)

#%% Generate the CSV file for Kaggle 


#Define the processImages function as provided (from Kaggle)
def processImages(imgDirectory: str, saveDirectory: str = os.getcwd(), returnDF:bool = False) -> pd.DataFrame | None:
    files = [f for f in os.listdir(imgDirectory) if f.endswith('.png')]
    if len(files) != 127:
        raise ValueError("Directory must contain exactly 127 .png files")

    files.sort(key=lambda x: int(x.split('_')[0]))

    data = []
    for file in files:
        imgPath = os.path.join(imgDirectory, file)
        img = np.array(Image.open(imgPath).convert('L'), dtype=np.uint8)

        if not np.array_equal(img, img.astype(bool).astype(img.dtype) * 255):
            raise ValueError(f"Image {file} is not binary")
        if img.shape != (512, 512):
            raise ValueError(f"Image {file} is not of size 512x512")

        status = 1 if np.any(img == 255) else 0
        maskIndices = ' '.join(map(str, np.nonzero(img.flatten() == 255)[0])) if status else '-100'

        data.append({'imageID': int(file.split('_')[0]), 'status': status, 'mask': maskIndices})

    df = pd.DataFrame(data).set_index('imageID')
    df.to_csv(os.path.join(saveDirectory, 'submission.csv'))

    if returnDF: return df


# save it as a csv file 
processImages(save_directory, save_directory)


#%% Evaluate Model Generalization : K-fold cross validation 

# number of folds 
n_folds = 5

# save the accuracy of each fold 
fold_accuracies=[]

# Kfold object (5 folds)
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)


fold_var = 1

# train and validation sets 
for train_index, val_index in kf.split(paired_images_masks):
    print(f"Training on fold {fold_var}...")
    
    # split the data into : train dataset and validation dataset 
    train_pairs = [paired_images_masks[i] for i in train_index]
    val_pairs = [paired_images_masks[i] for i in val_index]
    
    
    # images for training (image & mask pair)
    train_img_files, train_mask_files = zip(*train_pairs)
    # images for validation  (image & mask pair)
    val_img_files, val_mask_files = zip(*val_pairs)
    
    
    # Create data generators for training and validation ( batch size 8)
    train_gen = CTScanData(list(train_img_files), list(train_mask_files), batch_size=8)
    val_gen = CTScanData(list(val_img_files), list(val_mask_files), batch_size=8)
    
    # Rebuild the U-Net model with the new datasets (using the images generated for cross validation)
    model = build_unet(input_shape)

    # Compile the model
    # optimizer: Adaptive otimizer 
    # loss: Binary crossentropy 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model with train_gen and val_gen: run this for less number of epochs for each fold= 20 instead of 50 
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        # 20 epochs for each fold 
        epochs=20,
        callbacks=[
            ModelCheckpoint(f'best_model_fold_{fold_var}.h5', verbose=1, save_best_only=True, save_weights_only=True),
            EarlyStopping(patience=10, verbose=1)
        ],
        verbose=1
    )
    
    # evaluate the model and compare the accuracies 
    scores = model.evaluate(val_gen, verbose=0)
    fold_accuracies.append(scores[1]) 
    
    fold_var+=1


# Cross correlation with k=5 folds 
# accuracy 
print (f"The accuracy for each fold is {fold_accuracies}")
# the average accuracy
print(f"The mean of fold-accuracies is {np.mean(fold_accuracies)}")


