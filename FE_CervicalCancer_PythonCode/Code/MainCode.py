# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:45:37 2023

@author: FE
"""


# Importing Necessary Libraries
import cv2
import os
import shutil
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage import io
from collections import Counter
import imageio
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc,classification_report
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from PIL import Image


# Importing Keras for Image Classification
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPool2D, Dropout,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from keras import models,optimizers
from keras.applications import DenseNet201

import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, Callback


# # Google Drive'ın Colab ile bağlantısı
# from google.colab import drive
# drive.mount('/content/drive/')


import warnings
warnings.filterwarnings("ignore")


# Source Location for Filtered Dataset  LOCAL
input_folder = ".\CervicalCancer"

# # Source Location for Filtered Dataset   COLAB
# input_folder = "/content/drive/MyDrive/CervicalCancerCode/"

#drive_path = '/content/drive/MyDrive/CervicalCancerCode/input/cervical-cancer-largest-dataset-sipakmed/im_Dyskeratotic/im_Dyskeratotic/CROPPED/'


# Source Location for Dataset   LOCAL
src = '../input/cervical-cancer-largest-dataset-sipakmed';

# # Source Location for Dataset   COLAB
# src = '/content/drive/MyDrive/CervicalCancerCode/input/cervical-cancer-largest-dataset-sipakmed';



# Destination Location for Dataset   LOCAL
dest = './CervicalCancer';

# # Destination Location for Dataset  COLAB
# dest = '/content/drive/MyDrive/CervicalCancerCode/CervicalCancer/';


# Image Classes
classes = ["Dyskeratotic","Koilocytotic","Metaplastic","Parabasal","Superficial-Intermediate"];


# Dataset Root Folder  LOCAL
root_dir = "./CervicalCancer"

# # Dataset Root Folder  COLAB
# root_dir = "/content/drive/MyDrive/CervicalCancerCode/CervicalCancer"

#ImagePreProcessTypes = ["Blur","Bilateral","BlurBilateral","MedianBlur","GaussianBlur"];
ImagePreProcessTypes = ["Blur","Bilateral","BlurBilateral","MedianBlur","GaussianBlur"];

input_shape = (77,77,3)
batch_sizes = [64]
#optimizer = ['Adam', 'RMSPROP','SGD', 'Adam', 'SGD']
optimizers = ['Adam','RMSPROP']
#epochs = [20, 30, 20, 30]
epochs = [50]
learning_rates = [0.001,0.0001]

liste = ['PreProcess','SaveName','Accuracy']



def image_segmentation(original_img,ImagePreProcessType):
    if ImagePreProcessType == "Blur":
        # Blur Filtering
        J = cv2.blur(original_img, (2, 2));
    elif ImagePreProcessType == "Bilateral":
        # Bilateral Filtering
        J=cv2.bilateralFilter(original_img,9,75,75)
    elif ImagePreProcessType == "BlurBilateral":
        # BlurBilateral Filtering
        J = cv2.blur(original_img, (2, 2));
        J=cv2.bilateralFilter(original_img,9,75,75)
    elif ImagePreProcessType == "MedianBlur":
        # MedianBlur Filtering
        J=cv2.medianBlur(original_img, 5)
    elif ImagePreProcessType == "GaussianBlur":
        # GaussianBlur Filtering
        J=cv2.GaussianBlur(original_img, (3,3), 7)
    return J;


def FormatDataset(dataset_src, dataset_dest, classes,ImagePreProcessType):
    # Making a Copy of Dataset
    new_cropped_dest = [os.path.join(dataset_dest, cls) for cls in classes];
    cropped_src = [ dataset_src + "/im_" + cls + "/im_" + cls + "/CROPPED" for cls in classes ];

    for dest1 in new_cropped_dest:
        os.makedirs(dest1);
    #Formating Cropped Images
    for (src,new_dest) in zip(cropped_src, new_cropped_dest):
        for file in os.listdir(src):
            filename, file_ext = os.path.splitext(file);
            #if file_ext == '.bmp':
            if file_ext.endswith(".bmp"):
                img_des = os.path.join(new_dest, filename + '.jpg');
                #print(os.path.join(src, file))
                img = cv2.imread(os.path.join(src, file));
                img = cv2.resize(img, (75, 75));
                img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0);
                img = cv2.blur(img, (2, 2));
                img = image_segmentation(img,ImagePreProcessType);
                cv2.imwrite(img_des,img)
                
                
for i in range(len(ImagePreProcessTypes)):
    ImagePreProcessType = ImagePreProcessTypes[i]
    
    # Destination Location for Dataset  LOCAL
    dest = './'+ImagePreProcessTypes[i]+'/CervicalCancer';
    
    # # Destination Location for Dataset COLAB
    # dest = '/content/drive/MyDrive/CervicalCancerCode/'+ImagePreProcessTypes[i]+'/CervicalCancer';

    # Formatting Dataset
    FormatDataset(src, dest, classes,ImagePreProcessTypes[i]);
    
   

def load_images_from_folder(folder):
    images = []
    labels = []

    #class_names = sorted(os.listdir(folder))

    for class_name in classes:
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            label = class_name  # Sınıf adını etiket olarak kullan
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if img_path.endswith(".jpg"):
                    image = io.imread(img_path)
                    images.append(image)
                    labels.append(label)
    return np.array(images), np.array(labels)

def split_data(x, y, test_size=0.15, val_size=0.15):
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(test_size + val_size), random_state=42, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_temp, y_temp, test_size=(val_size / (test_size + val_size)), random_state=42, stratify=y_temp)
    return x_train, y_train, x_test, y_test, x_val, y_val


def save_images_with_names(images, labels, folder_path):
    for i, (image, label) in enumerate(zip(images, labels)):
        label_folder = os.path.join(folder_path, str(label))
        os.makedirs(label_folder, exist_ok=True)

        # Resmi sıralı numara ile kaydet
        image_name = f"{label}_{i}.jpg"
        image_path = os.path.join(label_folder, image_name)

        # Görüntüyü kaydet
        imageio.imwrite(image_path, image)


# Ploting Accuracy In Training Set & Validation Set

def show_history(ModelHistory,ImagePreProcessType,SaveName):
    # Plot model performance
    acc = ModelHistory.history['accuracy']
    val_acc = ModelHistory.history['val_accuracy']
    loss = ModelHistory.history['loss']
    val_loss = ModelHistory.history['val_loss']
    epochs_range = range(1, len(ModelHistory.epoch) + 1)

    #plt.subplot(1, 2, 1)
    #the figure has 1 row, 2 columns, and this plot is the first plot.
    #plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Set')
    plt.plot(epochs_range, val_acc, label='Val Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(ImagePreProcessType+'-'+SaveName+' Model Accuracy')
    plt.tight_layout()
    plt.show()

    #plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Set')
    plt.plot(epochs_range, val_loss, label='Val Set')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(ImagePreProcessType+'-'+SaveName+' Model Loss')
    plt.tight_layout()
    plt.show()
    


def Prediction(model,ImagePreProcessType,SaveName):
    # predictions: Modelin tahmin sonuçları
    predictions = model.predict(x_test)

    # Tahmin edilen sınıfları al
    predicted_classes = np.argmax(predictions, axis=1)

    # Finding precision and recall
    accuracy = accuracy_score(y_test_encoded, predicted_classes)
    print("Accuracy ("+ImagePreProcessType+","+SaveName+")  :", accuracy)

    liste.append([ImagePreProcessType,SaveName,str(accuracy*100)+" %"])

    precision = precision_score(y_test_encoded, predicted_classes, average='macro')
    print("Precision ("+ImagePreProcessType+","+SaveName+") :", precision)

    recall = recall_score(y_test_encoded, predicted_classes, average='macro')  # micro, macro, weighted gibi değerleri seçebilirsiniz
    print("Recall ("+ImagePreProcessType+","+SaveName+")   :", recall)

    F1_score = f1_score(y_test_encoded, predicted_classes, average='macro')
    print("F1-score ("+ImagePreProcessType+","+SaveName+") :", F1_score)

    # Confusion Matrix Çiz
    # compute the confusion matrix
    cm = confusion_matrix(y_test_encoded,predicted_classes)

    #Plot the confusion matrix.
    sns.heatmap(cm,
                annot=True,
                fmt='g')
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title(ImagePreProcessType+'-'+SaveName+' Confusion Matrix',fontsize=17)
    plt.show()


    # evaluate the model
    print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, predicted_classes))
    print("\nClassification Report:\n", classification_report(y_test_encoded, predicted_classes))

    # Her bir sınıf için ROC eğrisini çizmek için hazırlıklar
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Her bir sınıf için ROC eğrisini çiz
    for j in range(len(classes)):
        fpr[j], tpr[j], _ = roc_curve(y_test_bin[:, j], predictions[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])
        plt.plot(fpr[j], tpr[j], label=f'Class {classes[j]} (AUC = {roc_auc[j]:.2f})')

    # Rasgele tahminin ROC eğrisini çiz
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    # Grafik ayarları
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(ImagePreProcessType+'-'+SaveName+' Multiclass ROC Curves (One-vs-Rest)')
    plt.legend()
    plt.show()
    

class CustomHistoryCallback(Callback):
    def __init__(self):
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))

def RunModelHistoryPrediction(ModelName,Model):
  for k in range(len(optimizers)):
    for m in range(len(learning_rates)):
      for n in range(len(batch_sizes)):
        for x in range(len(epochs)):
          print('Runing Optimizer: ' + optimizers[k]+ ', Learning Rate: ' + str(learning_rates[m])+ ', Batch Size: ' + str(batch_sizes[n]) + ', Epoch: ' + str(epochs[x]))

          if optimizers[k]=='Adam':
            optimizer = tf.optimizers.Adam(learning_rate=learning_rates[i], beta_1=0.9, beta_2=0.999, epsilon=1e-7, weight_decay=0.0, amsgrad=False)
            print(optimizer)
          elif optimizers[k] == 'RMSPROP':
            optimizer = tf.optimizers.RMSprop(learning_rate=learning_rates[i], rho=0.9, epsilon=1e-7,weight_decay=0.0)
            print(optimizer)

          Model.compile(
              loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy']
          )
          Model.summary()
          SaveName = ModelName+'Model'+optimizers[k]+'Optimizer'+str(m)+'LR'+str(batch_sizes[n])+'BachSize'+str(epochs[x])+'Epochs'
          SaveNamePath = SaveName+'_best_model.hdf5'

          # Adding Model check point Callback  LOCAL
          mc = ModelCheckpoint(filepath=ImagePreProcessType+"/"+SaveNamePath,
                                  monitor= 'val_accuracy', verbose= 1,
                                  save_best_only= True, mode = 'auto');
          
          # # Adding Model check point Callback  COLAB
          # mc = ModelCheckpoint(filepath=input_folder+ImagePreProcessType+"/"+SaveNamePath,
          #                         monitor= 'val_accuracy', verbose= 1,
          #                         save_best_only= True, mode = 'auto');
          
          #call_back = [ mc ];
          
          
          # CustomHistoryCallback for All History
          custom_history_callback = CustomHistoryCallback()

          # Fitting the Model
          History = Model.fit(
              x_train,y_train_bin,
              steps_per_epoch = len(x_train)//batch_sizes[n],
              epochs = epochs[x],
              validation_data = (x_val,y_val_bin),
              validation_steps = len(x_val)//batch_sizes[n],
              callbacks=[mc, custom_history_callback]
              )
          
         
          # Eğitim sürecinde elde edilen history bilgilerini Pickle formatında kaydet   LOCAL
          with open(ImagePreProcessType+"/"+SaveName+'complete_model_history.pkl', 'wb') as file:
              pickle.dump(custom_history_callback.history, file)
        
          # # Eğitim sürecinde elde edilen history bilgilerini Pickle formatında kaydet  COLAB
          # with open(input_folder+ImagePreProcessType+"/"+SaveName+'complete_model_history.pkl', 'wb') as file:
          #     pickle.dump(custom_history_callback.history, file)
              
             

          # Loading the Best Fit Model  LOCAL
          modelBest = load_model(ImagePreProcessType+"/"+SaveNamePath)
          
          # # Loading the Best Fit Model   COLAB
          # modelBest = load_model(input_folder+ImagePreProcessType+"/"+SaveNamePath)


          show_history(History,ImagePreProcessType,SaveName)
          Prediction(modelBest,ImagePreProcessType,SaveName)


def RunVGG19Model(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin):

    # VGG19 Model
    model_VGG = VGG19(weights='imagenet',
                  input_shape = input_shape, # Shape of our images
                  include_top = False  # Leave out the last fully connected layer
                  )

    # Close the layers of vgg19
    for layer in model_VGG.layers:
        layer.trainable = False


    # Last layer
    modelVGG19 = Sequential()
    modelVGG19.add(model_VGG)
    modelVGG19.add(Flatten())
    modelVGG19.add(Dense(len(classes),activation='softmax'))

    RunModelHistoryPrediction('VGG19',modelVGG19)
    

def RunInceptionV3Model(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin):

    #InceptionV3 Model

    model_InceptionV3 = InceptionV3(input_shape = input_shape,
                             include_top = False,
                             weights = 'imagenet')

    for layer in model_InceptionV3.layers:
           layer.trainable = False

    x = layers.Flatten()(model_InceptionV3.output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = layers.Dense(5, activation='sigmoid')(x)

    modelInceptionV3 = tf.keras.models.Model(model_InceptionV3.input, x)

    RunModelHistoryPrediction('InceptionV3',modelInceptionV3)
    
    
def RunResnet50Model(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin):

    # Resnet50  Model
    model_Resnet50 = ResNet50(input_shape=input_shape,
                          include_top=False,
                          weights="imagenet")

    for layer in model_Resnet50.layers:
           layer.trainable = False


    modelResnet50 = Sequential()
    modelResnet50.add(model_Resnet50)
    modelResnet50.add(layers.Dropout(0.3))
    modelResnet50.add(layers.Flatten())
    modelResnet50.add(layers.Dropout(0.5))
    modelResnet50.add(layers.Dense(5, activation='sigmoid'))

    RunModelHistoryPrediction('Resnet50',modelResnet50)
    

def RunAlexnetModel(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin):

    # Alexnet Model
    #lamb = 0.9
    lamb = 0.
    modelAlexnet = Sequential([
        # 1st layer
        #keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(75,75,3)),
        Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu', input_shape=input_shape, kernel_regularizer=l2(lamb), name='conv1'),
        BatchNormalization(),
        #keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'),
        # 2nd layer
        Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu', kernel_regularizer=l2(lamb), name='conv2'),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'),
        # 3rd layer
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=l2(lamb), name='conv3'),
        BatchNormalization(),
        # 4th layer
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=l2(lamb), name='conv4'),
        BatchNormalization(),
        # 5th layer
        Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=l2(lamb), name='conv5'),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2), padding='valid'),
        # Flatten
        Flatten(),
        # 6th layer
        Dense(4096, activation='relu'),Dropout(0.5),
        # 7th layer
        Dense(4096, activation='relu'),Dropout(0.5),
        # 8th layer (output)
        Dense(units=5, activation='sigmoid')
        #keras.layers.Dense(5, activation='softmax')
    ], name='AlexNet')

    RunModelHistoryPrediction('Alexnet',modelAlexnet)
    
def RunDenseNet201Model(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin):
    conv_base = DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape)
    conv_base.summary()

    modelDenseNet = models.Sequential()
    modelDenseNet.add(conv_base)
    modelDenseNet.add(layers.GlobalAveragePooling2D())
    modelDenseNet.add(layers.Dense(1024, activation='relu'))
    modelDenseNet.add(layers.BatchNormalization())
    modelDenseNet.add(layers.Dropout(0.3))
    modelDenseNet.add(layers.Dense(5, activation='sigmoid'))
    conv_base.trainable = True

    RunModelHistoryPrediction('DenseNet201',modelDenseNet)


for i in range(len(ImagePreProcessTypes)):
    ImagePreProcessType = ImagePreProcessTypes[i]
    
    # Destination Location for Dataset   LOCAL
    dest = ImagePreProcessTypes[i]+'/CervicalCancer';
    
    # # Destination Location for Dataset  COLAB
    # dest = input_folder+ImagePreProcessTypes[i]+'/CervicalCancer';
    
    # Load Data
    x, y = load_images_from_folder(dest)

    # Shuffle data
    x, y = shuffle(x,y)

    # Split Data
    x_train, y_train, x_test, y_test, x_val, y_val = split_data(x, y, test_size=0.15, val_size=0.15)

    # Save Train,Val,Test in Data Folders    LOCAL
    save_images_with_names(x_train, y_train,ImagePreProcessTypes[i]+"/train")
    save_images_with_names(x_val, y_val,ImagePreProcessTypes[i]+"/val")
    save_images_with_names(x_test, y_test,ImagePreProcessTypes[i]+"/test")
    
     
    # # Save Train,Val,Test in Data Folders   COLAB
    # save_images_with_names(x_train, y_train,input_folder+ImagePreProcessTypes[i]+"/train")
    # save_images_with_names(x_val, y_val,input_folder+ImagePreProcessTypes[i]+"/val")
    # save_images_with_names(x_test, y_test,input_folder+ImagePreProcessTypes[i]+"/test")

     # Create LabelEncoder
    label_encoder = LabelEncoder()
    #Convert class labels to numeric values
    classes_encoded = label_encoder.fit_transform(classes)

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_val_encoded = label_encoder.transform(y_val)

    # Etiketleri binary formata dönüştür (One-vs-Rest için)
    y_train_bin = label_binarize(y_train, classes=np.unique(y))
    y_val_bin = label_binarize(y_val, classes=np.unique(y))
    y_test_bin = label_binarize(y_test, classes=np.unique(y))


    print('Total images ('+ImagePreProcessType+'  Filtered)   : ', len(y))
     # Metinlerin kaç kez geçtiğini bulma
    counter = Counter(y)
    # Sonuçları yazdırma
    for item, count in counter.items():
        print(f"     {item}: {count}")
    print('Training  ('+ImagePreProcessType+'  Filtered) : ', len(y_train))
    counter = Counter(y_train)
    # Sonuçları yazdırma
    for item, count in counter.items():
        print(f"     {item}: {count}")
    print('Validation  ('+ImagePreProcessType+'  Filtered) : ', len(y_val))
    counter = Counter(y_val)
    # Sonuçları yazdırma
    for item, count in counter.items():
        print(f"     {item}: {count}")
    print('Testing ('+ImagePreProcessType+'  Filtered) : ', len(y_test))
    counter = Counter(y_test)
    # Sonuçları yazdırma
    for item, count in counter.items():
        print(f"     {item}: {count}")

    RunVGG19Model(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin)
    RunInceptionV3Model(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin)
    RunResnet50Model(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin)
    RunAlexnetModel(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin)
    RunDenseNet201Model(ImagePreProcessType,x_train,y_train_bin,x_val,y_val_bin)


print(liste)