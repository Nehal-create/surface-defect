import os
import numpy as np
# import pandas as pd
import streamlit as st
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.datasets import load_files
from keras.utils import np_utils
from termcolor import colored
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import array_to_img, img_to_array, load_img





# import libraries
import streamlit as st
import tkinter as tk
from tkinter import filedialog
st.snow()
# Set up tkinter
root = tk.Tk()
root.withdraw()



# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

# Folder picker button
st.title('Folder Picker')
st.write('Please select a folder:')
clicked = st.button('Folder Picker')
if clicked:
    dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))

    train_dir = './NEU Metal Surface Defects Data/train'
    val_dir = './NEU Metal Surface Defects Data/valid'
    test_dir = './NEU Metal Surface Defects Data/test'
    print("Path: ", os.listdir("./NEU Metal Surface Defects Data"))
    print("Train: ", os.listdir("./NEU Metal Surface Defects Data/train"))
    print("Test: ", os.listdir("./NEU Metal Surface Defects Data/test"))
    print("Validation: ", os.listdir("./NEU Metal Surface Defects Data/valid"))

    print("Inclusion Defect")
    print("Training Images:", len(os.listdir(train_dir + '/' + 'Inclusion')))
    print("Testing Images:", len(os.listdir(test_dir + '/' + 'Inclusion')))
    print("Validation Images:", len(os.listdir(val_dir + '/' + 'Inclusion')))

    # Rescaling all Images by 1./255
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Training images are put in batches of 10
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')

    # Validation images are put in batches of 10
    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(200, 200),
        batch_size=10,
        class_mode='categorical')


    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.98):
                print("\nReached 98% accuracy so cancelling training!")
                self.model.stop_training = True


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # tf.keras.utils.plot_model(
    #     model,
    #     to_file='cnn_architecture.png',
    #     show_shapes=True)

    callbacks = myCallback()
    history = model.fit(train_generator,
                        batch_size=32,
                        epochs=1,
                        validation_data=validation_generator,
                        callbacks=[callbacks],
                        verbose=1, shuffle=True)

    sns.set_style("whitegrid")
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    sns.set_style("whitegrid")
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    # Loading file names & their respective target labels into numpy array
    def load_dataset(path):
        data = load_files(path)
        files = np.array(data['filenames'])
        targets = np.array(data['target'])
        target_labels = np.array(data['target_names'])
        return files, targets, target_labels


    x_test, y_test, target_labels = load_dataset(dirname)
    no_of_classes = len(np.unique(y_test))
    no_of_classes

    y_test = np_utils.to_categorical(y_test, no_of_classes)


    def convert_image_to_array(files):
        images_as_array = []
        for file in files:
            # Convert to Numpy Array
            images_as_array.append(img_to_array(load_img(file)))
        return images_as_array


    x_test = np.array(convert_image_to_array(x_test))
    print('Test set shape : ', x_test.shape)

    x_test = x_test.astype('float32') / 255
    # Plotting Random Sample of test images, their predicted labels, and ground truth
    y_pred = model.predict(x_test)
    fig = plt.figure(figsize=(10, 10))
    
    # k=trget_labels[0]   
    # def color_text(true_idx, pred_idx,k):
    #     if true_idx == pred_idx:
    #         return '\033[32m' + k + '\033[0m'  # green color
    #     else:
    #         return '\033[31m' + k + '\033[0m' 
       
    for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

        st.image(np.squeeze(x_test[idx]))
        pred_idx = np.argmax(y_pred[idx])
        true_idx = np.argmax(y_test[idx])
        # st.write("predicted:\n", target_labels[pred_idx])
        # st.write("Ground Truth : \n",target_labels[true_idx])
        def header(url):
            st.markdown(f'<h6 style="color:#33ff33;">{url}</h6>', unsafe_allow_html=True)
        def header1(url):
            st.markdown(f'<h6 style="color:#Ff0000;">{url}</h6>', unsafe_allow_html=True)
            
   

        if(target_labels[pred_idx]==target_labels[true_idx]):
            st.write("Predicted:")
            header(target_labels[pred_idx])
            st.write("Ground_truth:\n")
            header(target_labels[true_idx])
            
        else:
        
            st.write("Predicted:")
            header1(target_labels[pred_idx])
            st.write("Ground_truth:\n")
            header(target_labels[true_idx])
        # st.write("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]))

 






