# Make all the imports
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pickle

tf.disable_v2_behavior() 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# utility functions

def process_jpg_image(img):
  img = tf.convert_to_tensor(img[:,:,:3])
  img = np.expand_dims(img, axis = 0)
  img = tf.image.resize(img,[224,224])
  img = (img/255.0)
  return img

def show_confusion_matrix(cm, labels):
    '''
    plots heatmap of confusion matrix'''
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
              annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

# Load all the images. training folder is split 20% as validation set and testing folder is loaded as test dataset.

train_dir = r"D:\Machine_learning\projects\Velozity_systems\fake_currency_detection\dataset"
preprocess_input = tf.keras.applications.vgg16.preprocess_input
TARGET_SIZE = 224
BATCH_SIZE = 8


train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    subset="training",
                                                    shuffle = True,
                                                    target_size=(TARGET_SIZE,TARGET_SIZE))

validation_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    subset="validation",
                                                    shuffle = False,
                                                    target_size=(TARGET_SIZE,TARGET_SIZE))
#test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
#test_generator = test_datagen.flow_from_directory(test_dir,
#                                                  batch_size=BATCH_SIZE,
#                                                  class_mode='categorical',
#                                                  shuffle=False,
#                                                  target_size=(TARGET_SIZE,TARGET_SIZE))



# Print all the classes

print(train_generator.class_indices)

for image_batch, labels_batch in train_generator:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# Using a VGG model for training

from tensorflow.keras.applications.vgg16 import VGG16


base_model = VGG16(weights='imagenet', input_shape=(TARGET_SIZE, TARGET_SIZE, 3), include_top=False)
base_model.trainable = False

# Adding a model on top

inputs = tf.keras.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))

x = base_model.output
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
output = tf.keras.layers.Dense(2, activation='softmax')(x)

vgg = tf.keras.Model(inputs=base_model.input, outputs=output)


vgg.summary()

opt = tf.keras.optimizers.Adam()
cce = tf.keras.losses.BinaryCrossentropy()
vgg.compile(optimizer=opt, loss=cce, metrics= ['acc'])

checkpoint_filepath = '/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)


EPOCHS = 50
NUM_STEPS = train_generator.samples/BATCH_SIZE
VAL_NUM_STEPS = validation_generator.samples/BATCH_SIZE
model = vgg.fit(train_generator, epochs = EPOCHS, steps_per_epoch = NUM_STEPS, validation_steps = VAL_NUM_STEPS, validation_data = validation_generator, callbacks=[reduce_lr])


# Save the model to disk

#vgg.save("vgg_model_2.h5")
pickle.dump(vgg, open('/vgg-model.pkl','wb'))

vgg.save('vgg.h5')

# Plot the training and validation accuracy and loss graphs

acc = model.history['acc']
val_acc = model.history['val_acc']

loss = model.history['loss']
val_loss = model.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Testing the model on test set

class_names = list(train_generator.class_indices.items())
print(class_names)

test_image_path = "/data/dataset/fake/10 rupees Images_ Stock Photos....jpg"

test_image_read_1 = cv2.imread(test_image_path)
test_image_1 = process_jpg_image(test_image_read_1)
prediction_1 = vgg.predict(test_image_1)
print(f'dimensions of image used for prediction is: ',test_image_1.shape)

prediction = int(np.argmax(prediction_1))
print(f"prediction is: ", class_names[prediction][0])

## Predicting all the images in the test dataset and plotting a confusion matrix

# Get the labels of all the images
true_labels = validation_generator.labels
# Making the predictions of all the validation images
all_predictions = vgg.predict(validation_generator)

preds = []
for items in all_predictions:
    preds.append(np.argmax(items))
    

# plot the confusion matrix

confusion_mat = tf.math.confusion_matrix(
    true_labels, preds, dtype=tf.dtypes.int32)

show_confusion_matrix(confusion_mat, class_names)

f1 = f1_score(true_labels, preds, average='weighted')
print(f"F1 score of the model is", f1)