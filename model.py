import os
import random
import numpy as np
from shutil import move
from matplotlib import pyplot
from matplotlib.image import imread
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.models import Model

dir = 'H:\\Code\\current_files\\My_Code\\Dogs_v_Cats'
folder = dir + '\\Dataset\\train_data\\'


#Viewing first nine dogs in the dataset

# for i in range(9):
#     pyplot.subplot(330+1+i)
#     pyplot.axis('off')
#     filename = folder + 'dog.' + str(i) + '.jpg'
#     image = imread(filename)
#     pyplot.imshow(image)
# pyplot.show()


#Viewing first nine cats in the dataset

# for i in range(9):
#     pyplot.subplot(330+1+i)
#     pyplot.axis('off')
#     filename = folder + 'cat.' + str(i) + '.jpg'
#     image = imread(filename)
#     pyplot.imshow(image)
# pyplot.show()


#Pre-processing the images - Method 1

# photos, labels = list(), list()
# for file in os.listdir(folder):
#     if file.startswith('cat'):
#         output = 1.0
#     else:
#         output = 0.0
#     photo = load_img(folder + file, target_size=(200,200))
#     photo = img_to_array(photo)
#     photos.append(photo)
#     labels.append(output)
# photos = np.asarray(photos)
# labels = np.asarray(labels)
# print(photos.shape, labels.shape)
# np.save('/media/sam189239/Backup Plus/dogs_v_cats_photos.npy', photos)
# np.save('/media/sam189239/Backup Plus/dogs_v_cats_labels.npy', labels)


#Pre-processing the images - Method 2 (using ImageDataGenerator)
# random.seed(1)
# val_ratio = 0.25
# for file in os.listdir(folder):
#     src = folder + '/' + file
#     dst_folder = dir + '/Dataset/train/'
#     if random.random()<val_ratio:
#         dst_folder = dir + '/Dataset/test/'
#     if file.startswith('cat'):
#         dst = dst_folder+'cats/'+file
#         move(src,dst)
#     elif file.startswith('dog'):
#         dst = dst_folder+'dogs/'+file
#         move(src,dst)


#define CNN Model

# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3,3),activation = 'relu', kernel_initializer = 'he_uniform', padding = 'same', input_shape = (200, 200, 3)))
#     model.add(MaxPooling2D((2,2)))
#     model.add(Flatten())
#     model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
#     model.add(Dense(1,activation = 'sigmoid'))
    
#     opt = SGD(lr = 0.001, momentum = 0.9)
#     model.compile(optimizer = opt, loss = 'binary_crossentropy',metrics = ['accuracy'])
#     return model

def define_model():
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	for layer in model.layers:
		layer.trainable = False
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	model = Model(inputs=model.inputs, outputs=output)
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model


#plot learning curves

def plot_learning_curves(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	#filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(dir + '/_plot.png')
	pyplot.close()


#run test and evalute model

def run_test():
	model = define_model()
	datagen = ImageDataGenerator(featurewise_center=True)
	datagen.mean = [123.68, 116.779, 103.939]
    #iterators
	train_it = datagen.flow_from_directory(dir + '\\Dataset\\train\\', class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory(dir + '\\Dataset\\test\\', class_mode='binary', batch_size=64, target_size=(200, 200))
    #fit model
	hist = model.fit(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
    #evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	plot_learning_curves(hist)
	model.save('final_model.h5')


run_test()

