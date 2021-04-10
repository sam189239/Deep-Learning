from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os

dir = 'H:\\Code\\current_files\\My_Code\\Dogs_v_Cats'
folder = dir + '\\Dataset\\test\\'

def load_image(filename):
	img = load_img(filename, target_size=(224, 224))
	img = img_to_array(img)
	img = img.reshape(1, 224, 224, 3)
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

def run():
    print("Calculating Accuracy... ")
    s = 0
    c = 0
    model = load_model(dir + '\\final_model_vgg16.h5')
    for file in os.listdir(folder + "cats\\"):
        img = load_image(folder + "cats\\" + file)    
        result = model.predict(img)
        s = s + int(result[0]<=0.5)
        c = c + 1
    for file in os.listdir(folder + "dogs\\"):
        img = load_image(folder + "dogs\\" + file)    
        result = model.predict(img)
        s = s + int(result[0]>=0.5)
        c = c + 1
    acc = s/c
    print("Accuracy = " + str(acc))

run()