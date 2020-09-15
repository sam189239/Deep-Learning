from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

dir = 'H:\\Code\\current_files\\My_Code\\Dogs_v_Cats'

def load_image(filename):
	img = load_img(filename, target_size=(224, 224))
	img = img_to_array(img)
	img = img.reshape(1, 224, 224, 3)
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 
def run():
	img = load_image(dir + '\\Dataset\\train\\cats\\cat.12000.jpg')
	model = load_model(dir + '\\final_model_vgg16.h5')
	result = model.predict(img)
	if result[0]>=0.5:
		print("It's a Dog!")
	else:
		print("It's a Cat!")
 
run()