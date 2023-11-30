# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras_facenet import FaceNet


embedder = FaceNet()

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	detections = model.extract(face_pixels, threshold = 0.95)
	# standardize pixel values across channels (global)
	#mean, std = face_pixels.mean(), face_pixels.std()
	#face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	# yhat = model.predict(samples)
	yhat = model.embeddings(samples)

	return yhat[0]

# load the face dataset
data = load('ss.npz')

trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
# load the facenet model
# model = load_model('facenet_keras.h5')
# model.summary()
model = embedder
# model = SVC(kernel='linear', probability=True)
# model = load_model('face-rec_Google.npz')
# model = load('test.npz')
# model = tf.keras.applications.ResNet50(weights='imagenet')

print('Loaded Model')
# convert each face in the train set to an embedding
newTrainX = list()
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)
	newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
# convert each face in the test set to an embedding
newTestX = list()
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)
	newTestX.append(embedding)
newTestX = asarray(newTestX)
print(newTestX.shape)
# save arrays to one file in compressed format
savez_compressed('ss-embd.npz', newTrainX, trainy, newTestX, testy)




