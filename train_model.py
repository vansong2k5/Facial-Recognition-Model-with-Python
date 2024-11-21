import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.src.utils import to_categorical
from PIL import Image

TRAIN_DATA = 'train_data'
list_of_data = []

# change key here 
############################
dict = {
        'bao_lam': 0,
        'q_hai': 1,
        'tran_thanh': 2,
        'truong_giang': 3,
        'tu_long': 4,
        'xuan_bac': 5
}
############################

for label in os.listdir(TRAIN_DATA):
        label_path = os.path.join(TRAIN_DATA, label)
        label = label_path.split('\\')[1]

        if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                        filename_path = os.path.join(label_path, filename)

                        img = np.array(Image.open(filename_path).convert('L'))
                        list_of_data.append((img, dict[label]))

def prepare_data(list_of_data):
    images = []
    labels = []
    
    for img, label in list_of_data:
        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    labels = to_categorical(labels)

    return train_test_split(images, labels, test_size=0.2, random_state=1)

Xtrain, Xtest, ytrain, ytest = prepare_data(list_of_data)

Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0

Model = models.Sequential()
shape = (64, 64, 1)

Model.add(layers.Conv2D(32, (3, 3), padding="same", input_shape=shape))
Model.add(layers.Activation("relu"))
Model.add(layers.Conv2D(32, (3, 3), padding="same"))
Model.add(layers.Activation("relu"))
Model.add(layers.MaxPooling2D(pool_size=(2, 2)))

Model.add(layers.Conv2D(64, (3, 3), padding="same"))
Model.add(layers.Activation("relu"))
Model.add(layers.MaxPooling2D(pool_size=(2, 2)))

Model.add(layers.Flatten())
Model.add(layers.Dense(512))
Model.add(layers.Activation("relu"))
Model.add(layers.Dense(6))  
Model.add(layers.Activation("softmax"))

Model.summary()

Model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Start training")
Model.fit(Xtrain, ytrain, batch_size=6, epochs=10)

Model.save("khuonmat.h5")