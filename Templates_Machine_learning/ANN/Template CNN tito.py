#Accuracies acima de 90%
#https://www.udemy.com/deeplearning/learn/v4/questions/2276518

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#Conv2D(32, (3, 3) = 32 é o número de feature maps que será criado. (3, 3) é o número de colunas e linhas do feature detector. Como estamos usando CPU, 32 é um bom número. Caso esteja usando GPU, 64, 128 e 256 podem ser usados.
#input_shape = (64, 64, 3) = 64, 64 é o dimensional do array. Para GPU, pode-se utilizar 256, 256. O 3 é quanto as imagens são coloridas. Quando for preto e brano utilizar 1 ao invés de 3.
#activation = 'relu' ]e me lhor função para garantir a noa linearidade das fotos.

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# pool_size = Recomendado utilizar 2 x 2.

# Adding a second convolutional layer (Para obter um melhor test set accuracy. Não é necessário input_shape = (64, 64, 3))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#Caso queira adicionar mais terceiro convolutional layer, para obter um melhor resultado passar Conv2D(32, (3, 3) para Conv2D(64, (3, 3). Caso queira adicionar terceiro convolutional layer Conv2D(128, (3, 3) Sempre dobrando.

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#units = 100 é um ótimo número.
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# activation = Como temos uma classificaçãobinária, será utilizado 'sigmoid'. Caso não seja binário, utilizar softmax function

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer = algorítmo para selecionar o melhor peso do ANN. 'Adam' é um dos mais utilizados
#loss =  Algorítmo que minimiza as perdas do gradiente descendete estocástico. Como a saida é binária, utilizou binary_crossentropy. Se houver mais que uma variável categorical_crossentropy
#metrics = Padrão

# Part 2 - Fitting the CNN to the images 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
#target_size = Tem que ser igual input_shape do # Step 1 - Convolution.(64, 64, 3)
#batch_size = Manter 32
#class_mode =" binary" quando binário. Quando houver mais de uma classe, utilizar categorical

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#target_size = Tem que ser igual input_shape do # Step 1 - Convolution.(64, 64, 3)
#batch_size = Manter 32
#class_mode =" binary" quando binário. Quando houver mais de uma classe, utilizar categorical

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
# steps_per_epoch = número de fotos na pasta training set
# epochs = 25 é um bom número.
# validation_steps = número de fotos na pasta test set

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
    

