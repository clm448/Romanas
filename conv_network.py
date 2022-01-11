# This file will contain everything relating to the convolutional neural network that will be used in the projects
# As well as all the pre processing and related tasks
def get_sample(paths):
    import glob
    import cv2
    import numpy as np

    sample = None
    for path in paths:
        for elem in glob.glob(str(path + '/*.jpg')):
            img = cv2.imread(elem)
            img = np.expand_dims(img, axis=0)

            if sample is None:
                sample = img
            else:
                sample = np.concatenate((sample, img), axis=0)
    return sample


def get_model(shape=(200, 200, 3)):
    from tensorflow.keras import layers
    from tensorflow.keras import models

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def data_preprocessing(train_dir, validation_dir, test_dir, sample_dir_paths, size=(200, 200)):
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import optimizers

    img_sample = get_sample(sample_dir_paths)

    # Create the Data generator associated to each data division
    train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                       horizontal_flip=True, vertical_flip=True)
    validation_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                            horizontal_flip=True, vertical_flip=True)
    test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True,
                                      vertical_flip=True)

    # fit the data augmentation
    train_datagen.fit(img_sample)
    validation_datagen.fit(img_sample)
    test_datagen.fit(img_sample)

    # Get the corresponding images from the directories
    train_generator = train_datagen.flow_from_directory(train_dir, size, batch_size=20, class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory(validation_dir, size, batch_size=20,
                                                                  class_mode='binary')
    test_generator = test_datagen.flow_from_directory(test_dir, size, batch_size=20, class_mode='binary')

    # Get the desired model
    model = get_model()

    # Compile the model
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])

    # Train the model
    history = model.fit_generator(generator=train_generator, epochs=30, validation_data=validation_generator)

    # Test the model
    test_lost, test_acc = model.evaluate(test_generator)

    print("Accuracy:", test_acc)
    print("Loss:", test_lost)

    return (model, test_acc, test_lost)