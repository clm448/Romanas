# Author: Carmen López Murcia
# This file will contain everything relating to the convolutional neural network that will be used in the projects
# As well as all the pre processing and related tasks
def get_model(shape=(200, 200, 3)):
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import regularizers

    base_model = Sequential()

    base_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=shape,
                                 kernel_regularizer=regularizers.l2(1e-4)))
    base_model.add(layers.MaxPooling2D(2, 2))
    base_model.add(layers.Dropout(0.25))
    base_model.add(layers.BatchNormalization())

    base_model.add(layers.Conv2D(64, (3, 3), activation='relu',
                                 kernel_regularizer=regularizers.l2(1e-4)))
    base_model.add(layers.MaxPooling2D(2, 2))
    base_model.add(layers.Dropout(0.25))
    base_model.add(layers.BatchNormalization())

    base_model.add(layers.Conv2D(128, (3, 3), activation='relu',
                                 kernel_regularizer=regularizers.l2(1e-4)))
    base_model.add(layers.MaxPooling2D(2, 2))
    base_model.add(layers.Dropout(0.25))
    base_model.add(layers.BatchNormalization())

    base_model.add(layers.Flatten())


    base_model.add(layers.Dense(256))
    base_model.add(layers.Dropout(0.25))
    base_model.add(layers.BatchNormalization())

    base_model.add(layers.Dense(1, activation='sigmoid'))

    return base_model


def scheduler(epoch, lr):
  from tensorflow import exp
  if epoch < 20:
    return lr
  else:
    return lr * exp(-0.1)


def plot_history(history):
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def data_preprocessing(train_dir, validation_dir, test_dir, rgb=True, size=(1024, 1024)):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import callbacks

    if rgb:
        shape = size + (3,)

        train_datagen = ImageDataGenerator(dtype='float32', horizontal_flip=True, vertical_flip=True, rescale=1. / 255)
        validation_datagen = ImageDataGenerator(dtype='float32', horizontal_flip=True, vertical_flip=True,
                                                rescale=1. / 255)
        test_datagen = ImageDataGenerator(dtype='float32', horizontal_flip=True, vertical_flip=True, rescale=1. / 255)

        # Get the corresponding images from the directories
        train_generator = train_datagen.flow_from_directory(train_dir, size, batch_size=16, class_mode='binary',
                                                            color_mode="rgb")
        validation_generator = validation_datagen.flow_from_directory(validation_dir, size, batch_size=16,
                                                                      class_mode='binary', color_mode="rgb")
        test_generator = test_datagen.flow_from_directory(test_dir, size, batch_size=16, class_mode='binary',
                                                          color_mode="rgb")

    else:
        shape = size + (1,)

        train_datagen = ImageDataGenerator(dtype='float32', horizontal_flip=True, vertical_flip=True, rescale=1. / 255)
        validation_datagen = ImageDataGenerator(dtype='float32', horizontal_flip=True, vertical_flip=True,
                                                rescale=1. / 255)
        test_datagen = ImageDataGenerator(dtype='float32', horizontal_flip=True, vertical_flip=True, rescale=1. / 255)

        # Get the corresponding images from the directories
        train_generator = train_datagen.flow_from_directory(train_dir, size, batch_size=16, class_mode='binary',
                                                            color_mode="grayscale")
        validation_generator = validation_datagen.flow_from_directory(validation_dir, size, batch_size=16,
                                                                      class_mode='binary', color_mode="grayscale")
        test_generator = test_datagen.flow_from_directory(test_dir, size, batch_size=16, class_mode='binary',
                                                          color_mode="grayscale")

    # Get the desired model
    model = get_model(shape)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.03), loss='binary_crossentropy', metrics=['acc'])

    # Define the callbacks
    my_callbacks = [callbacks.EarlyStopping(monitor='val_acc', patience=10),
                    callbacks.LearningRateScheduler(scheduler),
                    callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_lr=0.001)]

    # Train the model
    history = model.fit_generator(generator=train_generator, epochs=30, validation_data=validation_generator, callbacks = my_callbacks)
    plot_history(history)

    # Test the model
    test_loss, test_acc = model.evaluate(test_generator)

    print("Accuracy:", test_acc)
    print("Loss:", test_loss)

    return model, test_acc, test_loss

