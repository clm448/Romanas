# Author: Carmen López Murcia
# Description: Group of functions to formatting a group of .TIF images into a dataset made of regular sized labelled
# jpg images

def dynamic_step(measure, size):
    # Function that returns the step in with to move the slidding window according to the desired size
    # of the final image
    n = measure - size
    for i in range(1, 21):
        if n % i == 0:
            if n / i < 1000:
                step = i
                break
    return int(n/step), step+1


def get_tif_info(filepath):
    # Function that returns the metadata that wants to be keep from the original .TIF images
    import gdal
    r = gdal.Open(filepath)
    band = r.GetRasterBand(1)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = r.GetGeoTransform()
    return upper_left_x, upper_left_y


def tiff_to_np(filepath):
    import gdal
    import numpy as np
    raster = gdal.Open(filepath)
    npArray = raster.ReadAsArray()
    npArray = np.swapaxes(npArray, 0, 2)
    return npArray


def check_in_range(to_check, range_):
    matches = []
    if to_check.shape == ():
        elem = to_check
        if range_[0] < elem < range_[1]:
            matches.append(1)
        else:
            matches.append(0)
    else:
        for elem in to_check:
            if range_[0] < elem < range_[1]:
                matches.append(1)
            else:
                matches.append(0)
    return matches


# noinspection SpellCheckingInspection
def camps_in_image(img_coordinates, camp_centroids, size):
    import numpy as np
    # Check which camps can be found in that image
    # Camps centroids must be between these coordinates
    x_range = (img_coordinates[0], img_coordinates[0] + size[0])
    y_range = (img_coordinates[1], img_coordinates[1] + size[1])
    # Get camps that are in those ranges
    x_coords = camp_centroids['X']
    y_coords = camp_centroids['Y']

    matches_x = check_in_range(x_coords, x_range)
    matches_y = check_in_range(y_coords, y_range)

    # Camps in desired range
    matches = matches_x and matches_y
    matches = np.asarray(matches)
    return matches.astype(dtype=bool)


def sliding_window(big_np_array, width, height, stride_x, stride_y):
    from skimage.util.shape import view_as_windows
    import numpy as np
    # TIFF 14200x9960   ECW 14800x10060
    window_shape = (width, height)
    stride = (stride_x, stride_y)
    # Separate into different channels and crop each channel
    channel_windows_r = view_as_windows(big_np_array[:, :, 0], window_shape, stride)
    channel_windows_g = view_as_windows(big_np_array[:, :, 1], window_shape, stride)
    channel_windows_b = view_as_windows(big_np_array[:, :, 2], window_shape, stride)
    # Stack all channels into a new array
    windows = np.stack((channel_windows_r, channel_windows_g, channel_windows_b), axis=-1)
    return windows


# noinspection SpellCheckingInspection
def get_labels(img_coordinates, size, camp_centroids, width, height, stride_x, stride_y):
    # Check which camps can be found in that image
    matches = camps_in_image(img_coordinates, camp_centroids, size)
    # Check for no matches in the area
    if sum(matches) == 0:
        labels = None
    # If there are matches:
    else:
        # Create a new column that mark witch camps can be found in the image
        camp_centroids['In_Image'] = matches
        camp_centroids = camp_centroids.set_index('In_Image')
        # Keep only the camps that are represented
        camp_centroids = camp_centroids.loc[True]

        x_coords = camp_centroids['X']
        y_coords = camp_centroids['Y']

        # Check in wich division falls each camp
        horizontal_div = range(0, (size[0] - width + 1), stride_x)
        vertical_div = range(0, (size[1] - height + 1), stride_y)

        labels = []
        for hd in horizontal_div:
            range_h = (img_coordinates[0] + hd, img_coordinates[0] + hd + 1000)
            matches_h = check_in_range(x_coords, range_h)
            if sum(matches_h) > 0:
                for vd in vertical_div:
                    range_v = (img_coordinates[1] + vd, img_coordinates[1] + vd + 1000)
                    matches_v = check_in_range(y_coords, range_v)
                    if sum(matches_v) > 0:
                        labels.append(1)
                    else:
                        labels.append(0)
            else:
                labels = labels + [ele for ele in [0] for i in range(len(vertical_div))]
    return labels


def labeling(images, labels, img_filepath, dataset_path, name_start='PNOA'):
    from os.path import exists
    import numpy as np
    from PIL import Image
    import json
    # Get the original's image filename
    img_name = img_filepath[img_filepath.index(name_start):]
    extension = img_name.index('.tif')
    # Go through every image extracted from the original and match them to their tags
    # A dictionary will be used to assign the metadata
    metadata = {}

    hd, vd = images.shape[0:2]
    arr = np.array(labels)
    label_mat = arr.reshape(hd, vd)

    json_path = str(dataset_path+'/metadata_global.json')

    for i in range(hd):
        for j in range(vd):
            # Generate division's filename
            div_name = str('/'+img_name[0:extension]+'_'+str(i)+'_'+str(j))
            # Get corresponding division
            image = images[i, j, 0:1000, 0:1000, 0:3]
            label = label_mat[i, j]
            # Save the division as a new file in jpg format
            i = Image.fromarray(image, 'RGB')
            i.save(str(dataset_path+div_name+'.jpg'))
            # Add metadata to de dictionary
            metadata[div_name] = int(label)
    # Once the whole array has been saved, the global metadata dictionary is updated
    # If there's already saved data
    if exists(json_path):
        f = open(json_path)
        metadata_global = json.load(f)
        metadata_global.update(metadata)
        json_ = json.dumps(metadata_global)
        f = open(json_path, 'w')
        f.write(json_)
        f.close()

    # If this is the first iteration
    else:
        json_ = json.dumps(metadata)
        f = open(json_path, 'w')
        f.write(json_)
        f.close()

    return json_path


def generate_dataset(image_path, camps_path, dataset_path, w=1000, h=1000):
    import pandas as pd
    # Get the data from the image and turn it into a numpy array
    img_coordinates = get_tif_info(image_path)
    image_npy = tiff_to_np(image_path)

    size = image_npy.shape
    sx, div_x = dynamic_step(size[0], w)
    sy, div_y = dynamic_step(size[1], h)

    # Open the csv with all the camps data
    camps_list = pd.read_csv(camps_path)

    # Starting with the image cropping
    windows = sliding_window(image_npy, w, h, sx, sy)

    # Check witch camps are in the current image
    camps = camps_in_image(img_coordinates, camps_list, size)
    camps_list['In_Image'] = camps
    camps_list = camps_list.set_index('In_Image')

    # Get the corresponding labels for the subimages generated from the bigger image
    labels = get_labels(img_coordinates, size, camps_list, w, h, sx, sy)

    # Save the images and the labels
    if labels is None:
        print('No camps in the image '+str(image_path))
    else:
        json_path = labeling(windows, labels, image_path, dataset_path)

    return json_path


def list_to_dict(list_, filename):
    import json
    dict_ = {}
    for elem in list_:
        dict_[elem[0]] = elem[1]
    json = json.dumps(dict_)
    f = open(filename, "w")
    f.write(json)
    f.close()


def data_formatting(json_camp_info, dataset_path, dataset_division=[70, 20, 10]):
    import json
    import numpy as np
    import random
    import os
    import shutil
    import glob

    # Open json file with dataset metadata
    dataset_dict = open(json_camp_info)
    dataset_dict = json.load(dataset_dict)

    # Check how many images are with camps
    array_ = np.asarray(list(dataset_dict.values()))

    n_img_camps = sum(array_ == 1)

    # Separate images according to the presence of camps
    dict_camps = {}
    dict_no_camps = {}
    for elem in dataset_dict:
        if dataset_dict[elem] == 1:
            dict_camps[elem] = 1
        else:
            dict_no_camps[elem] = 0

    # As there are more images without than with camps, chose randomly as many
    # images without camp as with.
    random_dict_no_camps = {}
    for i in range(n_img_camps):
        k, v = random.choice(list(dict_no_camps.items()))
        random_dict_no_camps[k] = v
        dict_no_camps.pop(k)

    # From the images selected, divide them into train, test and validation groups
    # First create the directory tree for the divisions, the remainder images will be stored
    train_dir = os.path.join(dataset_path, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(dataset_path, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(dataset_path, 'test')
    os.mkdir(test_dir)
    other_dir = os.path.join(dataset_path, 'other')
    os.mkdir(other_dir)

    # For each division, separate between images with and without camps
    # Directory with our training camp images
    train_camps_dir = os.path.join(train_dir, 'camps')
    os.mkdir(train_camps_dir)

    # Directory with our training no camps images
    train_no_camps_dir = os.path.join(train_dir, 'no_camps')
    os.mkdir(train_no_camps_dir)

    # Directory with our test camp images
    test_camps_dir = os.path.join(test_dir, 'camps')
    os.mkdir(test_camps_dir)

    # Directory with our test no camp images
    test_no_camps_dir = os.path.join(test_dir, 'no_camps')
    os.mkdir(test_no_camps_dir)

    # Directory with our validation camp images
    validation_camps_dir = os.path.join(validation_dir, 'camps')
    os.mkdir(validation_camps_dir)

    # Directory with our validation no camp images
    validation_no_camps_dir = os.path.join(validation_dir, 'no_camps')
    os.mkdir(validation_no_camps_dir)

    # Get the number of images for each category
    per_train, per_val, per_test = dataset_division
    n_train = round(n_img_camps * per_train / 100)
    n_val = round(n_img_camps * per_test / 100)
    n_test = n_img_camps - n_train - n_val

    div_train = n_train
    div_val = div_train + n_val
    div_test = div_val + n_test

    # Move images to their corresponding directory
    # Train camps
    for elem in list(dict_camps.keys())[0:div_train]:
        src = str(dataset_path + elem + '.npy')
        dst = str(train_camps_dir + elem + '.npy')
        shutil.move(src, dst)
    # Train no camps
    for elem in list(random_dict_no_camps.keys())[0:div_train]:
        src = str(dataset_path + elem + '.npy')
        dst = str(train_no_camps_dir + elem + '.npy')
        shutil.move(src, dst)
    # Save the corresponding dictionaries
    list_to_dict(list(dict_camps.items())[0:div_train], 'train_camps.json')
    list_to_dict(list(random_dict_no_camps.items())[0:div_train], 'train_no_camps.json')

    # Val camps
    for elem in list(dict_camps.keys())[div_test:div_val]:
        src = str(dataset_path + elem + '.npy')
        dst = str(validation_camps_dir + elem + '.npy')
        shutil.move(src, dst)
    # Val no camps
    for elem in list(random_dict_no_camps.keys())[div_test:div_val]:
        src = str(dataset_path + elem + '.npy')
        dst = str(validation_no_camps_dir + elem + '.npy')
        shutil.move(src, dst)
    # Save the corresponding dictionaries
    list_to_dict(list(dict_camps.items())[div_test:div_val], 'val_camps.json')
    list_to_dict(list(random_dict_no_camps.items())[div_test:div_val], 'val_no_camps.json')

    # Test camps
    for elem in list(dict_camps.keys())[div_train:div_test]:
        src = str(dataset_path + elem + '.npy')
        dst = str(test_camps_dir + elem + '.npy')
        shutil.move(src, dst)
    # Test no camps
    for elem in list(random_dict_no_camps.keys())[div_train:div_test]:
        src = str(dataset_path + elem + '.npy')
        dst = str(test_no_camps_dir + elem + '.npy')
        shutil.move(src, dst)
    # Save the corresponding dictionaries
    list_to_dict(list(dict_camps.items())[div_train:div_test], 'test_camps.json')
    list_to_dict(list(random_dict_no_camps.items())[div_train:div_test], 'test_no_camps.json')

    # Save the remaining images in another directory
    for elem in glob.glob1(dataset_path, "*.npy"):
        src = str(dataset_path + '/' + elem)
        dst = str(other_dir + '/' + elem)
        shutil.move(src, dst)

    return [train_dir, test_dir, validation_dir]


    # Pasos a seguir
    # Para el entrenamiento
    # Se tienen todas las imagenes descargadas, hay que dividirlas y etiquetarlas + centroides campamentos
    # Se tiene la imagen
    # A partir del nombre sacar las coordenadas con el shapefile // Metadatos .TIFF, coordenadas de la imagen
    # Una vez se tienen las coordenadas de la imagen se procede a dividirla
    # Se da por hecho que en todas las imagenes grandes va a haber al menos un campamento
    # Se debe identificar en que subdivisión se encuentra el campamento
    # Al cortarlo se debe añadir un label que indique si la imagen tiene o no campamento

# En la evaluación, a partir de las coordenadas se descarga la imagen y se debe dividir
