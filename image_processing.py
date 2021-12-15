def dynamic_step(measure, size):
    n = measure - size
    for i in range(1, 21):
        if n % i == 0:
            if n / i < 1000:
                step = i
                break
    return int(n/step), step+1


def get_tif_info(filepath):
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
        for vd in vertical_div:
            range_v = (img_coordinates[1] + vd, img_coordinates[1] + vd + 1000)
            matches_v = check_in_range(y_coords, range_v)
            if sum(matches_v) > 0:
                for hd in horizontal_div:
                    range_h = (img_coordinates[0] + hd, img_coordinates[0] + hd + 1000)
                    matches_h = check_in_range(x_coords, range_h)
                    if sum(matches_h) > 0:
                        labels.append(1)
                    else:
                        labels.append(0)
            else:
                labels = labels + [ele for ele in [0] for i in range(len(horizontal_div))]
    return labels


def labeling(images, labels):
    dataset = images + labels


def generate_dataset(image_path, camps_path, w=1000, h=1000, sx=825, sy=896):
    import matplotlib as plt
    import pandas as pd
    # Get the data from the image and turn it into a numpy array
    img_coordinates = get_tif_info(image_path)
    image_npy = tiff_to_np(image_path)

    plt.imshow(image_npy)

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

    labeling(windows, labels)

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
