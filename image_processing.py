def get_tif_info(filepath):
    import gdal
    r = gdal.Open(filepath)
    # band = r.GetRasterBand(1)
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = r.GetGeoTransform()
    return upper_left_x, upper_left_y


def tiff_to_np(filepath): # Comprobar si afecta que lo devuelva en vertical y no horizontal
    import gdal
    import numpy as np
    raster = gdal.Open(filepath)
    npArray = raster.ReadAsArray()
    npArray = np.swapaxes(npArray,0,2)
    return npArray


def check_in_range(to_check, range_):
    matches = []
    for elem in to_check:
        if range_[0] < elem < range_[1]:
            matches.append(1)
        else:
            matches.append(0)
    return matches


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
    # ------ 1 -------
    # Check which camps can be found in that image
    # Camps centroids must be between these coordinates
    x_range = (img_coordinates[0], img_coordinates[0]+size[0])
    y_range = (img_coordinates[1], img_coordinates[1]+size[1])
    # Get camps that are in those ranges
    x_coords = camp_centroids['X']
    y_coords = camp_centroids['Y']

    matches_x = check_in_range(x_coords, x_range)
    matches_y = check_in_range(y_coords, y_range)

    # Camps in desired range
    matches = matches_x and matches_y

    # ------ 2 -------
    # Check for no matches in the area
    if sum(matches)==0:
        labels = None
    # If there are matches:
    else:
        # Check in wich division falls each camp
        # Import o generate divisions in image
        horizontal_div = range(0, (size[0]-width+1), stride_x)
        vertical_div = range(0, (size[1]-height+1), stride_y)

    return labels


def labeling(images, labels):
    dataset = images + labels
    return dataset


def trial():
    import pandas as pd
    w = 1000
    h = 1000
    sx = 825
    sy = 896
    size = (14200, 9960) # Cambiarlo por el size del array, aunque todos van a ser iguales
    img_filepath = ''
    camps_filepath = ''
    camps_list = pd.read_csv(camps_filepath)

    img_coordinates = get_tif_info(img_filepath)
    image_npy = tiff_to_np(img_filepath)
    # Crop
    windows = sliding_window(image_npy, w, h, sx, sy)
    # Get labels
    labels = get_labels(img_coordinates, size, camps_list, w, h, sx, sy)
    # Match image to label
    dataset = labeling(windows, labels)

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
