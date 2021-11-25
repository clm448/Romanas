def tiff_to_np(filepath):
    import gdal
    import numpy as np
    raster = gdal.Open(filepath)
    npArray = raster.ReadAsArray()
    return npArray


def sliding_window(big_np_array, width, height, stride_x, stride_y):
    from skimage.util import view_as_window
    # TIFF 14200x9960   ECW 14800x10060
    window_shape = (width, height)
    stride = (stride_x, stride_y)

    # Crop
    windows = view_as_window(big_np_array, window_shape, stride)
    return windows


def get_labels(img_coordinates, size, camp_centroids, width, height, stride_x, stride_y):
    # Check which camps can be found in that image

    # Of those camps, see which falls in which sub division
    # Import o generate divisions in image
    horizontal_div = range(0, (size[0]-width+1), stride_x)
    vertical_div = range(0, (size[1]-height+1), stride_y)

    return labels


def labeling(images, labels):
    dataset = images + labels
    return dataset


def trial():
    w = 1000
    h = 1000
    sx = 825
    sy = 896
    image_npy = 'PNOA_CYL_SW_2009_50cm_OF_rgb_etrs89_hu30_H10_0554_1-1.npy'
    camps_list = ''
    # Crop
    windows = sliding_window(image_npy, w, h, sx, sy)
    # Get labels
    labels = get_labels(camps_list)
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
