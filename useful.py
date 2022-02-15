# Author: Carmen López Murcia
# converter from numpy to jpg, used in testing
def numpy_to_image(path, image_extension='.jpg', name_start='PNOA'):
    import glob
    import os
    import numpy as np
    from PIL import Image
    for numpy_image in glob.iglob(str(path + '/*.npy')):
        img_name = numpy_image[numpy_image.index(name_start):]
        file_path = numpy_image[:numpy_image.index(name_start)]
        extension = img_name.index('.npy')

        filename = str(file_path + img_name[0:extension] + image_extension)

        data = np.load(numpy_image)
        i = Image.fromarray(data, 'RGB')
        i.save(filename)

        os.remove(numpy_image)