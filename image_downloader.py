# Function that returns the df entry corresponding to a pair of coordinates (lat, long)
def find_point(x, y, df_ref):
    # Function that returns the df entry corresponding to a pair of coordinates (lat, long)
    """Given lat and long, returns the shapefile entry."""
    cont = 0
    print("Looking for this pair of points: ({},{})".format(x, y))
    for index, row in df_ref.iterrows():
        cont = cont + 1
        # print just a pair of boundaries in the df, to see if the format matches
        # the input
        print("Comparing with this boundaries: X = ({},{}), Y = ({},{})".format(row['X_south_array'],
                                                                                row['X_north_array'],
                                                                                row['Y_west_array'],
                                                                                row['Y_east_array']))
        if (row['X_south_array'] <= x < row['X_north_array'] and
                row['Y_west_array'] <= y < row['Y_east_array']):
            print(row['X_south_array'], row['X_north_array'], row['Y_west_array'], row['Y_east_array'])
            break
    # return cont - 1
    return cont


def get_dataframe_from_shapefile(sf):
    # The function receive a shapefile in input and returns a dataframe with the
    # coordinates of all the points
    """Given a shapefile object, returns a dataframe with the coordinates"""
    import pandas as pd
    import numpy as np

    # Get the shape object from the shapefile
    shapes = sf.shapes()

    # Prepare the lists for the coordinates
    X_south_list = []
    X_north_list = []
    Y_east_list = []
    Y_west_list = []

    # Fill the lists
    for i in range(len(shapes)):
        s = sf.shape(i)
        Y_west_list.append(s.bbox[0])
        X_south_list.append(s.bbox[1])
        Y_east_list.append(s.bbox[2])
        X_north_list.append(s.bbox[3])

    # Convert the lists to np.arrays to ease the dataframe creation
    X_south_array = np.array(X_south_list)
    X_north_array = np.array(X_north_list)
    Y_east_array = np.array(Y_east_list)
    Y_west_array = np.array(Y_west_list)

    # Create a dataframe
    df_coordinates = pd.DataFrame()
    df_coordinates['X_south_array'] = X_south_array
    df_coordinates['X_north_array'] = X_north_array
    df_coordinates['Y_east_array'] = Y_east_array
    df_coordinates['Y_west_array'] = Y_west_array

    df_coordinates.head()
    return df_coordinates


def map_plot(sf, shapes):
    # Plot all the points in a 2D map
    import numpy as np
    import matplotlib.pyplot as plt

    X_list = []
    Y_list = []
    for i in range(len(shapes)):
        s = sf.shape(i)
        X_list.append(0.5 * (s.bbox[0] + s.bbox[2]))
        Y_list.append(0.5 * (s.bbox[1] + s.bbox[3]))
    X_array = np.array(X_list)
    Y_array = np.array(Y_list)

    # Actual plotting
    fig, ax = plt.subplots()
    ax.scatter(x=X_array, y=Y_array, s=5)


def get_leaf(x, y, shp):
    # Function that returns the leaf in which the coordinates can be found
    import shapefile
    # Load shapefile
    sf = shapefile.Reader(shp)

    # Reading geometry
    shapes = sf.shapes()
    map_plot(sf, shapes)

    df = get_dataframe_from_shapefile(sf)

    leaf = find_point(x, y, df)
    return leaf


def check_route(leaf_name, urls):
    # The function checks if the given address is correct
    import requests
    status = 'failure'
    for url in urls:
        r_ = url + 'Color/' + leaf_name
        request = requests.get(r_)

        print("Trying the following route: {}".format(r_))

        if request.status_code == 200:
            status = 'success'
            route = r_
            break
        else:
            route = r_
            print("Bad request status code!")
    return status, route

    # The function builds the address of a given leaf


def get_route(leaf, urls):
    # The function builds the address of a given leaf
    """Builds the address of a leaf"""
    Leaf_name = 'H-' + str(leaf).zfill(4) + '/'
    status, my_route = check_route(Leaf_name, urls)
    if status == 'success':
        return my_route


def get_images_urls(url):
    import requests
    from bs4 import BeautifulSoup
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    urls_tif = []
    urls_ECW = []
    for link in soup.find_all('a'):
        link = link.get('href')
        if link.endswith(".tif"):
            urls_tif.append(link)
        if link.endswith(".ECW"):
            urls_ECW.append(link)

    if len(urls_tif) > 0:
        urls = urls_tif
        format_tif = 1
    elif len(urls_ECW) > 0:
        urls = urls_ECW
        format_tif = 0
    else:
        print('Error')
    return urls, format_tif


def image_downloader(myx, myy, shp):
    # Library needed to download maps from url
    import requests
    import sys, os

    # Web pages with the maps database
    urls = ["http://ftp.itacyl.es/cartografia/01_Ortofotografia/2009/",
            "http://ftp.itacyl.es/cartografia/01_Ortofotografia/2010/"]

    # Create directory to store images
    output_folder = "maps"
    os.system("mkdir -p {}".format(output_folder))

    # Find the leaf that contains your coordinates
    leaf = get_leaf(x=myx, y=myy, shp=shp)

    # Get the web route to that leaf
    my_route = get_route(leaf, urls)
    # route=get_route(56,urls)

    # Get the urls to all images to download
    images_urls, format_tif = get_images_urls(my_route)

    # Download images
    for image_url in images_urls:
        # print(ruta+image_url)
        response = requests.get(my_route + image_url)
        file = open(output_folder + '/' + image_url, "wb")
        file.write(response.content)
        file.close()

    return format_tif
