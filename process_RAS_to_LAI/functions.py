import numpy as np
from PIL import Image
import math
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


_errstr = "Mode is unknown or incompatible with input array shape."

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

def num_to_rgb(val, max_val=18):
    i = (val * 255 / max_val);
    r = round(math.sin(0.024 * i + 0) * 127 + 128);
    g = round(math.sin(0.024 * i + 2) * 127 + 128);
    b = round(math.sin(0.024 * i + 4) * 127 + 128);
    return [r,g,b]

def num_to_rgb_all(val, max_val=18):
    i = (val * 255 / max_val);
    r = np.round(np.sin(0.024 * i + 0) * 127 + 128);
    g = np.round(np.sin(0.024 * i + 2) * 127 + 128);
    b = np.round(np.sin(0.024 * i + 4) * 127 + 128);
    rgbs = np.dstack((r,g,b)).reshape(-1,3)
    return rgbs


def extract_image_from_RAS_file(datapath, filename, image_length, image_width, select_image):
    file = open(datapath + filename, "r")
    a = np.fromfile(file, dtype=np.int16)
    
    count_column = 0
    count_line = 0
    image = np.zeros((image_length, image_width, 3), dtype=np.int16)
    for pixel in a[select_image*(image_length*image_width):(select_image+1)*(image_length*image_width)]:
        if count_line == image_width:
            break
        if pixel == -999:
            image[count_line, count_column] = [255, 0, 0]
        elif pixel == -911 or pixel == -920 or len(str(pixel)) < 3:
            image[count_line, count_column] = [100, 100, 100]
        elif pixel == -910:
            image[count_line, count_column] = [0, 0, 0]
        elif pixel == -961:
            image[count_line, count_column] = [48, 138, 78]
        elif pixel == -950:
            image[count_line, count_column] = [143, 72, 56]
        elif pixel == -940:
            image[count_line, count_column] = [70, 190, 199]
        elif pixel == -930:
            image[count_line, count_column] = [235, 246, 247]
        else:
            print("float(str(pixel)[:-2])/40.0", float(str(pixel)[:-2])/40.0)
            print("float(str(pixel)[:-2])",float(str(pixel)[:-2]))
            print("str(pixel)[:-2]", str(pixel)[:-2])
            print(" str(pixel)", str(pixel))
            print("pixel", pixel)
            print("pixel//100", (pixel//100)*1.0)
            image[count_line, count_column] = num_to_rgb(float(str(pixel)[:-2])/40.0)
        count_column += 1
        if count_column == image_length:
            count_column = 0
            count_line += 1
            if count_line % 100 == 0:
                print(count_line)

    img = toimage(image)
    return img

def extract_image_from_RAS_file_cupd_all(datapath, filename, image_length, image_width):
    file = open(datapath + filename, "r")
    a_all = np.fromfile(file, dtype=np.int16)
    cluster_len = a_all.shape[0] // (image_width * image_width)
    all_image_array = []
    all_imgs = []
    for select_image in range(cluster_len):
        
        image = np.zeros((image_length, image_width, 3), dtype=np.int16)
        a = a_all[select_image*(image_length*image_width):(select_image+1)*(image_length*image_width)].reshape(image_length, image_width)

        image[np.where(a == -999)[0], np.where(a == -999)[1]] == [255, 0, 0]
        image[np.where(a == -911)[0], np.where(a == -911)[1]] == [100, 100, 100]
        image[np.where(a == -920)[0], np.where(a == -920)[1]] == [100, 100, 100]
        image[np.where(a == -910)[0], np.where(a == -910)[1]] == [0, 0, 0]
        image[np.where(a == -961)[0], np.where(a == -961)[1]] == [48, 138, 78]
        image[np.where(a == -950)[0], np.where(a == -950)[1]] == [143, 72, 56]
        image[np.where(a == -940)[0], np.where(a == -940)[1]] == [70, 190, 199]
        image[np.where(a == -930)[0], np.where(a == -930)[1]] == [235, 246, 247]
        image[np.where(a >= 0)[0], np.where(a >= 0)[1]] = num_to_rgb_all((a[np.where(a >= 0)]//100)/40.0)
        
        img = toimage(image)
        all_imgs.append(img)

        all_image_array.append(image)

    return all_imgs, all_image_array


def extract_image_from_RAS_file_cupd(datapath, filename, image_length, image_width, select_image):
    file = open(datapath + filename, "r")
    a = np.fromfile(file, dtype=np.int16)
    
    image = np.zeros((image_length, image_width, 3), dtype=np.int16)

    a = a[select_image*(image_length*image_width):(select_image+1)*(image_length*image_width)].reshape(image_length, image_width)

    image[np.where(a == -999)[0], np.where(a == -999)[1]] == [255, 0, 0]
    image[np.where(a == -911)[0], np.where(a == -911)[1]] == [100, 100, 100]
    image[np.where(a == -920)[0], np.where(a == -920)[1]] == [100, 100, 100]
    image[np.where(a == -910)[0], np.where(a == -910)[1]] == [0, 0, 0]
    image[np.where(a == -961)[0], np.where(a == -961)[1]] == [48, 138, 78]
    image[np.where(a == -950)[0], np.where(a == -950)[1]] == [143, 72, 56]
    image[np.where(a == -940)[0], np.where(a == -940)[1]] == [70, 190, 199]
    image[np.where(a == -930)[0], np.where(a == -930)[1]] == [235, 246, 247]
    image[np.where(a >= 0)[0], np.where(a >= 0)[1]] = num_to_rgb_all((a[np.where(a >= 0)]//100)/40.0)
    
    img = toimage(image)

    return img, image

def extract_image_from_RAS_file_new(datapath, filename, image_length, image_width, select_image):
    file = open(datapath + filename, "r")
    a = np.fromfile(file, dtype=np.int16)
    img_raw = a[select_image*(image_length*image_width):(select_image+1)*(image_length*image_width)].reshape(image_length,image_width)
    """
    -999 = missing value (blackfill) -> should be interpolated!
	-961 = forest
	-950 = urban areas
	-940 = water
	-930 = snow
    -920 = cloud shadow -> should be interpolated!
    -911 = cirrus clouds -> should be interpolated!
    -910 = clouds -> should be interpolated!
    """
    ind_misvalue = (img_raw == -999)
    ind_forest  = (img_raw == -961)
    ind_urban = (img_raw == -950)
    ind_water = (img_raw == -940)
    ind_snow = (img_raw == -930)
    ind_cloudshadow  = (img_raw == -920)
    ind_cirrusclouds  = (img_raw == -911)
    ind_clouds = (img_raw == -910) 
    ind_invalid= (img_raw < 0) & np.logical_not(ind_misvalue |ind_forest|ind_urban|ind_water|ind_snow|ind_cloudshadow|ind_cirrusclouds|ind_clouds)
    ind_small = (img_raw // 100) > 3
    ind_lai = (img_raw // 100) >=3
    img_np = np.zeros((image_length, image_width, 3), dtype=np.int16)
    img_np[(ind_misvalue |ind_invalid)] = [255,0,0] 
    img_np[(ind_cirrusclouds | ind_cloudshadow | ind_small)] = [100, 100, 100]
    img_np[ind_clouds] = [0, 0, 0]
    img_np[ind_forest] = [48, 138, 78]
    img_np[ind_urban] = [143, 72, 56]
    img_np[ind_water] = [70, 190, 199]
    img_np[ind_snow] = [235, 246, 247]

    max_val = 18
    ir= (img_raw[ind_lai] //100)/40.
    i = (ir * 255 / max_val)
    r = np.around(np.sin(0.024 * i + 0) *127 + 128)
    g = np.around(np.sin(0.024 * i + 2) *127 + 128)
    b = np.around(np.sin(0.024 * i + 4) *127 + 128)

    img_np[ind_lai] = np.stack((r,g,b),axis=-1)

    img = toimage(img_np)
    return img



def extract_all_images_frmo_RAS_folder(folder, image_length, image_width):
    for filename in os.listdir(folder):
        if 'RAS' in filename:
            print(filename)
            with open(os.path.join(folder, filename), 'r') as f:            
                a = np.fromfile(f, dtype=np.int16)
                count = 0
                count_column = 0
                count_line = 0
                image = np.zeros((image_length, image_width, 3), dtype=np.int16)
                for pixel in a:
                    if count_line == image_width:
                        img = toimage(image)
                        img = img.save(images_folder + filename[:-4] + '_' + str(count) + '.png')
                        print('Done ' + filename)
                        count += 1
                        count_column = 0
                        count_line = 0
                        image = np.zeros((image_length, image_width, 3), dtype=np.int16)
                    if pixel == -999:
                        image[count_line, count_column] = [255, 0, 0]
                    elif pixel == -911 or pixel == -920 or len(str(pixel)) < 3:
                        image[count_line, count_column] = [100, 100, 100]
                    elif pixel == -910:
                        image[count_line, count_column] = [0, 0, 0]
                    elif pixel == -961:
                        image[count_line, count_column] = [48, 138, 78]
                    elif pixel == -950:
                        image[count_line, count_column] = [143, 72, 56]
                    elif pixel == -940:
                        image[count_line, count_column] = [70, 190, 199]
                    elif pixel == -930:
                        image[count_line, count_column] = [235, 246, 247]
                    else:
                        image[count_line, count_column] = num_to_rgb(float(str(pixel)[:-2])/40.0)
                    count_column += 1
                    if count_column == image_length:
                        count_column = 0
                        count_line += 1
                        if count_line % 1000 == 0:
                            print(count_line)
                count+=1
                img = toimage(image)
                img = img.save(folder + filename[:-4] + '_' + str(count) + '.png')
                print('Done ' + filename)
            
            
def detect_fields(filename):
    image = cv2.imread(filename)
  
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Canny edges
    t_lower = 10 # Lower Threshold
    t_upper = 100 # Upper threshold
    aperture_size = 3 # Aperture size
    L2Gradient = False # Boolean
    edged = cv2.Canny(gray, t_lower, t_upper,
                     apertureSize = aperture_size, 
                     L2gradient = L2Gradient)

    # Find Contours
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print("Number of Contours found = " + str(len(contours)))

    final_contours = []
    for cnt in contours:
        epsilon = 0.08*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 4 and len(approx) < 20:
            if cv2.contourArea(cnt) > cv2.arcLength(cnt, True):
                final_contours.append(cnt)

    cv2.polylines(image, final_contours, isClosed=True, color=(255, 0, 0), thickness=2)
    return image, final_contours

def extract_ts_from_detected_fields(datapath, final_contours, image_length, image_width):
    contours = {}
    shapes = []
    count = 0
    for contour in final_contours:
        indexes = []
        points = []
        for c in contour:
            indexes.append((image_width*c[0][0]) + c[0][1])
            points.append((c[0][0], c[0][1]))
        contours[count] = indexes
        count += 1
        polygon = Polygon(points)
        shapes.append(polygon)
        
    field_ts = {}
    for filename in os.listdir(datapath):
        if 'RAS' in filename:
            dates = []
            with open(os.path.join(datapath, filename[:-4]+'.RHD'), 'r') as f: 
                lines = f.readlines()
                for i in range(5,len(lines)):
                    date = lines[i][6:10] + '-' + lines[i][11:14] + '-' + lines[i][14:17]
                    dtime = datetime.strptime(date.replace(' ', ''), '%Y-%m-%d')
                    dates.append(dtime)
            with open(os.path.join(datapath, filename), 'r') as f:            
                a = np.fromfile(f, dtype=np.int16)
                for i in range(len(dates)):
                    contour_ts = {}
                    for c in contours.keys():
                        if c in contour_ts.keys():
                            for p in contours[c]:
                                contour_ts[c].append(a[p+(i*image_length*image_width)])
                        else:
                            contour_ts[c] = []
                            for p in contours[c]:
                                contour_ts[c].append(a[p*(i+1)])
                    field_ts[dates[i]] = contour_ts
                print('Done ' + filename)
    field_ts = pd.DataFrame.from_dict(field_ts)
    return field_ts


def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=600, bbox_inches=bbox)

    
def create_legend():
    labels = ["Missing Values", "Forests", "Urban Areas", "Water", "Snow"]
    colors = [[0,0,0], [48/255,138/255,78/255], [143/255,72/255,56/255], [70/255,190/255,199/255], [235/255,246/255,247/255]]
    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(5)]
    legend = plt.legend(handles, labels, loc=3, framealpha=1, frameon=True)
    export_legend(legend)
    
    
def explore_image(filepath):
    
    def onclick(event): 
        print("button=%d, x=%d, y=%d, xdata=%f, ydata=%f" % (event.button, event.x, event.y, event.xdata, event.ydata)) 

    plt.figure()
    img1 = mpimg.imread(filepath)
    ax1 = plt.imshow(img1)
    fig1 = ax1.get_figure()
    cid1 = fig1.canvas.mpl_connect('button_press_event', onclick) 

    plt.show()
    
def extract_LAI_from_RAS_file_OLD(datapath, filename, image_length, image_width, select_image):
    file = open(datapath + filename, "r")
    a = np.fromfile(file, dtype=np.int16)
    print("i sit here?")
    count_column = 0
    count_line = 0
    image = np.zeros((image_length, image_width, 1), dtype=np.int16)
    for pixel in a[select_image*(image_length*image_width):(select_image+1)*(image_length*image_width)]:
        if count_line == image_width:
            break

        if len(str(pixel)) < 3:
            image[count_line, count_column] = 0
        elif pixel < 0:
            image[count_line, count_column] = pixel
        else:
            image[count_line, count_column] = float(str(pixel)[:-2])

        count_column += 1
        if count_column == image_length:
            count_column = 0
            count_line += 1
            if count_line % 100 == 0:
                print(count_line)
            
    return image
   
def extract_LAI_from_RAS_file(datapath, filename, image_length, image_width, select_image):
    file = open(datapath + filename, "r")
    a = np.fromfile(file, dtype=np.int16)
    print('a.shape', a.shape[0])
    print('a.shape', a.shape[0] // (10980 * 10980))

    print('select_image 1', select_image)
    img = a[select_image*(image_length*image_width):(select_image+1)*(image_length*image_width)].reshape(image_length,image_width)
    print('select_image 2', select_image)
    mask = img <= 0
    negvals = img[mask]
    
    # Temporary, for getting first digits without errors
    img[mask] = 1
    
    # Get first 3 digits
    img = img * 10**(4 - np.log10(img).astype(int)) // 100

    lai = img / 40
    
    # Set back to old values
    lai[mask] = negvals
    
    return lai

def extract_all_LAI_from_RAS_file(datapath_filename, image_length, image_width):
    file = open(datapath_filename, "r")
    a = np.fromfile(file, dtype=np.int16)
    print("a.shape", a.shape)
    cluster_len = a.shape[0] // (image_width * image_width)
    print("cluster_len", cluster_len)
    all_lai = np.array([])
    for select_image in range(cluster_len):

        img = a[select_image*(image_length*image_width):(select_image+1)*(image_length*image_width)].reshape(image_length,image_width)
        
        mask = img <= 0
        negvals = img[mask]
        
        # Temporary, for getting first digits without errors
        img[mask] = 1
        
        # Get first 3 digits
        #img = img * 10**(4 - np.log10(img).astype(int)) // 100
        img = img // 100

        lai = img / 40
        
        # Set back to old values
        lai[mask] = negvals
        
        all_lai = np.append(all_lai, lai)
    all_lai = all_lai.reshape(cluster_len, image_length, image_width)
    return all_lai

def extract_spec_LAI_from_RAS_file(datapath_filename, cluster_ind, image_length, image_width):
    file = open(datapath_filename, "r")
    a = np.fromfile(file, dtype=np.int16)
    #print("a.shape", a.shape)
    cluster_len = a.shape[0] // (image_width * image_width)
    #print("cluster_len", cluster_len)
    all_lai = np.array([])
    #for select_image in range(cluster_len):

    img = a[cluster_ind*(image_length*image_width):(cluster_ind+1)*(image_length*image_width)].reshape(image_length,image_width)
    
    mask = img <= 0
    negvals = img[mask]
    
    # Temporary, for getting first digits without errors
    img[mask] = 1
    
    # Get first 3 digits
    #img = img * 10**(4 - np.log10(img).astype(int)) // 100
    img = img // 100

    lai = img / 40
    
    # Set back to old values
    lai[mask] = negvals
    
    #all_lai = np.append(all_lai, lai)
    #all_lai = all_lai.reshape(cluster_len, image_length, image_width)
    return lai

def get_cluster_length(datapath_filename, image_length, image_width):
    file = open(datapath_filename, "r")
    a = np.fromfile(file, dtype=np.int16)
    #print("a.shape", a.shape)
    cluster_len = a.shape[0] // (image_width * image_width)
    return cluster_len

def expand_pixel(image, pixel_of_interest, thresh):
    curr_point_set = {pixel_of_interest}
    loop_over = curr_point_set.copy()
    prev_point_set = curr_point_set.copy()
    max_x = 0
    max_y = 0
    min_x = 20000
    min_y = 20000
    thresh = 0.1

    while True:
        for point in loop_over:
            up_left = (point[0]-1, point[1]-1)
            up = (point[0]-1, point[1])
            up_right = (point[0]-1, point[1]+1)
            right = (point[0], point[1]+1)
            down_right = (point[0]+1, point[1]+1)
            down = (point[0]+1, point[1])
            down_left = (point[0]+1, point[1]-1)
            left = (point[0], point[1]-1)

            if abs(int(image[up_left])-int(image[point])) < int(image[point])*thresh and int(image[up_left]) > 0:
                curr_point_set.add(up_left)
                if up_left[0] > max_x:
                    max_x = up_left[0]
                if up_left[0] < min_x:
                    min_x = up_left[0]
                if up_left[1] > max_y:
                    max_y = up_left[1]
                if up_left[1] < min_y:
                    min_y = up_left[1]
            if abs(int(image[up])-int(image[point])) < int(image[point])*thresh and int(image[up]) > 0:
                curr_point_set.add(up)
                if up[0] > max_x:
                    max_x = up[0]
                if up[0] < min_x:
                    min_x = up[0]
                if up[1] > max_y:
                    max_y = up[1]
                if up[1] < min_y:
                    min_y = up[1]
            if abs(int(image[up_right])-int(image[point])) < int(image[point])*thresh and int(image[up_right]) > 0:
                curr_point_set.add(up_right)
                if up_right[0] > max_x:
                    max_x = up_right[0]
                if up_right[0] < min_x:
                    min_x = up_right[0]
                if up_right[1] > max_y:
                    max_y = up_right[1]
                if up_right[1] < min_y:
                    min_y = up_right[1]
            if (abs(int(image[right])-int(image[point])) < int(image[point])*thresh) and (int(image[right]) > 0):
                curr_point_set.add(right)
                if right[0] > max_x:
                    max_x = right[0]
                if right[0] < min_x:
                    min_x = right[0]
                if right[1] > max_y:
                    max_y = right[1]
                if right[1] < min_y:
                    min_y = right[1]
            if abs(int(image[down_right])-int(image[point])) < int(image[point])*thresh and int(image[down_right]) > 0:
                curr_point_set.add(down_right)
                if down_right[0] > max_x:
                    max_x = down_right[0]
                if down_right[0] < min_x:
                    min_x = down_right[0]
                if down_right[1] > max_y:
                    max_y = down_right[1]
                if down_right[1] < min_y:
                    min_y = down_right[1]
            if abs(int(image[down])-int(image[point])) < int(image[point])*thresh and int(image[down]) > 0:
                curr_point_set.add(down)
                if down[0] > max_x:
                    max_x = down[0]
                if down[0] < min_x:
                    min_x = down[0]
                if down[1] > max_y:
                    max_y = down[1]
                if down[1] < min_y:
                    min_y = up_left[1]
            if abs(int(image[down_left])-int(image[point])) < int(image[point])*thresh and int(image[down_left]) > 0:
                curr_point_set.add(down_left)
                if down_left[0] > max_x:
                    max_x = down_left[0]
                if down_left[0] < min_x:
                    min_x = down_left[0]
                if down_left[1] > max_y:
                    max_y = down_left[1]
                if down_left[1] < min_y:
                    min_y = down_left[1]
            if abs(int(image[left])-int(image[point])) < int(image[point])*thresh and int(image[left]) > 0:
                curr_point_set.add(left)
                if left[0] > max_x:
                    max_x = left[0]
                if left[0] < min_x:
                    min_x = left[0]
                if left[1] > max_y:
                    max_y = left[1]
                if left[1] < min_y:
                    min_y = left[1]

        if len(curr_point_set - prev_point_set) == 0:
            break
        else:
            loop_over = curr_point_set - prev_point_set
            prev_point_set = curr_point_set.copy()
            
    field = np.zeros((max_x-min_x+1, max_y-min_y+1, 3), dtype=np.int16)
    for point in curr_point_set:
        x = point[0]-min_x
        y = point[1]-min_y
        field[x, y] = num_to_rgb(image[point]/40)
    
    img = toimage(field)
    return curr_point_set, img


def get_pixels_ts(folder, field, image_length, image_width):
    field_ts_dict = {}
    for filename in os.listdir(folder):
        if 'RAS' in filename:
            dates = []
            with open(os.path.join(folder, filename[:-4]+'.RHD'), 'r') as f: 
                lines = f.readlines()
                for i in range(5,len(lines)):
                    date = lines[i][6:10] + '-' + lines[i][11:14] + '-' + lines[i][14:17]
                    dtime = datetime.strptime(date.replace(' ', ''), '%Y-%m-%d')
                    dates.append(dtime)
            with open(os.path.join(folder, filename), 'r') as f:            
                a = np.fromfile(f, dtype=np.int16)
                for i in range(len(dates)):
                    field_ts = {}
                    for pixel in field:
                        pixal_value = a[((image_width*pixel[0])+pixel[1])+(i*image_length*image_width)]
                        if len(str(pixal_value)) < 3:
                            field_ts['(' + str(pixel[0]) + ',' + str(pixel[1]) + ')'] = 0
                        elif pixal_value < 0:
                            field_ts['(' + str(pixel[0]) + ',' + str(pixel[1]) + ')'] = pixal_value
                        else:
                            field_ts['(' + str(pixel[0]) + ',' + str(pixel[1]) + ')'] = float(str(pixal_value)[:-2])/40.0
                    field_ts_dict[dates[i]] = field_ts
                print('Done ' + filename)
                
    single_field_LAI_ts_df = pd.DataFrame.from_dict(field_ts_dict).T
    single_field_LAI_ts_df.index = pd.to_datetime(single_field_LAI_ts_df.index)
    single_field_LAI_ts_df = single_field_LAI_ts_df.sort_index()
    single_field_LAI_ts_df = single_field_LAI_ts_df.mask(single_field_LAI_ts_df < 0)
    single_field_LAI_ts_df = single_field_LAI_ts_df.reset_index()
    single_field_LAI_ts_df = single_field_LAI_ts_df.rename(columns={"index": "date"})

    return single_field_LAI_ts_df

def find_gaps_in_df(df):
    is_nan = df.isna()
    n_groups = is_nan.ne(is_nan.shift()).cumsum()
    gap_list = df[is_nan].groupby(n_groups).aggregate(
        lambda x: (
            x.index[0] + pd.DateOffset(days=-2),
            x.index[-1] + pd.DateOffset(days=+2)
        )
    )
    
    return gap_list
