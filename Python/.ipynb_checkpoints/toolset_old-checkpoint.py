#DEPENDENCIES

#pip install probreg matplotlib scipy opencv-python-headless scikit-image tensorflow shapely pyometiff pyarrow fastparquet opencv-contrib-python openslide-python openslide-bin


import cv2
from matplotlib import pyplot as plt
import numpy
from PIL import Image
import PIL
import pandas as pd
import shapely 
import math
from skimage.morphology import square, erosion, dilation
from skimage.measure import label, regionprops
import tifffile as tf
import sys
from scipy.spatial import distance
from pyometiff import OMETIFFReader
import itertools
import random
#from openslide import OpenSlide
from IPython.display import clear_output


#from probreg import bcpd
#import probreg

def save_image(image,path,view=False):
    iamge=Image.fromarray(image)
    iamge.save(path+".png")

def open_image(filePath,subRes=0,view=False):
    """
    What Do: Open (*and View) Images

    Args:
        filePath: path to image file

    Returns:
        Image (Numpy Array)
    """
    ext=filePath.split(".")
  
    img=None

    if ext[-2]=="ome" and ext[-1]=="tif":
        img,sf=load_ome_tif(filePath)
    elif ext[-1] in ["png","tif"]:
        img=Image.open(filePath)
    #elif ext[-1] in ["svs"]:
    #    slide=OpenSlide(filePath)
    #    dims=slide.level_dimensions[subRes]
    #    img = slide.read_region((0, 0), subRes, dims) 
    
    if isinstance(img, numpy.ndarray):
        None
    else:
        img=numpy.array(img)

    if view:
        v=img
        if v.shape[1] > 1000:
            v,sf=pixelate(img,newWidth=1000)

        #clear_output(wait=True)
        plt.figure()
        plt.title(filePath.split("/")[-1])
        plt.imshow(v,extent=[0, v.shape[1]*sf, 0, v.shape[0]*sf])
        plt.pause(0.001)

    return numpy.array(img)

def displayImage(img,title=False):
    """
    What Do: Displays an image

    Args:
        img: An image in the form of a Numpy Array

    Returns:
        None (Displays Image)
    """
    plt.figure()
    if img.shape[1] > 1000:
        img,sf=pixelate(img,newWidth=1000)
        plt.imshow(img,extent=[0, img.shape[0]*sf, 0, img.shape[1]*sf])
    else:
        plt.imshow(img,extent=[0, img.shape[0], 0, img.shape[1]])
    if title!=False:
        plt.title(title)
    plt.pause(0.001)
    
def write_ome_tif(filename, image, subresolutions = 7):
    """
    What Do: Create an OME TIFF from numpy array

    Args:
        filename: Name of OME.tiff out (or path without .ome.tiff at the end)
        image: Numpy Array of Image
        subresulutions: Number of downscaled iamges

    Returns:
        None (Saves image to filename)
    """
    
    #Specify Metadata
    channel_names="XYCZT"
    photometric_interp='rgb'
    metadata={
        'DimOrder': "XYCZT",
        'SizeX': image.shape[1], 
        'SizeY': image.shape[0], 
        'SizeC': image.shape[2], 
        'SizeZ': 1,
        'SizeT': 1, 
        'PhysicalSizeX': 1,
        'PhysicalSizeY': 1,
        'TotalSeries': 7
    }
    subresolutions = subresolutions

    #Begin Writing File
    fn = filename + ".ome.tif"
    with tf.TiffWriter(fn,  bigtiff=True) as tif:
        pixelsize = metadata['PhysicalSizeX']

        #Specify Options
        options = dict(
            photometric=photometric_interp,
            tile=(1024, 1024),
            dtype=image.dtype,
            compression='jpeg2000',
        )

        #Begin Writing Pyramid Level 0
        print("Writing pyramid level 0")
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4/pixelsize,1e4/pixelsize),
            metadata=metadata,
            **options
        )

        #Write other pyramid levels (each one 1/2 the resolution)
        scale = 1
        for i in range(subresolutions):
            scale /= 2
            
            downsample = img_resize(image,scale)
            print("Writing pyramid level {}".format(i+1))
            tif.write(
                downsample,
                subfiletype=1,
                resolution=(1e4/scale / pixelsize, 1e4/scale / pixelsize),
                metadata=metadata,
                **options
            )

def load_ome_tif(imPath,micron=False):
    """
    What Do: Loads an OME TIFF Image

    Args:
        xenoPath: Path to OME Tif
        micron: Convert Xenium Image to micron scale? (True/False)

    Returns:
        [image as np array, image metadata]
    """
    reader = OMETIFFReader(fpath=imPath)
    image, metadata, xml_metadata = reader.read()
    if micron ==True:
        image=Image.fromarray(image)
        imageSize=image.size
        image=image.resize([int(imageSize[0]*metadata["PhysicalSizeX"]),int(imageSize[1]*metadata["PhysicalSizeY"])])
   
    return image,metadata

def max_image_dimensions(images):
    """
    What Do: Find biggest image+

    Args:
        images: list of images

    Returns:
        max height, max width
    """
    max_height = 0
    max_width = 0

    for img in images:
        height, width = img.shape[:2]
        if height > max_height:
            max_height = height
        if width > max_width:
            max_width = width

    return max_height, max_width

def img_resize(img,scale_factor):
    """
    What Do: Resize an Image by a Scale Factor

    Args:
        img: Numpy Array of Image
        scale_factor: resize scale factor 

    Returns:
        resized image
    """
    #multipy width and height by scale factor
    width = int(numpy.floor(img.shape[1] * scale_factor))
    height = int(numpy.floor(img.shape[0] * scale_factor))
    #Return new scaled image
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)


def pixelate(im, newWidth=600,show=False):
    """
    What Do: Resize an Image to a given width 

    Args:
        im: Numpy Array of Image
        newWidth: New image width to scale to 
        show: Display image (yes/no)
        
    Returns:
        [resized image, scaleFactor]
    """
    #Load numpy image to PIL image
    im=Image.fromarray(im)
    #Calculate new dims
    pixelation_amount = newWidth/im.size[0]
    old_dims = im.size
    new_dims = [int(numpy.round(a*pixelation_amount)) for a in im.size]
    #Calculate scale factor and resize
    scaleFactor=1/pixelation_amount
    im=im.resize(new_dims)
    im=numpy.array(im)
    #Show image 
    if show==True:
        print("Old Dimensions: "+str(old_dims))
        print("New Dimensions: "+str(new_dims))
        print("Scale Factor: "+str(scaleFactor))
        displayImage(im)

    #Return image and scale factor
    return im, scaleFactor
    

def grayscale(im,show=False):
    """
    What Do: Convert Image to Grayscale

    Args:
        im: Numpy Array of Image
        show: display image (yes/no)

    Returns:
        grayscale image
    """
    #Convert to PIL image and convert to grayscale
    im=Image.fromarray(im)
    im = im.convert('L')
    if show==True:
        plt.figure()
        plt.imshow(im)
    #Return numpy array
    im=numpy.array(im)
    return im

def BW(im,show=False):
    """
    What Do: Convert image to black/white (false/true)

    Args:
        im: Numpy Array of Image
        show: Display output? (bool)
    Returns:
        black and white image
    """
    #Convert to B/W PIL Image
    im=Image.fromarray(im)
    im = im.convert('1')
    
    if show==True:
        plt.figure()
        plt.imshow(im)
    #Return numpy array
    im=numpy.array(im)
    return im

def invertColors(im,show=False):
    """
    What Do: Invert image colors 

    Args:
        im: Numpy Array of Image
        show: Display output? (bool)
    Returns:
        inverted image
    """
    #Convert to PIL Image
    im=Image.fromarray(im)
    #Try to Invert colors(will fail if wrong file type)
    try:
        im = PIL.ImageOps.invert(im)
    except:
        print("Error inverting colors")
        
    if show==True:
        plt.figure()
        plt.imshow(im)
    #Return numpy array 
    im=numpy.array(im)
    return im

def blur(im,blr=1,show=False):
    """
    What Do: Apply gaussian blur to image

    Args:
        im: Numpy Array of Image
        blr: radius of blur (odd integar)
        show: Display output? (bool)
    Returns:
        gaussian blurred image
    """
    im=cv2.GaussianBlur(im,(blr,blr),0)
    if show==True:
        plt.figure()
        plt.imshow(im)
    return im

def threshHold(im,threshhold=-1,show=False):
    """
    What Do: Convert pixels above threshold to 255 and below threshold to 0

    Args:
        im: Numpy Array of Image
        threshold: intensity cutoff (if -1 it will choose a cutoff value)
        show: Display output? (bool)
    Returns:
        thresholded image
    """
    if threshhold==-1:
        #Use an adaptive threshold
        ret, im = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        #Set all pixels >threshold to 255
        ret, im = cv2.threshold(im,threshhold,255,cv2.THRESH_BINARY)
    if show==True:
        plt.figure()
        plt.imshow(im)
    return im

def outLine(im,show=False):
    """
    What Do: Outline IMage

    Args:
        im: Numpy Array of Image
        show: Display output? (bool)
    Returns:
        outlined image
    """
    
    im = cv2.Canny(im,100,200)
    if show==True:
        plt.figure()
        plt.imshow(im)
    return im

def euroOpen(im,h=5,v=5,show=False):
    """
    What Do: Removes dark pixels from a white background (good for filtering out background junk in H&E Stain)

    Args:
        im: Numpy Array of Image
        h: size of kernel x
        v: size of kernel y 
        show: Display output? (bool)
    Returns:
        morphology close
    """
    #Create Kernel
    kernel = numpy.ones((h,v),numpy.uint8)
    #Apply morphology open
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    if show==True:
        plt.figure()
        plt.imshow(im)
    #Return Image
    return im

def euroClose(im,h=5,v=5,show=False):
    """
    What Do: Removes white pixels from dark background 

    Args:
        im: Numpy Array of Image
        h: size of kernel x
        v: size of kernel y 
        show: Display output? (bool)
    Returns:
        morphology close
    """
    #Create Kernel
    kernel = numpy.ones((h,v),numpy.uint8)
    #Apply morphology Close
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    if show==True:
        plt.figure()
        plt.imshow(im)
    #Return image
    return im

def blackToWhite(im,cutoff=10,show=False):
    """
    What Do: Turns black pixels to white

    Args:
        im: Numpy Array of Image
        cutoff: Pixels lower than cutoff will be turned white
        show: Display output? (bool)
    Returns:
        image where black pixels are white
    """
    #RGB < cutoff = [255,255,255]
    im[numpy.where((im<[cutoff,cutoff,cutoff]).all(axis=2))] = [255,255,255]
    
    if show==True:
        plt.figure()
        plt.imshow(im)
    return im

def whiteToBlack(im,cutoff=200,show=False):
    """
    What Do: Turns white pixels to black

    Args:
        im: Numpy Array of Image
        cutoff: Pixels lower than cutoff will be turned white
        show: Display output? (bool)
    Returns:
        image where black pixels are white
    """
    #Convert colors to hue, lightness, saturation
    HLS = cv2.cvtColor(im,cv2.COLOR_BGR2HLS)
    #Turn pixels where lightness > cutoff to [0,0,0]
    im[numpy.where(HLS[:,:,1]>cutoff)] = [0,0,0]
    if show==True:
        plt.figure()
        plt.imshow(im)
    return im

def siftDetect(im1,im2,scale1=1,scale2=1,numLandmarks=50000):
    """
    What Do: Detect matching landmarks with SIFT

    Args:
        im1: Numpy Array of Image to Transform
        im2: Numpy Array of Image to Fit to
        scale1: Scale factor of image(if downscaled)
        numLandmarks: Number of ladnmarks to detect
    Returns:
        [homographyMatrix, matching points]
    """
    #Create a SIFT detector
    sift = cv2.SIFT_create(numLandmarks) 

    #Compute keypoints with SIFT
    keypoints1,descriptors1 = sift.detectAndCompute(im1,None)
    keypoints2,descriptors2 = sift.detectAndCompute(im2,None)

    #Match Keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2,k=2)
    
    #Find good points
    good = []
    for m,n in matches:
     if m.distance < 0.75*n.distance:
         good.append([m])

    #Create list of good/sorted matches
    matches = sorted(good,key=lambda x: x[0].distance,reverse=False)
    numGoodMatches = int(len(matches)*0.1)
    matches = matches[:numGoodMatches]

    #Get matching points into 2 lists
    points1 = numpy.zeros((len(matches),2), dtype=numpy.float32)
    points2 = numpy.zeros((len(matches),2), dtype=numpy.float32)
    for i, match in enumerate(matches):
        points1[i,:] = keypoints1[match[0].queryIdx].pt
        points2[i,:] = keypoints2[match[0].trainIdx].pt

    #Find the Homography for Perspective Transform
    h, mask = cv2.findHomography(points1*scale1,points2*scale2,cv2.RANSAC)

    #Create an array of (scaled) matching points
    matchingPoints=[points1*scale1,points2*scale2]
    
    #Return homography matrix and matching points
    return h,matchingPoints

def orbDetect(im1,im2,scale1,scale2,affine=False,numLandmarks=5000,matchP=0.1,show=False):
    """
    What Do: Detect matching landmarks with ORB detect

    Args:
        im1: Numpy Array of Image to Transform
        im2: Numpy Array of Image to Fit to
        scale1: Scale factor of image(if downscaled)
        numLandmarks: Number of ladnmarks to detect
        show: display matches 
    Returns:
        [homographyMatrix, matching points]
    """
    #Create Orb
    orb = cv2.ORB_create(numLandmarks)

    #Detect and Compute Key Points
    keypoints1, descriptors1 = orb.detectAndCompute(im1,None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2,None)

    #Match Points
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1,descriptors2,None)
    
    matches = sorted(matches,key=lambda x: x.distance,reverse=False)

    #Use top 10% Good Matches
    numGoodMatches = int(len(matches)*matchP)
    matches = matches[:numGoodMatches]
    #Print number of matches 
    print("Matches:"+str(len(matches)))
    
    #Create Lists of Matching Points
    points1 = numpy.zeros((len(matches),2), dtype=numpy.float32)
    points2 = numpy.zeros((len(matches),2), dtype=numpy.float32)
    for i, match in enumerate(matches):
        points1[i,:] = keypoints1[match.queryIdx].pt
        points2[i,:] = keypoints2[match.trainIdx].pt

    #find Homography
    if affine:
        matrix, inliers = cv2.estimateAffinePartial2D(
        points1 * scale1,
        points2 * scale2,
        method=cv2.RANSAC
    )
    else:
        matrix, mask = cv2.findHomography(points1*scale1,points2*scale2,cv2.RANSAC)

    #Warp Image to be shape w/ homography
    height, width = im1.shape
    
    if show==True:
        im_matches = cv2.drawMatches(im1,keypoints1,im2,keypoints2,matches,None)
        plt.figure()
        plt.imshow(im_matches);
        plt.show()

    #Return homography matrix and scaled matching points
    matchingPoints=[points1*scale1,points2*scale2]
    return matrix,matchingPoints

def perspectiveTransform(im1,im2,h,show=False):
    """
    What Do: Apply a perspective transform

    Args:
        im1: Numpy Array of Image to Transform
        im2: Numpy Array of Image to Fit to
        h: Homography matrix for perspective transform
        show: display matches 
    Returns:
        trans: Transformed image
    """
    #Apply transofrm to im1
    trans = cv2.warpPerspective(im1 ,h,(int(im2.shape[1]),int(im2.shape[0])))
    if show==True:
        plt.figure(figsize=[40,10])
        plt.imshow(trans);plt.imshow(im2,alpha=0.4);
        plt.show();
    #Return transformed image
    return trans

def affineTransform(im1,im2,a,show=False):
    """
    What Do: Apply an Affine transform

    Args:
        im1: Numpy Array of Image to Transform
        im2: Numpy Array of Image to Fit to
        h: Homography matrix for perspective transform
        show: display matches 
    Returns:
        trans: Transformed image
    """
    #Make a threshold image to display easier
    ret, thresh = cv2.threshold(im2,20,255,cv2.THRESH_BINARY)
    
    #Apply transofrm to im1
    trans = cv2.warpAffine(im1 ,a,(int(thresh.shape[1]),int(thresh.shape[0])))
    if show==True:
        plt.figure(figsize=[40,10])
        plt.imshow(trans);plt.imshow(im2,alpha=0.4);
        plt.show();
    #Return transformed image
    return trans

def pp1(image):
    """
    What Do: Pre-Processing workflow1 (pp1)

    Args:
        image: Image to PP
    Returns:
        [Image scale factor,pre-processed image]
    """
    #Scale Down
    pixel,scale=pixelate(image,1080)
    #Grayscale
    gray=grayscale(pixel)
    #Filter H&E Background noise
    close=euroClose(gray)
    open=euroOpen(close)
    #Return scale factor and PPed image
    return scale,open

def xeniumMask(im,show=False):
    """
    What Do: Create a mask of xenium sample

    Args:
        im: Numpy Array of Image 
        show: display mask 
    Returns:
        Image
    """
    
    oldSize=im.shape[0],im.shape[1]
    im=grayscale(im)
    #Apply threshold 
    im=threshHold(im,threshhold=20)
    #Downscale, Remove background points, Upscale
    
    im=pixelate(im,newWidth=1000)[0]
    im=euroOpen(im)
    im=Image.fromarray(im).resize(oldSize)
    #Convert to numpy array and return im
    im=numpy.array(im)
    if show==True:
        plt.figure()
        plt.imshow(im)
        plt.show();

    return im

def heMask(im,show=False):
    """
    What Do: Create a mask of H&E Sample 

    Args:
        im: Numpy Array of Image 
        show: display image
    Returns:
        Image
    """
    def workflow1(im):
        oldSize=im.shape[0],im.shape[1]
        im=blackToWhite(im)
        im=grayscale(im)
        im=invertColors(im)
        im=euroOpen(im)
        im=threshHold(im,threshhold=20)
        im=pixelate(im,newWidth=1000)[0]
        im=euroOpen(im)
        im=Image.fromarray(im).resize(oldSize)
        im=numpy.array(im)
        return im
    im=workflow1(im)
    if show==True:
        plt.figure()
        plt.imshow(im)
        plt.show();
    return im

def filterHEBackground(og,show=False):
    """
    What Do: filter out H&E Background

    Args:
        og: Numpy Array of Image 
        show: display image
    Returns:
        Image
    """
    def workflow1(im):
        im=whiteToBlack(im)
        im=grayscale(im)
        im=threshHold(im,threshhold=20)
        oldSize=Image.fromarray(im).size
        im=pixelate(im,newWidth=1000)[0]
        im=euroOpen(im,h=5,v=5)
        im=Image.fromarray(im).resize(oldSize)
        im=numpy.array(im)
        im=threshHold(im,threshhold=1)
        kernel = numpy.ones((61, 61), numpy.uint8) 
        im = cv2.dilate(im, kernel, iterations=1) 
        im=cv2.blur(im,(80,80))
        return im

    #create a mask of H&E
    mask=workflow1(og)
    #Normalize mask to 0-1
    im=Image.fromarray(mask)
    im = im.convert('RGB')
    im=numpy.array(im)/255
    #Multiply mask with image (returns places of the image that align with the mask)
    im = numpy.uint8(og*im)
    
    
    if show==True:
        plt.figure(figsize=[40,40])
        plt.subplot(121);plt.imshow(mask)
        plt.subplot(122);plt.imshow(im)
        plt.show();
        

    return im

def transparentBg(im,cutoff=10):
    """
    What Do: Turns black pixels transparent

    Args:
        img: Numpy Array of Image
        cutoff: Cutoff to consider a pixel black/background
    Returns:
        Image with a transparent background
    """
    img=im
    mask = numpy.where((img<cutoff).all(axis=2), 0, 255)
    
    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
    result[:, :, 3] = mask
    return result



def pad_images(images,show=False):
    dims= max_image_dimensions(images)
    for i in range(len(images)):
        images[i] = pad_image(images[i],dims,show=show)
    return images
    

# Padding helper
def pad_image(img, dims,val=255, show=False):
    """
    Pad image to given dimensions.

    Args:
        img: Numpy Array of the image (H×W or H×W×C)
        dims: (height, width) target dimensions
        show: if True, display the padded image

    Returns:
        Padded image as a NumPy array
    """
    target_h, target_w = dims
    h, w = img.shape[:2]

    # Calculate required padding on each side
    pad_top = max((target_h - h) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)

    # Pad with black (0) pixels — handles grayscale or RGB automatically
    padded_img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=val
    )

    if show:
        plt.imshow(
            padded_img if padded_img.ndim == 3 else padded_img,
            cmap='gray' if padded_img.ndim == 2 else None
        )
        plt.title(f"Padded to {dims}")
        plt.axis('off')
        plt.show()

    return padded_img


def difference(im1, im2, show=False):
    """
    What Do: Create a difference image from 2 images

    Args:
        im1: Numpy Array of Image 1
        im2: Numpy Array of Image 2
        show: display mask
    Returns:
        [Difference Image, Normalized Difference Score]
    """

    # Convert images to height=1080 using your pixelate() function
    im1N = pixelate(im1, 1080)[0]
    im2N = pixelate(im2, 1080)[0]

    # Get shapes (height, width)
    h1, w1 = im1N.shape[:2]
    h2, w2 = im2N.shape[:2]

    # Find target size
    target_height = max(h1, h2)
    target_width = max(w1, w2)

    
    # Pad both images if needed
    im1N = pad_to_shape(im1N, target_height, target_width)
    im2N = pad_to_shape(im2N, target_height, target_width)

    # Convert to grayscale
    gray1 = grayscale(im1N)
    gray2 = grayscale(im2N)

    # Ensure both grayscales have the same shape
    assert gray1.shape == gray2.shape, "Grayscale images do not match in size!"

    # Generate difference image
    diff = cv2.absdiff(gray1, gray2)

    # Compute normalized difference score
    size = diff.size
    magnitude = np.sum(diff)
    magScore = magnitude / 255 / size

    # Optional visualization
    if show:
        plt.figure()
        plt.imshow(diff, cmap='gray')
        plt.title(f"Diff Score: {magScore:.4f}")
        plt.axis('off')
        plt.show()

    return diff, magScore

def cutoffHLS(img,show=False,h=0,l=100):
    """
    What Do: Filter image by Hue and Lightness

    Args:
        img: Numpy Array of Image
        h: Hue Cutoff (<)
        l: Lightness Cutoff(>)
        show: display mask
    Returns:
        [Differnece Image, Normalized Difference Score]
    """
    img=numpy.array(img)
    HLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    img[numpy.where(numpy.logical_or(HLS[:,:,0]<(h),HLS[:,:,1]>(l)))] = [0,0,0]
    if show==True:
        plt.figure(figsize=[40,40])
        plt.subplot(121)
        plt.imshow(img)
        plt.show();
    return(img)
    
def getCentroids(mask,euro=1,conect=1):
    """
    What Do: Calculate Cell/Nuclei Centroids from mask (Used after applying HLS cutoff to mask nuclei in H&E)

    Args:
        mask: Mask of cells
        euro: Eurosion amount (helps seperate cells)
        conect: connectivity (im not entirely sure what this one does but it takes a value between 1 and 2?)
    Returns:
        array of centroids like [[x1,y1],[x2,y2]]
    """
    v,h=mask.shape
    #Perform eurosion on mask
    if euro!= 0:
    	mask=erosion(mask, square(euro))
    #Seperate mask into inviduals masks
    individual_mask=label(mask, connectivity=conect)
    prop=regionumpyrops(individual_mask)
    centroids=[]
    #Calculate Centroids
    for cordinates in prop:
        centroid=cordinates.centroid
        if not math.isnan(centroid[0]) and not math.isnan(centroid[1]):
            centroids.append([int(centroid[0]),int(centroid[1])])
    return centroids

def gCentroids(mask):
	v,h=mask.shape
	center_mask=numpy.zeros([v,h])
	mask=erosion(mask, square(1))
	individual_mask=label(mask, connectivity=2)
	prop=regionprops(individual_mask)
	centers=[]
	for cordinates in prop:
		temp_center=cordinates.centroid
		if not math.isnan(temp_center[0]) and not math.isnan(temp_center[1]):
			centers.append([int(temp_center[0]),int(temp_center[1])])
            
	return centers

def calcXenoNucleiCentroids(folderPath):
    """
    What Do: Calculate Nuclei Centroids from nucleus_boundaries.parquet

    Args:
        folderPath: path to the xenium out folder that contains nucleus_boundaries.parquet. Must end in /
    Returns:
        pandas dataframe of nuclei centroids from parquet 
    """
    #Load and Read parquet
    filePath=folderPath+"nucleus_boundaries.parquet"
    nuclei = pd.read_parquet(filePath)
    nuclei_grouped = nuclei.groupby(nuclei["cell_id"])
    #Setup Variables
    group_numbers = []
    nuclei_centroids_x = []
    nuclei_centroids_y = []
    nuclei_ids = []
    
    group_number=1
    for group in nuclei_grouped:
        #Do stuff to calculate centroids (Evan's code ask him or something)    
        group_df = group[1]
        coords = list(zip(group_df["vertex_x"], group_df["vertex_y"]))
        group_polygon = shapely.Polygon(coords)
        centroid = shapely.centroid(group_polygon)
        centroid_x = centroid.x
        centroid_y = centroid.y
        group_numbers.append(group_number)
        nuclei_centroids_x.append(int(centroid_x))
        nuclei_centroids_y.append(int(centroid_y))
        try:
            nuclei_id=group_df["cell_id"][group_number-1]
        except:
            print(group_df)
            print(13*group_number-13)
        nuclei_ids.append(nuclei_id)
        group_number+=len(group_df)
        
    
    nuclei_centroids_df = pd.DataFrame()
    nuclei_centroids_df["x_centroid"] = nuclei_centroids_x
    nuclei_centroids_df["y_centroid"] = nuclei_centroids_y

    return nuclei_centroids_df

def rgb_to_hex(r, g, b):
    """
    What Do: Convert RGB color to hex code

    Args:
        r: Red (0-255)
        g: Green (0-255)
        b: Blue (0-255)
    Returns:
        hex code of color
    """
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def tile(im,s=130,savePath=False,saveType='',sum=15555,show=False):
    """
    What Do: Breaks image into a dictionary of small tiles (labeled by origin)

    Args:
        im: Image to Break Up (numpy array)
        s: Tile size (square length)
        savePath: Specify a path if you want it to save the tiles
        saveType: Specify DAPI or HE for filtering purposes
        sum: Cutoff sum (so we don't have empty tiles)
        show:Set True if you want to display tiles
    Returns:
        Dictionary of Tiles, labeled by origin 
    """
    tiles={}
    for x in range(0,im.shape[0],s):
        for y in range(0,im.shape[1],s):
            tile = im[x:x+s,y:y+s]
            if tile.shape[0] != s or tile.shape[1] != s:
                continue
            if saveType=="DAPI":
                if (numpy.array(tile) < 20).sum()>sum:
                    continue
                tile=cv2.normalize(numpy.array(tile), None, 0, 255,cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            if saveType=="HE":
                if (numpy.array(tile) > 235).sum()>sum:
                    continue
            if show==True:
                plt.axis('off');plt.imshow(tile)
                plt.show()
            tiles[str(x)+"_"+str(y)]=tile
            if savePath != False:
                tile=Image.fromarray(tile)
                tile.save(savePath+saveType+"_"+str(x)+"_"+str(y)+".png")
    return tiles
    
'''
def probPoint(sourcePoints,targetPoints,xOrigin=0,yOrigin=0):
    """
    What Do: Aligns two sets of points using Probreg

    Args:
        sourcePoints: Array of 2D Points (transform source)
        targetPoints: Array of 2D Points (transform target)
        xOrigin: The x-Origin of target points (source points assumed to have 0,0 origin) 
        yOrigin: The y-Origin of target points (source points assumed to have 0,0 origin) 
    Returns:
        [Source Points, Transformed Source Points] 
    """

    #Load Source Point
    source=sourcePoints.copy()
    source[:,0]=sourcePoints[:,1]
    source[:,1]=sourcePoints[:,0]

    #Load Target Points
    target=numpy.array(targetPoints)
    target[:,0]=(target[:,0]-xOrigin)
    target[:,1]=(target[:,1]-yOrigin)    

    #Convert Points to 3D for Probreg Methods
    source=c_2d_to_3d(source)
    target=c_2d_to_3d(target)

    #Use BCPD Registration to trasnform points
    reg = bcpd.registration_bcpd(source, target)
    i=reg._transform(source)

    #Convert Output points into DAPI Coordinate Space
    i[:,0]=i[:,0]+xOrigin
    i[:,1]=i[:,1]+yOrigin
    source[:,0]=source[:,0]+xOrigin
    source[:,1]=source[:,1]+yOrigin
    #Convert Points to 2D Numpy Array
    points2=i[:,0:2]
    points1=source[:,0:2]
    P1 = numpy.array(points1,numpy.int32).reshape(1,-1,2)
    P2 = numpy.array(points2,numpy.int32).reshape(1,-1,2)
    #Return P1(Input points) and P2(Output Points)
    return(P1,P2)
'''


def c_2d_to_3d(points_2d, z_coordinate=0):
    """
    What Do: Convert 2D points to 3D for Probreg

    Args:
        points_2d: Array of 2D Points
        z_coordinate: Z-coordinate for 3D Points(default of 0)
    Returns:
        Array of 3D Points
    """
    points_3d = numpy.empty((len(points_2d), 3), dtype=numpy.float64)
    points_3d[:, :2] = points_2d
    points_3d[:, 2] = z_coordinate
    return points_3d
    
def closest_point(P,Ps):
    """
    What Do: Calculate closest point to P in list of Ps
    
    Args:
        P: A 2D Point
        Ps: List of 2D Points
    Returns:
        [Index of closest point, distance to that point]
    """
    distances = numpy.linalg.norm(Ps-P, axis=1)
    min_index = numpy.argmin(distances)
    #print(f"the closest point is {Ps[min_index]}, at a distance of {distances[min_index]}")
    return Ps[min_index],distances[min_index]

def pointCloudAlign(source,target,xenoCentroids,s=130,tpsPoints=200,pointsPerTile=3,show=False,hsl={
    "H_min":140,
    "H_max":180,
    'S_min':74,
    'S_max':160,
    'L_min':0,
    'L_max':255
    }):

    #Setup Vars
    allPoints1=[]
    allPoints2=[]
    tps_match={}
    he_match={}
    s=130
    
    #Tile Images
    sourceTiles=tile(source,s=s,sum=100000000)
    targetTiles=tile(target,s=s,sum=100000000)

    #Find Good Tiles common to Both Images
    goodTiles=[]
    for key in sourceTiles:
        if key in targetTiles:
            goodTiles.append(key)

    #Loop through tiles
    for origin in goodTiles:
        tile0=targetTiles[origin]
        tile1=sourceTiles[origin]
        
        #Calculate Origin
        xyOrigin=origin.split("_")
        xOrigin=int(xyOrigin[1])
        yOrigin=int(xyOrigin[0])

        #Find DAPI Centroids within tile
        dapiCentroids=xenoCentroids[xenoCentroids['x_centroid'].between(xOrigin, xOrigin+s)]
        dapiCentroids=dapiCentroids[dapiCentroids['y_centroid'].between(yOrigin, yOrigin+s)]
        if (len(dapiCentroids))==0:
            continue
        #Find H&E Centroids 
        mask=tile1.copy()
        HSL = cv2.cvtColor(tile1, cv2.COLOR_BGR2HLS)
        mask[numpy.where(numpy.logical_or(HSL[:, :, 0] < hsl["H_min"], HSL[:, :, 0] >hsl["H_max"]))] = [0, 0, 0]
        mask[numpy.where(numpy.logical_or(HSL[:, :, 2] < hsl["S_min"], HSL[:, :, 2] >hsl["S_max"]))] = [0, 0, 0]
        mask[numpy.where(numpy.logical_or(HSL[:, :, 1] < hsl["L_min"], HSL[:, :, 1] > hsl["L_max"]))] = [0, 0, 0]
        
        mask=numpy.array(Image.fromarray(cv2.cvtColor(mask,cv2.COLOR_HLS2BGR)).convert("L"))
        mask=threshHold(mask)
        #mask=numpy.array(Image.fromarray(tile1[:,:,0]<160).convert('L'))
        centroids=numpy.array(gCentroids(mask))
        dapiCentroids=numpy.array(dapiCentroids)
        #Setup Dict
        new={}
        old={} 
        #Run point clouds

        try:
            Ps=probPoint(centroids,dapiCentroids,xOrigin=xOrigin,yOrigin=yOrigin)
        except:
            print("Point Cloud Failed at Tile:" +str(origin))
            continue

        #Calculate if Old Points or New Points are closer to DAPI                      
        for hePoint in Ps[1][0]:
            new[str(hePoint[0])+"_"+str(hePoint[1])] = closest_point(hePoint,dapiCentroids)[1]
        for hePoint in Ps[0][0]:
            old[str(hePoint[0])+"_"+str(hePoint[1])] = closest_point(hePoint,dapiCentroids)[1]
    
        new=dict(sorted(new.items(), key=lambda x:x[1]))
        old=dict(sorted(old.items(), key=lambda x:x[1]))
    
        newValList=list(new.values())
        valScore=sum(newValList)/len(newValList)
        new = dict(itertools.islice(new.items(), pointsPerTile))
        
        oldValList=list(old.values())
        oldValScore=sum(oldValList)/len(oldValList)
        old = dict(itertools.islice(old.items(), pointsPerTile))
    
        #If old points bette use old, if new better use new
        try:
            if valScore < oldValScore:
                tps_match.update(new)
                allPoints1.extend(Ps[0][0].tolist())
                allPoints2.extend(Ps[1][0].tolist())
            else:
                he_match.update(old)
                allPoints1.extend(Ps[0][0].tolist())
                allPoints2.extend(Ps[0][0].tolist())
        except:
            print("Failed to update DAPI & H&E Matches")
    
    #Pick a random sample of Points
    tps_match=dict(sorted(tps_match.items(), key=lambda item: item[1]))
    goodPoints=list(tps_match.keys())
    print(len(goodPoints))
    goodPoints=random.sample(goodPoints,int(tpsPoints*2/3))

    he_match=dict(sorted(he_match.items(), key=lambda item: item[1]))
    he_match=list(he_match.keys())
    goodPoints.extend( random.sample(he_match,int(tpsPoints/3)))

    #Add Points to List
    P1=[]
    P2=[]
    for p in goodPoints:
        p=p.split("_")
        p=[int(p[0]),int(p[1])]
        #print(allPoints2[index])
        try:
            index=(numpy.where((numpy.array(allPoints2)==p).all(1))[0][0])
        except:
            print("error with point: "+str(p))
            continue
        P1.append(allPoints1[index])
        P2.append(allPoints2[index])
    if show==True:
        print("Graph of TPS Point Locations:")
        plt.imshow(source);plt.scatter(numpy.array(P1)[:,0],numpy.array(P1)[:,1]);plt.scatter(numpy.array(P2)[:,0],numpy.array(P2)[:,1],c="r")
        plt.show()
    matches = []
    for p in range(len(P1)-1):
        p=p+1
        matches.append(cv2.DMatch(p,p,0))
    P1 = numpy.array(P1,numpy.int32).reshape(1,-1,2)
    P2 = numpy.array(P2,numpy.int32).reshape(1,-1,2)
    print("Calculating TPS")
    #Apply TPS from Points
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(P2,P1,matches)
    transNew = tps.warpImage(source)
    return transNew