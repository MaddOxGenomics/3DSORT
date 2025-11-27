from matplotlib import pyplot as plt
from monai.data import MetaTensor
from PIL import Image
import tifffile
import numpy
import numpy as np
import nrrd
from skimage.transform import resize
import torch
import torch.nn.functional as F
import copy
from pprint import pprint
import os
from copy import deepcopy
import xml.etree.ElementTree as ET

#--------------------------
# mTensor Creation Functions
#--------------------------


def mTensorCreate(tensor,meta,path=False):
    if path:
        if "." in path:
            if not meta.get("Name"):
                meta["Name"] = path.split("/")[-1].split(".")[0]
            if not meta.get("Ext"):
                meta["Ext"] = path.split("/")[-1].split(".")[-1]
        if not meta.get("Path"):
            meta["Path"] = path
    
    dimSizes=getDimSizes(meta["DimensionOrder"],tensor)
    
    meta["SizeX"]=dimSizes.get("X", 1)
    meta["SizeY"]=dimSizes.get("Y", 1)
    meta["SizeZ"]=dimSizes.get("Z", 1)
    meta["SizeC"]=dimSizes.get("C", 1)
    meta["SizeT"]=dimSizes.get("T", 1)

    mTensor = MetaTensor(tensor, meta=meta) 
    return mTensor

def omeTiff2mTensor(path,meta={},level=0):
    with tifffile.TiffFile(path) as tif:
        ome_xml = tif.ome_metadata
        
        series = tif.series[0]             # most OME-TIFFs store the image here
        
        print("Number of pyramid levels:", len(series.levels))
        
        for i, lvl in enumerate(series.levels):
            print(f"Level {i} shape:", lvl.shape)
    
        img = tif.series[0].levels[level].asarray()
        metadata={"ome_xml":ome_xml}
        try:
            meta = parseXeniumMetaData(img,ome_xml)|metadata
        except:
            print("Warning: Did not parse ome_xml meta data")
            meta= {"DimensionOrder":"YXC"} | metadata
    mTensor=mTensorCreate(img,meta=meta,path=path)
    return mTensor

def tif2mTensor(tifPath,meta={}):
    zstack = tifffile.imread(tifPath)
    new_meta={
        "DimensionOrder":"ZYXC",
    }
    return mTensorCreate(zstack,meta|new_meta,path=tifPath)
    
def folder2mTensor(folderPath,save=False,meta={},show=False):

    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

    # --- Collect image files ---
    files = sorted([f for f in os.listdir(folderPath)
                    if any(f.lower().endswith(ext) for ext in exts)])
    images=[]
    
    for i, f in enumerate(files):
        file_path = os.path.join(folderPath, f)

        # --- Load + background filter ---
        img = Image.open(file_path)
        img=np.array(img)
        
        if show:
            if i in show:
                plt.imshow(img)
                plt.pause(0.001)

        images.append(img)
    images=padImageStack(images)
    if save:
        tifffile.imwrite(save, np.stack(images))
        print(f"✅ Saved TIFF stack: {save}")

    newMeta={
        "DimensionOrder":"ZYXC"
    }
    meta=newMeta|meta
    mTensor=mTensorCreate(images,meta,path=folderPath)
    return mTensor


def seg2mTensor(segPath,meta={}):
    seg, seg_header = nrrd.read(segPath)
    seg = np.moveaxis(seg, -1, 0)
    seg = np.transpose(seg, (0, 2, 1)) 
    meta.update(seg_header)
    newMeta={
        "DimensionOrder":"ZYXC"
    }
    return mTensorCreate(seg,meta,path=segPath)
    
def image2mTensor(imgPath,meta={}):
    img = Image.open(imgPath)
    imgTensor = numpy.array(img)
    try:
        C=imgTensor.shape[2]
        dimOrder="YXC"
    except:
        C=1
        dimOrder="YX"
    new_meta={
        "DimensionOrder":dimOrder,
    }
    return mTensorCreate(imgTensor, meta|new_meta, path=imgPath)

def saveMetaTensor(mTensor, path):
    obj = {
        "tensor": mTensor.as_tensor(),  # raw PyTorch tensor
        "meta": mTensor.meta            # metadata dictionary
    }
    torch.save(obj, path)

def loadMetaTensor(path):
    obj = torch.load(path)
    tensor = obj["tensor"]
    meta = obj["meta"]
    return MetaTensor(tensor, meta=meta)

#------------------------
# mTensor Helper Functions
#------------------------
def displayMetaTensor2D(mTensor,showMetaData=False, cmap="plasma", showAxis=True,z=0):
    name = mTensor.meta.get("Name", " ")
    img_np = extraxtXYSlice(mTensor,z)
        
    max_size=1080
    scale_factor = 1.0
    h, w = img_np.shape[:2]
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        img_np = numpy.array(scale(mTensor,scale_factor))[:,:,z]
    print(img_np.shape)
     # --- Display ---
    plt.figure(figsize=(6,6))
    plt.imshow(img_np,cmap=cmap)
    
    title=f"{name} | "
    if mTensor.meta["SizeZ"]!=1:
        title+=f'Z({mTensor.meta["SizeZ"]}), '
    title+=f'Y({mTensor.meta["SizeY"]}), X({mTensor.meta["SizeX"]})'
    if mTensor.meta["SizeC"]!=1:
        title+=f', C({mTensor.meta["SizeC"]})'
    if mTensor.meta["SizeT"]!=1:
        title+=f', Z({mTensor.meta["SizeT"]})'
    if mTensor.meta["SizeZ"]!=1:
        title=title+ f' | z={z}'
    plt.title(title)
    
    if not showAxis:
        plt.axis("off")
    plt.show()
    plt.pause(0.001)
              
    if showMetaData:
        pprint(mTensor.meta)


def getDimSizes(dimOrder, arr):
    """
    Given a dimOrder string (e.g. 'CZYX') and an array/tensor,
    return a dict mapping each dimension letter to its size.
    
    arr can be:
        - numpy array
        - torch tensor
    """
    # Get shape tuple
    if hasattr(arr, "shape"):   # works for torch + numpy
        shape = tuple(arr.shape)
    else:
        raise ValueError("arr must be a numpy array or torch tensor")

    if len(dimOrder) != len(shape):
        raise ValueError(
            f"dimOrder '{dimOrder}' has length {len(dimOrder)} "
            f"but array has {len(shape)} dimensions."
        )

    return {dimOrder[i]: shape[i] for i in range(len(dimOrder))}

def createDimSizes(dimOrder, tensor):
    """
    Create a full dimSizes dict from dimOrder and a tensor.
    Missing dims (Z, C, T) are set to 1.

    dimOrder: e.g. "CZYX", "YX", "YXC"
    tensor: torch.Tensor or numpy array
    """

    shape = list(tensor.shape)  # supports torch & numpy

    if len(dimOrder) != len(shape):
        raise ValueError(
            f"dimOrder '{dimOrder}' has length {len(dimOrder)} but tensor has {len(shape)} dims."
        )

    # Map provided dims → sizes
    sizes = {dimOrder[i]: shape[i] for i in range(len(dimOrder))}

    # Ensure all dims exist, missing ones = 1
    fullSizes = {
        "X": sizes.get("X", 1),
        "Y": sizes.get("Y", 1),
        "Z": sizes.get("Z", 1),
        "C": sizes.get("C", 1),
        "T": sizes.get("T", 1),
    }

    return fullSizes

#------------------------
# Image Helper Functions
#------------------------

def padImageStack(images, fill_value=0):
    # Find max height and width
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    padded = []
    for img in images:
        # Determine number of channels
        if img.ndim == 2:  # grayscale
            pad_shape = (max_h, max_w)
        else:              # color image
            pad_shape = (max_h, max_w, img.shape[2])

        result = np.full(pad_shape, fill_value, dtype=img.dtype)
        result[:img.shape[0], :img.shape[1], ...] = img
        padded.append(result)

    return np.stack(padded)


def scale(mTensor, s,meta=False):
    
    try: 
        if meta:
            meta=meta | mTensor.meta
        else:
            meta=mTensor.meta
    except:
        raise ValueError(f"No meta for {mTensor}")
    
    try:
        shape=meta["DimensionOrder"]
       
    except:
        raise ValueError(f"No Shape in metadata: {meta}")
    
    try:
         PixelSize=meta["PixelSize"]
    except:
        PixelSize=1
        print("Warning: No PixelSize Metadata, defaulting to 1")
        
    orig_dtype = mTensor.dtype
    scale=s/PixelSize
    out_slices=[]

    C=mTensor.meta["SizeC"]
    X=mTensor.meta["SizeX"]
    Y=mTensor.meta["SizeY"]
    Z=mTensor.meta["SizeZ"]
    new_h, new_w = int(Y * scale), int(X * scale)
    for z in range(Z):
        slice_=torch.from_numpy(extraxtXYSlice(deepcopy(mTensor),z))
        if C==1:
            t = slice_.half()[None, None]  # (1,1,H,W)
            out = F.interpolate(t, size=(new_h, new_w), mode="area")
            out_slices.append(out[0,0])
        else:
            t = slice_.half().permute(2,0,1)[None]  # (1,C,H,W)
            out = F.interpolate(t, size=(new_h, new_w), mode="area")
            out_slices.append(out[0].permute(1,2,0))
        
    
    out = torch.stack(out_slices)

    if z==0:
        out=out[0]
    mTensor_out = copy.deepcopy(mTensor)
    out = out.to(orig_dtype)
    mTensor_out.set_array(out)
    mTensor_out.meta["PixelSize"]=scale*PixelSize
    mTensor_out.meta['SizeX'] = int(X*scale)
    mTensor_out.meta['SizeY'] = int(Y*scale)
    mTensor_out.meta['SizeC'] = C
    mTensor_out.meta["DimensionOrder"]=shape
    return mTensor_out

def extraxtXYSlice(mTensor,z=0):
    img_np = mTensor.detach().cpu().numpy()
    
    dimOrder=mTensor.meta["DimensionOrder"]
    dimSizes={
        "X":mTensor.meta["SizeX"],
        "Y":mTensor.meta["SizeY"],
        "Z":mTensor.meta["SizeZ"],
        "C":mTensor.meta["SizeC"],
        "T":mTensor.meta["SizeT"]
    }

    targetShape="YX"
    if dimSizes["C"]!=1:
        targetShape="YXC"
    

    axisMap = {dimOrder[i]: i for i in range(len(dimOrder))}

    if dimSizes["Z"] > 1:
        zAxis = axisMap["Z"]
        img_np = np.take(img_np, z, axis=zAxis)

    desiredOrder = []
    for d in ["Y", "X"]:
        desiredOrder.append(axisMap[d] if d in axisMap else None)

    if targetShape == "YXC":
        desiredOrder.append(axisMap["C"])


    desiredOrder = [a for a in desiredOrder if a is not None]

    remainingDims = [d for d in dimOrder if d != "Z"]
    newAxisMap = {remainingDims[i]: i for i in range(len(remainingDims))}
    finalAxes = []
    for d in (["Y", "X"] + (["C"] if targetShape=="YXC" else [])):
        if d in newAxisMap:
            finalAxes.append(newAxisMap[d])

    img_out = np.transpose(img_np, finalAxes)

    return img_out

    
#---------------------
# Other Help Functions
#---------------------

def strip_namespace(tag):
    """Remove {namespace} from tag strings."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def etree_to_dict(elem):
    """Convert an ElementTree node to a dict (namespace-stripped)."""
    d = {strip_namespace(elem.tag): {}}

    # Process attributes
    for k, v in elem.attrib.items():
        d[strip_namespace(elem.tag)][ k] = v

    # Process children
    children = list(elem)
    if children:
        child_dict = {}
        for child in children:
            cdict = etree_to_dict(child)
            tag = list(cdict.keys())[0]
            child_dict.setdefault(tag, []).append(cdict[tag])
        d[strip_namespace(elem.tag)].update(child_dict)

    # Process text
    text = (elem.text or '').strip()
    if text:
        if d[strip_namespace(elem.tag)]:
            d[strip_namespace(elem.tag)]['#text'] = text
        else:
            d[strip_namespace(elem.tag)] = text

    return d

    
def parseXeniumMetaData(img,ome_xml,meta={}):
    root = ET.fromstring(ome_xml)
    metadataOld=etree_to_dict(root)

    pixels = metadataOld['OME']['Image'][0]['Pixels'][0]

    # Sanity check XY scale
    if pixels['PhysicalSizeX'] != pixels['PhysicalSizeY']:
        raise ValueError(f"XY Scaling Error")

    # --- Build base metadata ---
    metadata = {
        'SizeX': int(pixels['SizeX']),
        'SizeY': int(pixels['SizeY']),
        'SizeZ': int(pixels['SizeZ']),
        'SizeC': int(pixels['SizeC']),
        "SizeT": int(pixels['SizeT']),
        'PixelSize': float(pixels['PhysicalSizeX']),
        'PixelSizeUnit': pixels['PhysicalSizeXUnit'],
        "OldOMExmlData":metadataOld
    }

    shape = img.shape
    dims = []

    # Check each standard OME axis
    if metadata["SizeZ"] > 1: 
        dims.append("Z")
    dims.extend(["X", "Y"])
    if metadata["SizeC"] > 1:
        dims.append("C")
    if metadata["SizeT"] > 1:
        dims.append("T")

    # Join to string
    metadata["DimensionOrder"] = "".join(dims)
    print(metadata["DimensionOrder"])
    return  meta | metadata


#def removeBackground(img):
    