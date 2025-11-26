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

def mTensorCreate(tensor,meta,path=False):
    if path:
        if "." in path:
            if not meta.get("Name"):
                meta["Name"] = path.split("/")[-1].split(".")[0]
            if not meta.get("Ext"):
                meta["Ext"] = path.split("/")[-1].split(".")[-1]
        if not meta.get("Path"):
            meta["Path"] = path
    mTensor = MetaTensor(tensor, meta=meta) 
    return mTensor
    

def seg2mTensor(segPath,meta={}):
    seg, seg_header = nrrd.read(segPath)
    seg = np.moveaxis(seg, -1, 0)
    seg = np.transpose(seg, (0, 2, 1)) 
    meta.update(seg_header)
    return mTensorCreate(seg,meta,path=segPath)

def tif2mTensor(tifPath,meta={}):
    zstack = tifffile.imread(tifPath)
    return mTensorCreate(zstack,meta,path=tifPath)
    
def image2mTensor(imgPath,meta={}):
    img = Image.open(imgPath)
    imgTensor = numpy.array(img)
    new_meta={
        "DimensionOrder":"YXC",
        'SizeX': imgTensor.shape[1],
        'SizeY': imgTensor.shape[0],
        'SizeZ': 1,
        'SizeC': imgTensor.shape[2],
        "SizeT": 1,
        'PixelSize': 1
    }
    return mTensorCreate(imgTensor,meta | new_meta,path=imgPath)


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

    # ---------------------------------------------------------
    # (H, W)
    # ---------------------------------------------------------
    if shape == "YX":
        C=1
        H, W = mTensor.shape
        new_h, new_w = int(Y * scale), int(X * scale)
        t = mTensor.half()[None, None]  # (1,1,H,W)
        out = F.interpolate(t, size=(new_h, new_w), mode="area")
        out= out[0,0]

    # ---------------------------------------------------------
    # (H, W, C)
    # ---------------------------------------------------------
    elif shape == "YXC":
        Y, X, C = mTensor.shape
        new_h, new_w = int(Y * scale), int(X * scale)
        t = mTensor.half().permute(2,0,1)[None]  # (1,C,Y,X)
        out = F.interpolate(t, size=(new_h, new_w), mode="area")
        out=out[0].permute(1,2,0)

    elif shape == "CYX":
        C, Y, X = mTensor.shape
        new_h, new_w = int(Y * scale), int(X * scale)
        t = mTensor.half()[None]  # (1,C,Y,X)
        out = F.interpolate(t, size=(new_h, new_w), mode="area")
        out=out[0]

    elif shape == "CXY":
        C, X, Y = mTensor.shape
        new_h, new_w = int(Y * scale), int(X * scale)
        t = mTensor.half().permute(0,2,1)[None]  # (1,C,Y,X)
        out = F.interpolate(t, size=(new_h, new_w), mode="area")
        out=out[0].permute(0,2,1)


    # ---------------------------------------------------------
    # (Z, H, W)
    # ---------------------------------------------------------
    elif shape == "ZYX":
        C=1
        Z, Y, X = mTensor.shape
        new_h, new_w = int(Y * scale), int(X * scale)

        out_slices = []
        for z in range(Z):
            slice_ = mTensor[z]  # (H, W)
            t = slice_.half()[None, None]  # (1,1,H,W)
            out = F.interpolate(t, size=(new_h, new_w), mode="area")
            out_slices.append(out[0,0])
    
        out=torch.stack(out_slices)

    # ---------------------------------------------------------
    # (Z, H, W, C)
    # ---------------------------------------------------------
    elif shape == "ZYXC":
        Z, Y, X, C = mTensor.shape
        new_h, new_w = int(Y * scale), int(X * scale)

        out_slices = []
        for z in range(Z):
            t = mTensor[z].half().permute(2,0,1)[None]  # (1,C,H,W)
            out = F.interpolate(t, size=(new_h, new_w), mode="area")
            out_slices.append(out[0].permute(1,2,0))
        out = torch.stack(out_slices)

    else:
        raise ValueError(f"Unsupported shape descriptor: {shape}")


    mTensor_out = copy.deepcopy(mTensor)
    out = out.to(orig_dtype)
    mTensor_out.set_array(out)
    mTensor_out.meta["PixelSize"]=scale*PixelSize
    mTensor_out.meta['SizeX'] = int(X*scale)
    mTensor_out.meta['SizeY'] = int(Y*scale)
    mTensor_out.meta['SizeC'] = C
    mTensor_out.meta["DimensionOrder"]=shape
    return mTensor_out

def displayMetaTensor(mTensor,showMetaData=False, showAxis=True,max_size=1080):
    name = mTensor.meta.get("Name", " ")
    
    img_np = mTensor.detach().cpu().numpy()
    scale_factor = 1.0
    h, w = img_np.shape[:2]
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        img_np = scale(img_np,scale_factor)

     # --- Display ---
    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.title(f"{name} | shape: {tuple(mTensor.shape)} | scale={scale_factor:.2f}")
    if not showAxis:
        plt.axis("off")
    plt.show()
    plt.pause(0.001)
              
    if showMetaData:
        print(mTensor.meta)


#def removeBackground(img):
    