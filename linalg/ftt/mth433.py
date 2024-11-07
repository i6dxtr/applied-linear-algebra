import numpy as np
from PIL import Image
from scipy.signal import convolve

def replicate_pad(arr,prow,pcol):
    """Replicate pads an array
    
    Parameters
    ----------
    arr : 2D array
    prow , pcol : int
    
    Returns
    -------
    2D array of size  (m + prow) x (n + pcol) where m,n = arr.shape
    
    """
    m,n = arr.shape
    B = np.zeros((m + prow,n + pcol))
    left = int(pcol/2)
    top = int(prow/2)
    B[top : m + top,left : n + left] = arr
    B[0:top,left: n + left] = arr[0:1,:]*np.ones((top,n))
    B[top + m:,left : left + n] =  arr[-1:,:]*np.ones((prow - top  ,n))
    B[: , 0:left] = B[:,left:left+1]*np.ones((m+prow,left))
    B[:,left + n:] = B[:,left + n - 1:left+ n]*np.ones((m+prow,pcol-left))
    
    return B

def finish_im(arr):
    """Cuts off the array at 0 and 255, changes the type to uint8 and then
       changes the array to a PIL Image
       
       Parameters
       ----------
       arr : ndarray
       
       Returns
       -------
       PIL Image
    """
    arr = np.minimum(arr,255)
    arr = np.maximum(arr,0)
    return Image.fromarray(np.uint8(arr))

def fast_filter(im , weight):
    """Replicate pads im and then convolves it with weight using scipy.signal.convolve

    Parameters
    ----------
    im : PIL Image
    weight : 2D array

    Returns
    -------
    PIL Image
    
    """
    p,q = weight.shape
    im_array = np.asarray(im,dtype=np.float64)
    m,n,_ = im_array.shape
    pad = np.zeros((m+p-1,n+q-1,3))
    for i in range(3):
        pad[:,:,i] = replicate_pad(im_array[:,:,i],p-1,q-1)
        im_array[:,:,i] = convolve(pad[:,:,i] , weight,mode='valid')

    return finish_im(im_array)