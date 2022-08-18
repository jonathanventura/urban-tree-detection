import tensorflow as tf

def preprocess_RGBN(images):
    R = images[...,0:1]
    N = images[...,3:4]
    ndvi = tf.math.divide_no_nan((N-R),(N+R))
    ndvi *= 127.5
    
    bgr = tf.keras.applications.vgg16.preprocess_input(images[:,:,:,:3])
    
    nir = (images[:,:,:,3:4]-127.5)
    
    images_out = tf.concat([bgr,nir,ndvi],axis=-1)

    return images_out

def preprocess_RGB(images):
    bgr = tf.keras.applications.vgg16.preprocess_input(images[:,:,:,:3])
    
    return bgr

