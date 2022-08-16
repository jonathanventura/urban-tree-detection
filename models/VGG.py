import numpy as np 
from tensorflow.keras import Model, layers, initializers, losses
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
import tensorflow as tf

class BaseConv(layers.Layer):
    def __init__(self, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv,self).__init__()
        self.use_bn = use_bn
        self.conv = layers.Conv2D(out_channels, kernel, strides=stride, padding='same',
                           kernel_initializer=initializers.RandomNormal(stddev=0.01))
        self.bn = layers.BatchNormalization()
        if activation is None:
            self.activation = layers.Activation(activation)
        else:
            self.activation = None
    
    def call(self,inputs):
        x = self.conv(inputs)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class VGG(Model):
    def __init__(self,output_layers):
        """ Initializes a custom VGG model.
            Arguments:
                output_layers: list of layers to output (0 for first layer, 1 for second layer, etc.)
        """
        super(VGG,self).__init__()
        self.output_layers = output_layers
        self.pool = layers.MaxPooling2D(2, 2)
        self.conv1_1 = BaseConv(64, 3, 1, activation='relu', use_bn=True)
        self.conv1_2 = BaseConv(64, 3, 1, activation='relu', use_bn=True)
        self.conv2_1 = BaseConv(128, 3, 1, activation='relu', use_bn=True)
        self.conv2_2 = BaseConv(128, 3, 1, activation='relu', use_bn=True)
        self.conv3_1 = BaseConv(256, 3, 1, activation='relu', use_bn=True)
        self.conv3_2 = BaseConv(256, 3, 1, activation='relu', use_bn=True)
        self.conv3_3 = BaseConv(256, 3, 1, activation='relu', use_bn=True)
        self.conv4_1 = BaseConv(512, 3, 1, activation='relu', use_bn=True)
        self.conv4_2 = BaseConv(512, 3, 1, activation='relu', use_bn=True)
        self.conv4_3 = BaseConv(512, 3, 1, activation='relu', use_bn=True)
        self.conv5_1 = BaseConv(512, 3, 1, activation='relu', use_bn=True)
        self.conv5_2 = BaseConv(512, 3, 1, activation='relu', use_bn=True)
        self.conv5_3 = BaseConv(512, 3, 1, activation='relu', use_bn=True)

    def load_pretrained_vgg(self,input_shape):
        """ Load weights from the pre-trained VGG16 model. 
            This can only be called after the model has been built.
            Arguments:
                input_shape: input shape [H,W,C] (without the batch dimension)
        """
        channels_in = input_shape[2]

        # get pre-trained VGG for BGR input
        vgg_bgr = VGG16(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))
        
        # get weights in initial layer
        w_bgr,b_bgr = vgg_bgr.layers[1].get_weights()
        
        # make new VGG with correct input shape
        vgg = VGG16(include_top=False, input_shape=input_shape, weights=None)

        # copy in pre-trained weights to first layer
        w,b = vgg.layers[1].get_weights()
        w[:,:,:3,:] = w_bgr
        b = b_bgr
        vgg.layers[1].set_weights([w,b])
        
        # copy in pre-trained weights to remaining layers
        for i in range(2,len(vgg.layers)):
            vgg.layers[i].set_weights(vgg_bgr.layers[i].get_weights())

        # copy weights to our layers
        def set_weights(layer,layer_in):
            weights = layer.get_weights()
            weights_in = layer_in.get_weights()
            weights[0] = weights_in[0]
            weights[1] = weights_in[1]
            layer.set_weights(weights)
        
        set_weights(self.conv1_1,vgg.layers[1])
        set_weights(self.conv1_2,vgg.layers[2])
        set_weights(self.conv2_1,vgg.layers[4])
        set_weights(self.conv2_2,vgg.layers[5])
        set_weights(self.conv3_1,vgg.layers[7])
        set_weights(self.conv3_2,vgg.layers[8])
        set_weights(self.conv3_3,vgg.layers[9])
        set_weights(self.conv4_1,vgg.layers[11])
        set_weights(self.conv4_2,vgg.layers[12])
        set_weights(self.conv4_3,vgg.layers[13])
        set_weights(self.conv5_1,vgg.layers[15])
        set_weights(self.conv5_2,vgg.layers[16])
        set_weights(self.conv5_3,vgg.layers[17])

    def call(self,inputs):
        x = inputs
        l = []

        x = self.conv1_1(x) # 0
        l.append(x)
        x = self.conv1_2(x) # 1
        l.append(x)
        x = self.pool(x)

        x = self.conv2_1(x) # 2
        l.append(x)
        x = self.conv2_2(x) # 3
        l.append(x)
        x = self.pool(x)

        x = self.conv3_1(x) # 4
        l.append(x)
        x = self.conv3_2(x) # 5
        l.append(x)
        x = self.conv3_3(x) # 6
        l.append(x)
        x = self.pool(x)

        x = self.conv4_1(x) # 7
        l.append(x)
        x = self.conv4_2(x) # 8
        l.append(x)
        x = self.conv4_3(x) # 9
        l.append(x)
        x = self.pool(x)

        x = self.conv5_1(x) # 10
        l.append(x)
        x = self.conv5_2(x) # 11
        l.append(x)
        x = self.conv5_3(x) # 12
        l.append(x)
        
        return tuple(l[i] for i in self.output_layers) 

