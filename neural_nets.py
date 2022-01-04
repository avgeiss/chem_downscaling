import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, concatenate, add, Layer, MaxPooling2D, Activation
from tensorflow.keras.optimizers import Adam
from keras.models import Model

#constants used for standardizing/dimensionalizing the data
MU = {'no2':            -25.1,
      'so2':            -24.4,
      'co':             -16.4,
      'pm25_rh35_gcc':  1.0}
SIGMA = {'no2':             3.0,
         'so2':             7.0,
         'co':              0.3,
         'pm25_rh35_gcc':   2.0}
O3_SCALE = 4E6
res_chan = 64

def LOG_MSE(y_true,y_pred):
    y_true = tf.math.log(y_true+1E-32)
    y_pred = tf.math.log(y_pred+1E-32)
    return tf.math.reduce_mean((y_true-y_pred)**2)

#NN-upsampling
def up(x,shape=(8,10)):
    return UpSampling2D(shape,interpolation='nearest')(x)

#2D-AVG-Downsampling
def down(x,shape = (8,10)):
    return AveragePooling2D(shape)(x)

#2D-downsampling weighted by cosine of latitude
def down_lat(x,clat,shape=(8,10)):
    return AveragePooling2D(shape)(x*clat)/AveragePooling2D(shape)(clat)  

#custom layer that can perform pixel-shuffle up-sampling for rectangular regions 
#(tf.nn.depth_to_space only supports square regions right now)
class DepthToSpace(Layer):
    def __init__(self,shape):
        super(DepthToSpace,self).__init__()
        if type(shape) is int:
            shape = (shape,shape)
        self.shape = shape
    
    def call(self,x):
        sz, psz = tf.shape(x), self.shape
        nchan = x.get_shape().as_list()[-1]
        x = tf.reshape(x, [sz[0],sz[1],sz[2],psz[0],psz[1],nchan//(psz[0]*psz[1])])
        x = tf.transpose(x, [0,1,3,2,4,5])
        x = tf.reshape(x, [sz[0],sz[1]*psz[0],sz[2]*psz[1],nchan//(psz[0]*psz[1])])
        return x

#layer for enforcing conservation rules. (Eqn (3) in paper)
class EnfConsv(Layer):
    def __init__(self,shape):
        super(EnfConsv,self).__init__()
        self.shape = shape
    
    def call(self,x_lr,x_hr):
        return x_hr*up(x_lr/(down(x_hr) + 1E-32))

#layer for enforcing conservation rules while accounting for lat/lon grid (Eqn (5) in paper)
class EnfConsvLat(Layer):
    def __init__(self,shape):
        super(EnfConsvLat,self).__init__()
        self.shape = shape
    
    def call(self,x_lr,x_hr,clat):
        return x_hr*up(x_lr/(down_lat(x_hr,clat) + 1E-32))

#standardizes the CNN inputs to be ~normal
class Standardize(Layer):
    def __init__(self,chem):
        super(Standardize,self).__init__()
        self.mu = MU[chem]
        self.sigma = SIGMA[chem]
    
    def call(self,x):
        return (tf.math.log(x+1E-32)-self.mu)/self.sigma

#re-dimensionalizes the CNN outputs so they have correct units without post-processing
class Dimensionalize(Layer):
    def __init__(self,chem):
        super(Dimensionalize,self).__init__()
        self.mu = MU[chem]
        self.sigma = SIGMA[chem]
    
    def call(self,x):
        return tf.math.exp(x*self.sigma + self.mu)
    
#the enhanced deep residual super resolution architecture:
def edrn_core(x,chan=64,blocks=16):
    #the core section of the edrn containing multiple residual blocks:
    long_skip = x
    def res_block(x_in):
        x = Conv2D(chan,(3,3),padding='same',activation='relu')(x_in)
        x = Conv2D(chan,(3,3),padding='same',activation='linear')(x)
        return add([x_in,x])
    
    for i in range(blocks):
        x = res_block(x)
        
    x = add([long_skip,x])
    
    #the upsampling module:
    x = Conv2D(chan*5*4,(3,3),padding='same',activation='linear')(x)
    x = DepthToSpace((4,5))(x)
    x = Conv2D(chan*2*2,(3,3),padding='same',activation='linear')(x)
    x = DepthToSpace((2,2))(x)
    
    return x

def edrn_8x10x( chem , use_cnsv = False, use_lat = False, use_clim = False):
    
    #define the input standardization and the output dimensionalization + transfer function,
    #this is done differently for O3 because it has a different distribution
    if chem == 'o3':
        in_func = lambda x: x
        out_func = lambda x: Activation('sigmoid')(x)
        loss = 'MSE'
        LR = 0.000005
    else:
        in_func = Standardize(chem)
        out_func = Dimensionalize(chem)
        loss = LOG_MSE
        LR = 0.0001
    
    
    res_chan = 64
    x_in = Input((None,None,1))      #input sample
    inputs = [x_in]
    
    #determine how the HR input data will be downsampled. This layer means we dont have to
    #do downsampling of HR samples on CPU before each batch and should be removed before use
    if use_lat:
        l_in = Input((None,None,1)) 
        x_lr = down_lat(x_in,l_in)
        inputs.append(l_in)
    else:
        x_lr = down(x_in)
    
    #The first layers of the resnet. If we are including climatology it will be added here
    #and at the end of the network. If a latitude weighted grid is used add the latitude as an input channel
    x = in_func(x_lr)
    x = Conv2D(res_chan,(9,9),padding='same',activation='linear')(x)
    if use_clim:
        c_in = Input((None,None,1))
        inputs.append(c_in)
        c = in_func(c_in)
        c = Conv2D(res_chan,(11,11),padding='same',activation='relu')(c)
        c = MaxPooling2D((4,5))(c)
        c = Conv2D(res_chan,(5,5),padding='same',activation='relu')(c)
        c = MaxPooling2D((2,2))(c)
        x = concatenate([x,c])
        x = Conv2D(res_chan,(3,3),padding='same',activation='linear')(x)
    
    #the residual blocks and upsampling module:
    x = edrn_core(x,chan = res_chan)
    
    #if climatology is being used add it in here at full resolution as well.
    if use_clim:
        c = in_func(c_in)
        c = Conv2D(res_chan,(3,3),padding='same',activation='relu')(c)
        x = concatenate([x,c])
        x = Conv2D(res_chan,(3,3),padding='same',activation='linear')(x)
    
    #the final output layer:
    x = Conv2D(1,(9,9),padding='same',activation='linear')(x)
    x = out_func(x)
    
    #enforce conservation rules:
    if use_cnsv:
        if use_lat:
            x = EnfConsvLat((8,10))(x_lr,x,l_in)
        else:
            x = EnfConsv((8,10))(x_lr,x)
        
    cnn = Model(inputs,x)
    cnn.compile(optimizer=Adam(learning_rate = LR),loss=loss,metrics='MAE') 
    return cnn

