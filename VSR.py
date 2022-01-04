import os
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
import gc
from neural_nets import DepthToSpace, Standardize, Dimensionalize, LOG_MSE, EnfConsv
from keras.layers import Input, Conv2D, Conv3D, Reshape, add
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from glob import glob
from os.path import expanduser
import keras.backend as K
import sys
from datetime import datetime

#determine settings:
DATA_DIR = expanduser('~/data/deepclimate/')    #contains folders labeled 'o3', 'no2', 'met', etc. with daily npy files
if not len(sys.argv)==1:
    chem, postprocessing = sys.argv[1:3]        #process arguments
else:
    chem, postprocessing = 'o3', 'cnsv'         #defaults if the script is run without arguments
    
use_cnsv = postprocessing == 'cnsv'
exp_name = chem + '_vsr_' + postprocessing

print('------------------------------------------------ Running: ' + exp_name + ' ------------------------------------------------')

#object for reading/preprocessing data and generating random batches
class Dataset:
    
    def __init__(self,chem,nframes=7):
        files = glob(DATA_DIR + 'met/201*.npy')
        files.sort()
        self.files = [f.split('/')[-1] for f in files]
        self.nframes = nframes
        self.chem = chem
        self.met_scales = np.array([0.02,2E-5,0.02])[np.newaxis,np.newaxis,np.newaxis,:]
        self.met_offset = np.array([0,-2.0,0])[np.newaxis,np.newaxis,np.newaxis,:]
    
    #performs 8x10 averaging
    def downsample(self,data):
        sz = data.shape
        data = np.reshape(data,[sz[0],8,sz[1]//8,10,sz[2]//10],order='F')
        data = np.mean(data,axis=(1,3))
        return data
    
    #selects a random time sample and loads the data for +/- 3 hours around that sample
    def random_sample(self):
        hour = np.random.randint(24)
        fidx = np.random.randint(len(self.files)-1)
        mxr = np.double(np.load(DATA_DIR + self.chem + '/' + self.files[fidx]))
        met = np.double(np.load(DATA_DIR + 'met/' + self.files[fidx]))
        if hour+self.nframes > 24:
            mxr = np.concatenate([mxr, np.load(DATA_DIR + self.chem + '/' + self.files[fidx+1])],axis=0)
            met = np.concatenate([met, np.load(DATA_DIR + 'met/' + self.files[fidx+1])],axis=0)
        targets, met = mxr[hour:hour+self.nframes,4:-5,...], met[hour:hour+self.nframes,...]
        if self.chem == 'o3':
            targets = targets*1E7 - 1.0 #o3 data is scaled when loaded, others are scaled internally by the CNN
        met = met*self.met_scales + self.met_offset
        mxr = self.downsample(targets)[...,np.newaxis]
        inputs = np.concatenate([mxr,met],axis=-1)
        return inputs, targets
    
    #returns a training batch by calling random_sample() repeatedly
    def batch(self,bsize=4):
        inputs, targets = zip(*[self.random_sample() for _ in range(bsize)])
        targets =  list(np.transpose(np.array(targets),[1,0,2,3]))
        return np.array(inputs,dtype='float64'), targets
            
        
def EDRN3D(chem):
    
    #this function generates the core of the EDRN architecture, by creating a set
    #of residual blocks with one long skip connection
    def res_blocks(x,chan,blocks=12):
        def res_block(x_in,chan):
            x = Conv3D(chan,(3,3,3),padding='same',activation='relu')(x_in)
            x = Conv3D(chan,(3,3,3),padding='same',activation='linear')(x)
            return add([x_in,x])
        long_skip = x
        for i in range(blocks):
            x = res_block(x,chan)
        x = add([long_skip,x])
        return x
    
    #process inputs
    xin = Input((7,89,144,4))
    if chem == 'o3':
        x = Conv3D(256,(3,3,3),padding='same',activation='linear')(xin)
    else:
        #these lines are needed to standardize the log-normally distributed chemical species
        inputs = tf.unstack(xin,axis=-1)
        inputs[0] = Standardize(chem)(inputs[0])
        inputs = tf.stack(inputs,axis=-1)
        x = Conv3D(256,(3,3,3),padding='same',activation='linear')(inputs)
    
    #add the core edrn architecture to the CNN
    x = res_blocks(x,256,12)
    x = tf.transpose(x,(0,2,3,1,4))
    x = Reshape((89,144,7*256))(x)
    
    #the upsampling module:
    x = Conv2D(64*8*10,(1,1),padding='same',activation='relu')(x)
    x = DepthToSpace((8,10))(x)
    
    #the output layer:
    if chem == 'o3':
        sr_frame = Conv2D(7,(3,3),padding='same',activation='tanh')(x)
    else:
        sr_frame = Dimensionalize(chem)(sr_frame)
        sr_frame = Conv2D(7,(3,3),padding='same',activation='linear')(x)
    
    #layers for enforcing conservation rules
    if use_cnsv:
        lr_frame = tf.transpose(xin[...,0],[0,2,3,1])
        if chem == 'o3':
            sr_frame = EnfConsv((8,10))(lr_frame+1.0,sr_frame+1.0)-1.0
        else:
            sr_frame = EnfConsv((8,10))(lr_frame,sr_frame)
    
    #the output frames are unstacked so diffferent loss weights can be used during training
    return Model(xin,tf.unstack(sr_frame,axis=-1))

#initialize training variables:
lr = 1E-4
mse, mae = [],[]
i0 = 0

#use the -rind flag followed by a batch number to resume interrupted training
cnn = EDRN3D(chem)
if '-rind' in sys.argv:
    i0 = int(sys.argv[sys.argv.index('-rind') + 1])
    print('Resuming from ' + str(i0) + ' ----------------------------')
    cnn = load_model('./experiments/' + exp_name + '/cnn',custom_objects={'LOG_MSE':LOG_MSE})
    mse = list(np.load('./experiments/' + exp_name + '/training_loss.npy'))
    mae = list(np.load('./experiments/' + exp_name + '/training_mae.npy'))
    if i0 > 32_000:
        lr = 1E-5

#loss weights for each of the output frames (the center frame is the target)
#asking the CNN to accurately reconstruct all frames gave better performance on the 
#center frame than computing loss on the center frame alone
weights = np.array([1,4,16,64,16,4,1])/106

#select loss function and compile model
if chem == 'o3':
    loss = 'MSE'
else:
    loss = LOG_MSE
cnn.compile(optimizer=Adam(learning_rate = lr),loss = loss,loss_weights=list(weights),metrics = 'MAE')
dataset = Dataset(chem)

#training loop
for i in range(i0,40_000):
    inputs, targets = dataset.batch(4)
    l = cnn.train_on_batch(inputs,targets)
    mse.append(l[4]); mae.append(l[11])
    if i % 10 == 0:
        print('Iter: ' + str(i) + '   ' + str(datetime.now()) + ':   MSE: ' + str(np.mean(mse[-3*24:])) + '   MAE: ' + str(np.mean(mae[-3*24:])))
        gc.collect()
    if i % 1000 == 999:
        print('Saving...')
        cnn.save('./experiments/' + exp_name + '/cnn')
        np.save('./experiments/' + exp_name + '/training_loss.npy', mse)
        np.save('./experiments/' + exp_name + '/training_mae.npy', mae)
    if i == 32_000:
        print('REDUCING LEARNING RATE...')
        K.set_value(cnn.optimizer.lr,K.get_value(cnn.optimizer.lr)*0.1)