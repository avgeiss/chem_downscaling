#set constants:
from utils import read_config
PATHS = read_config()
CHEMS = ['no2','o3','so2','co','pm25_rh35_gcc']#list of chemical species names
N_EPOCHS = 1000 #total number of epochs
LR_REDUCE = 800 #after how many epochs to reduce the learning rate by 1/10
EPOCH_SIZE = 500 #days (x24 samples)  
BATCH_SIZE = 12 #samples per mini-batch
VALID_FREQ = 10 #use only every Nth day from the validation set (for speed)
CHIP_SHAPE = (256,320) #size of the samples used for training (hi-res size)


#process CLI flags:
import sys
if len(sys.argv)==5:
    CHEM, POSTPROCESSING, GRID, CLIMATOLOGY = sys.argv[1:]
else:
    CHEM, POSTPROCESSING, GRID, CLIMATOLOGY = 'o3','cnsv','lat','noclim'   #testing code
    #raise ValueError('Requires 5 inputs: CHEM, LOSS, POSTPROCESSING, GRID, CLIMATOLOGY')
    
if not CHEM in CHEMS:
    raise ValueError('Unrecognized species name')
    
if not POSTPROCESSING in ['ctrl','cnsv']:
    raise ValueError('Postprocessing method must be one of: (ctrl,cnsv)')
else:
    use_cnsv = (POSTPROCESSING == 'cnsv')

if not GRID in ['lat','eqa']:
    raise ValueError('Grid option must be one of: (eqa,lat)')
else:
    use_lat = (GRID == 'lat')

if not CLIMATOLOGY in ['clim','noclim']:
    raise ValueError('Climatology option must be one of: (clim,noclim)')
else:
    use_clim = (CLIMATOLOGY == 'clim')
    
EXPERIMENT_NAME = CHEM + '_' + POSTPROCESSING + '_' + GRID + '_' + CLIMATOLOGY
print('Running experiment: ' + EXPERIMENT_NAME)

#make sure the output directory exists
import os
if not os.path.isdir(PATHS['out_dir'] + EXPERIMENT_NAME):
    print('Output directory doesn''t exist, making directory: ' + EXPERIMENT_NAME)
    os.mkdir(PATHS['out_dir'] + EXPERIMENT_NAME)




# functions for trianing and validation data ################################################################

#get the file names for training and validation:
from glob import glob
from utils import load_dates
#use 2018 and 2019 for training, 2020 for testing, and 2021 for validation:
dates = [date[-12:-4] for date in glob(PATHS['npy_dir'] + CHEM + '/*.npy')]
dates.sort()
train_dates = [date for date in dates if date[:4] == '2018' or date[:4] == '2019']
valid_dates = [date for date in dates if date[:4] == '2021']
valid_dates = valid_dates[::VALID_FREQ]
#load the validation set:
validation = load_dates(CHEM, valid_dates, use_lat, use_clim)
validation = [v[:,3:-4,:,:] for v in validation]
if CHEM == 'o3':
    validation[0] *= 4.0E6
    if use_clim:
        validation[-1] *= 4.0E6

#generates training epochs
import numpy as np
DATA_SHAPE = validation[0].shape[1:3]
def get_epoch():
    
    #unpack info about data shapes:
    Ny,Nx = CHIP_SHAPE
    Dy,Dx = DATA_SHAPE[0],DATA_SHAPE[1]
    N = EPOCH_SIZE*24 #number of samples in the epoch
    
    #choose the samples and locations to sub-sample:
    dates = np.random.choice(train_dates, size=EPOCH_SIZE)
    files = [PATHS['npy_dir'] + CHEM + '/' + d + '.npy' for d in dates]
    Ry,Rx = np.random.randint(0,Dy-Ny,(N,)), np.random.randint(0,Dx-Nx,(N,))
    
    #preallocate memory:
    samples = np.zeros(shape=(N,Ny,Nx,1),dtype='float32')
    
    #load climatology if needed:
    if use_clim:
        climo = np.load(PATHS['npy_dir'] + CHEM + '_climo.npy')[1:-1,:,np.newaxis]
        climo_samples = np.zeros(shape=(N,Ny,Nx,1),dtype='float32')
    else:
        climo_samples = None
        
    #prep latitudes if needed:
    if use_lat:
        lat = np.cos(np.linspace(-90,90,721)[1:-1]*np.pi/180)[:,np.newaxis,np.newaxis]
        lat = np.repeat(lat,1440,axis=1)
        lat_samples = np.zeros(shape=(N,Ny,Nx,1),dtype='float32')
    else:
        lat_samples = None
    
    #load samples:
    for i in range(EPOCH_SIZE):
        data = np.load(files[i])[:,1:-1,:,np.newaxis]
        for j in range(24):
            e = i*24 + j
            samples[e,...] = data[j,Ry[e]:Ry[e]+Ny,Rx[e]:Rx[e]+Nx,:]
            if use_clim:
                climo_samples[e,...] = climo[Ry[e]:Ry[e]+Ny,Rx[e]:Rx[e]+Nx,:]
            if use_lat:
                lat_samples[e,...] = lat[Ry[e]:Ry[e]+Ny,Rx[e]:Rx[e]+Nx,:]
    
    if CHEM == 'o3':
        samples *= 4.0E6
        if use_clim:
            climo_samples *= 4.0E6
    
    inputs = [samples, lat_samples, climo_samples]
    inputs = [inp for inp in inputs if not inp is None]
    return inputs




#  Training code  ###########################################################################################
import gc
import concurrent.futures
from datetime import datetime
import keras.backend as K
def train(cnn):
    save_dir = PATHS['out_dir'] + EXPERIMENT_NAME + '/'
    threader = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    TL, VL, MAE = [], [], []
    
    def fit(x):
        print(datetime.now(),end = '  -->  ',flush=True)
        history = cnn.fit(x,x[0],batch_size=BATCH_SIZE,verbose=True).history
        loss = history['loss'][0]
        mae = history['MAE'][0]
        print(datetime.now(),end = '   ',flush=True)
        return loss, mae
    
    x = get_epoch()
    for e in range(N_EPOCHS):
        gc.collect()
        
        #get the next epoch data and train on the previous epoch simultaneously
        fit_thread = threader.submit(fit,x)
        epoch_thread = threader.submit(get_epoch)
        
        #wait for outputs from each thread:
        tl, mae = fit_thread.result()
        TL.append(tl)
        MAE.append(mae)
        x = epoch_thread.result()
        
        #log performance
        print('::: EPOCH ' + str(e) + ' Training: ' + str(tl) + ' ::: MAE: ' + str(mae),flush=True)
        
        if e % 10 == 9:
            #compute validation loss every 10th epoch
            output = cnn.predict(validation,verbose=False,batch_size=1)
            VL.append(np.mean(np.abs(output-validation[0])))
            
            #save losses
            np.save(save_dir + 'validation_mae.npy', VL)
            np.save(save_dir + 'training_loss.npy', TL)
            np.save(save_dir + 'training_mae.npy', MAE)
            print('::: Epoch ' + str(e) + ' Validation: ' + str(VL[-1]) + ' :::',flush=True)
        
            #save model snapshot if it has achieved its best validation score
            if VL[-1] == np.min(VL):
                cnn.save(save_dir + 'cnn')
            
        if e % 100 == 99:
            cnn.save(save_dir + 'cnn_' + str(e+1).zfill(4))
            
        if e == LR_REDUCE:
            print('Reducing Learning Rate...',flush=True)
            K.set_value(cnn.optimizer.lr,K.get_value(cnn.optimizer.lr)*0.1)


from neural_nets import edrn_8x10x
cnn = edrn_8x10x(CHEM, use_cnsv = use_cnsv,use_lat = use_lat,use_clim = use_clim)
train(cnn)