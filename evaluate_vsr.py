#code for computing error metrics for the VSR schemes on the test set.
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model
from tqdm import tqdm
from neural_nets import LOG_MSE as loss
from utils import LOG_SSIM, MAE, LOG_MSE, LOG_PSD, MSE, SSIM, PSD
import pickle
import sys

c = sys.argv[1]                                 #one CLI argument for the species e.g. 'no2','o3','so2', etc.
DATA_DIR = '/home/andrew/data/deepclimate/'     #location of data files
trunc_files = None                              #change to an int to use a shortened file
                                                #list (for testing), otherwise None
error_data = {}                                 #container for computed error metrics

#does 8x10 averge downsampling of input data
def downsample(x):
    sz = x.shape
    x = np.reshape(x,[sz[0],8,sz[1]//8,10,sz[2]//10],order='F')
    return np.mean(x,axis=(1,3))

#load the meteorology and mixing ratio data:
print('Loading meteorology data...')
files = glob(DATA_DIR + 'met/2020*.npy')
files.sort()
files = files[:trunc_files]
met = np.concatenate([np.load(f) for f in tqdm(files)])
met_scales = np.array([0.02,2E-5,0.02])[np.newaxis,np.newaxis,np.newaxis,:]
met_offset = np.array([0,-2.0,0])[np.newaxis,np.newaxis,np.newaxis,:]
met = met*met_scales + met_offset
print('\n\nProcessing ' + c + '-----------------------------------------------')
files = glob(DATA_DIR + c + '/2020*.npy')
files.sort()
files = files[:trunc_files]
print('Loading mixing ratio data...')
data = np.zeros((len(files)*24,712,1440),dtype='float32')
for i in tqdm(range(len(files))):
    data[i*24:i*24+24,...] = np.load(files[i])[:,4:-5,:]
if c == 'o3':
    data = data*1E7-1.0

print('Evaluating CNSV VSR CNN...')
cnn = load_model('./experiments/' + c + '_vsr_cnsv/cnn',custom_objects={'LOG_MSE':loss})
mae,mse,ssim,psd = [],[],[],[]
for i in range(data.shape[0]-7):
    inputs = np.concatenate([downsample(data[i:i+7,...])[...,np.newaxis],met[i:i+7,...]],axis=-1)
    output = cnn.predict(inputs[np.newaxis,...])[3].squeeze()
    target = data[i+3,...]
    if c == 'o3':
        output = (output+1.0)/1E7
        target = (target+1.0)/1E7
        mae.append(MAE(output,target))
        mse.append(MSE(output,target))
        ssim.append(SSIM(output,target))
        psd.append(PSD(output))
    else:
        mae.append(MAE(output,target))
        mse.append(LOG_MSE(output,target))
        ssim.append(LOG_SSIM(output,target))
        psd.append(LOG_PSD(output))
    print(str(i/24) + ': ' + str(mae[-1]))
error_data['cnsv'] = {'MAE':mae,'MSE':mse,'SSIM':ssim,'PSD':psd}

print('Evaluating CTRL VSR CNN...')
cnn = load_model('./experiments/' + c + '_vsr_ctrl/cnn',custom_objects={'LOG_MSE':loss})
mae,mse,ssim,psd = [],[],[],[]
for i in range(data.shape[0]-7):
    inputs = np.concatenate([downsample(data[i:i+7,...])[...,np.newaxis],met[i:i+7,...]],axis=-1)
    output = cnn.predict(inputs[np.newaxis,...])[3].squeeze()
    target = data[i+3,...]
    if c == 'o3':
        output = (output+1.0)/1E7
        target = (target+1.0)/1E7
        mae.append(MAE(output,target))
        mse.append(MSE(output,target))
        ssim.append(SSIM(output,target))
        psd.append(PSD(output))
    else:
        mae.append(MAE(output,target))
        mse.append(LOG_MSE(output,target))
        ssim.append(LOG_SSIM(output,target))
        psd.append(LOG_PSD(output))
    print(str(i/24) + ': ' + str(mae[-1]))
error_data['ctrl'] = {'MAE':mae,'MSE':mse,'SSIM':ssim,'PSD':psd}

#save results using pickle
f = open('./' + c + '_vsr_errors.pkl','wb')
pickle.dump(error_data,f)
f.close()