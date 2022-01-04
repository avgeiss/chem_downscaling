#evaluates each of the CNNs alongside several benchmark schemes:
from glob import glob
import numpy as np
from utils import bilin_interp, bicub_interp, nearest_interp, climo_downscale, log_bicub_interp, log_bilin_interp
from functools import partial
from neural_nets import LOG_MSE as custom_loss
from utils import LOG_MSE, LOG_PSD, LOG_SSIM, MAE, MSE, PSD, SSIM
from keras.models import load_model
from utils import read_config
import sys
PATHS = read_config()
data_dir = PATHS['npy_dir']
compound = 'no2'#sys.argv[1]
print('Evaluating test set for ' + compound)

metric_names = ['MAE','LOG_MSE','LOG_SSIM']
exp_names = ['ctrl_eqa_noclim','cnsv_eqa_noclim','cnsv_lat_noclim','ctrl_lat_clim','cnsv_lat_clim'] #filenames for the CNN experiments
model_names = ['nearest','bilinear','bicubic','climo']  #names of the benchmark schemes
model_names.extend(exp_names)

#load the data:
test_files = glob(data_dir + compound + '/2020*.npy')
test_files.sort()
climo = np.load(data_dir + compound + '_climo.npy')[4:-5,:]
climo = climo[np.newaxis,:,:,np.newaxis]
lat = np.cos(np.linspace(-90,90,721)[4:-5]*np.pi/180)[:,np.newaxis]
lat = np.repeat(lat,1440,axis=1)[np.newaxis,...,np.newaxis]

#define all the super-res schemes:
if compound != 'o3':
    models = [nearest_interp,log_bilin_interp,log_bicub_interp,partial(climo_downscale,climo=climo.squeeze())]
else:
    models = [nearest_interp,bilin_interp,bicub_interp,partial(climo_downscale,climo=climo.squeeze())]

#loads the trained neural networks
def load_cnn(name):
    return load_model('./experiments/' + compound + '_' + name + '/cnn', custom_objects={'LOG_MSE':custom_loss})

#helper function that generates outputs for a CNN
def predict(x,cnn):
    x = x[np.newaxis,:,:,np.newaxis]
    inputs = [x,lat,climo][:len(cnn.inputs)]
    if len(cnn.inputs)==1:
        inputs = inputs[0]
    return cnn.predict(inputs).squeeze()

#now add the CNNs to the list of super-res models
models.extend([partial(predict,cnn=load_cnn(name)) for name in exp_names])

#error metric functions:
if compound != 'o3':
    metrics = [MAE, LOG_MSE, LOG_SSIM]
else:
    metrics = [MAE, MSE, SSIM]

#computes all error metrics for a given sample/ground-truth pair
def compute_metrics(x,y):
    if compound != 'o3':
        psd = LOG_PSD(y)
    else:
        psd = PSD(y)
    scalar_errors = [mf(x,y) for mf in metrics]
    return scalar_errors, psd

#iterates over all of the models
def eval_models(x):
    metrics = []
    for m in models:
        metrics.append(compute_metrics(x,m(x)))
    return metrics

#iterate over the test datset and compute metrics
err, true_psd = [], []
for f in test_files:
    data = np.load(f)[:,4:-5,:]
    print('\n' + f,end='',flush=True)
    for i in range(data.shape[0]):
        print('.',end='',flush=True)
        err.append(eval_models(data[i,:,:]))
        if compound != 'o3':
            true_psd.append(LOG_PSD(data[i,:,:]))
        else:
            true_psd.append(PSD(data[i,:,:]))
    
#convert error metrics to matrix format
N = data.shape[0]*len(test_files)
scalar_metrics = np.zeros((len(models),len(metrics),N))
psd = np.zeros((len(models),len(err[0][0][1]),N))
for i in range(len(test_files)*24):
    for m in range(len(models)):
        scalar_metrics[m,:,i], psd[m,:,i] = err[i][m][:]

np.savez('./' + compound + '_metrics.npz',errors = scalar_metrics, psd=psd, true_psd=true_psd)