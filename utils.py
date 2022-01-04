#Andrew Geiss 2021
#
#Utility functions for atmos. chem. super resolution project


#  Reads in info from the config.ini file  ##################################################################
#im running these on an hpc system and desktop so this config.ini file contains
#information about input/output file paths depending on the system
import configparser
def read_config():
    #parses the config.ini file and returns dictionaries containing file paths, and scale factors 
    config = configparser.ConfigParser()
    config.read('config.ini')
    paths = dict(config['PATHS'])
    return paths

PATHS = read_config()
CHEMS = ['no2','o3','so2','co','pm25_rh35_gcc']

#  Code for converting netCDF files to numpy format  ########################################################
import numpy as np
from glob import glob
from multiprocess import Pool
from netCDF4 import Dataset
def proc_date(fnames):
    def ncload(file_name,var_name=None):
        #loads 'var_name' from 'file_name'. If var_name==None returns a list with all variables
        if var_name == None:
            return np.array([ncload(file_name,var) for var in CHEMS])
        else:
            data = np.squeeze(Dataset(file_name).variables[var_name.upper()][:].data)
            return np.float32(data)
    
    #helper function for parallelization
    date = fnames[0].split('.')[-2][:8]
    print('Converting ' + date + ' to numpy format...',flush=True)
    data = np.array([ncload(f) for f in fnames])
    for i in range(len(CHEMS)):
        np.save(PATHS['npy_dir'] + CHEMS[i] + '/' + date + '.npy',np.squeeze(data[:,i,:,:]))

def convert_to_numpy():
    #code to convert netCDF files to daily .npy files:
    nc_files = glob(PATHS['nc_path'])
    nc_files.sort()
    dates = [f.split('.')[-2][:8] for f in nc_files]
    unique_dates = list(set(dates))
    unique_dates.sort()
    date_files = []
    for date in unique_dates:
        date_files.append([f for f in nc_files if date in f])
    p = Pool(24)
    p.map(proc_date, date_files, chunksize=1)
    p.close()



#  These functions compute the climatological averages of each of the variables  ############################
def daily_mean(f):
    #helper function to compute climatologies. takes a daily mean for the file f
    print(f,flush=True)
    data = np.double(np.load(f))
    return np.mean(data,axis=0)
        
def compute_climatologies():
    for v in CHEMS:
        print('Computing Climatology for ' + v + '...',flush=True)
        dates = [date[-12:-4] for date in glob(PATHS['npy_dir'] + v + '/*.npy')]
        dates.sort()
        train_dates = [date for date in dates if date[:4] == '2018' or date[:4] == '2019']
        files = []
        for d in train_dates:
            files.append(PATHS['npy_dir'] + v + '/' + d + '.npy')
        p = Pool(24)
        daily_means = p.map(daily_mean,files,chunksize=1)
        p.close()
        mean = np.mean(daily_means,axis=0).squeeze()
        np.save(PATHS['npy_dir'] + v + '_climo.npy',mean)




#  computes some statistics about the training set  #########################################################
def get_stats_from_file(f):
    print(f,flush=True)
    data = np.load(f)
    data[data<=0] = np.nan
    data = np.log(data)
    mean = np.nanmean(data)
    mx = np.nanmax(np.abs(data-mean))
    return [mean, mx]

def compute_stats():
    
    def get_stats_from_file(f):
        print(f,flush=True)
        data = np.load(f)
        data[data<=0] = np.nan
        data = np.log(data)
        mean = np.nanmean(data)
        mx = np.nanmax(np.abs(data-mean))
        return [mean, mx]
    
    p = Pool(24)
    
    for c in CHEMS:
        files = glob(PATHS['npy_dir'] + c + '/2018*.npy') + glob(PATHS['npy_dir'] + c + '/2019*.npy')
        files.sort()
        stats = p.map(get_stats_from_file,files,chunksize=1)
        np.save(c + '_stats.npy',np.array(stats))
    
    p.close()


# loads data from numpy files (for loading a validation set)  ###############################################
def load_dates(chem, dates, use_lat = False, use_clim = False):
    valid = []
    for d in dates:
        valid.append(np.float32(np.load(PATHS['npy_dir'] + chem + '/' + d + '.npy')))
    valid = [np.concatenate(valid,axis=0)[:,1:-1,:,np.newaxis]]
    
    #get latitude inputs if requested
    if use_lat:
        lat = np.cos(np.linspace(-90,90,721)[np.newaxis,1:-1,np.newaxis,np.newaxis]*np.pi/180)
        lat = np.repeat(lat,1440,axis=2)
        lat = np.repeat(lat,valid[0].shape[0],axis=0)
        valid.append(lat)
    
    #get climatology inputs if requested:
    if use_clim:
        climo = np.load(PATHS['npy_dir'] + chem + '_climo.npy')
        climo = np.repeat(climo[np.newaxis,1:-1,:,np.newaxis],valid[0].shape[0],axis=0)
        valid.append(climo)
        
    return valid


# these functions compute benchmark SR scores using conventional downscaling/interpolation  #################

from scipy.interpolate import interp2d

def coarsen(x,scale = (8,10)):
    sz = x.shape
    x = np.reshape(x,[scale[0],sz[0]//scale[0],scale[1],sz[1]//scale[1]],order='F')
    x = np.mean(x,axis=(0,2),keepdims=False)
    return x

def scipy_interp(z, scale, kind):
    sz = z.shape
    y = np.linspace(0,1,sz[0],endpoint=True)
    x = np.linspace(0,1,sz[1],endpoint=True)
    interpolator = interp2d(x,y,z,kind=kind)
    y = np.linspace(0,1,sz[0]*scale[0],endpoint=True)
    x = np.linspace(0,1,sz[1]*scale[1],endpoint=True)
    z = interpolator(x,y)
    return z

def nearest_interp(x, scale = (8,10)):
    x = coarsen(x,scale)
    sz = x.shape
    x = np.reshape(x,[1,sz[0],1,sz[1]],order='F')
    x = np.repeat(x,scale[0],axis=0)
    x = np.repeat(x,scale[1],axis=2)
    x = np.reshape(x,[sz[0]*scale[0],sz[1]*scale[1]],order='F')
    return x
    
def climo_downscale(x,climo,scale=(8,10)):
    climo_weights = climo/nearest_interp(climo,scale)
    x = nearest_interp(x, scale)*climo_weights
    return x

def bicub_interp(x, scale=(8,10)):
    x = coarsen(x,scale)
    x = scipy_interp(x,scale,'cubic')
    return x

def bilin_interp(x, scale=(8,10)):
    x = coarsen(x,scale)
    x = scipy_interp(x,scale,'linear')
    return x

def log_bicub_interp(x, scale=(8,10)):
    x = coarsen(x,scale)
    x = np.log(x+1E-16)
    x = scipy_interp(x,scale,'cubic')
    x = np.exp(x)
    return x

def log_bilin_interp(x, scale=(8,10)):
    x = coarsen(x,scale)
    x = np.log(x+1E-16)
    x = scipy_interp(x,scale,'linear')
    x = np.exp(x)
    return x


#define testing metrics:
def MAE(x,y):
    return np.mean(np.abs(x-y))

from scipy.ndimage.filters import gaussian_filter
def SSIM(imx,imy,window=4):
    
    mn = np.min([np.min(imx),np.min(imy)])
    mx = np.max([np.max(imx),np.max(imy)])
    imx = (imx-mn)/(mx-mn)
    imy = (imy-mn)/(mx-mn)    
    
    #constants:
    c1 = 0.01**2
    c2 = 0.03**2
    c3 = c2/2.0
    
    # #patch statistics
    mux = gaussian_filter(imx,window)
    muy = gaussian_filter(imy,window)
    muxy = mux*muy
    mux2 = mux*mux
    muy2 = muy*muy
    sxy = gaussian_filter(imx*imy,window)-muxy
    sx2 = gaussian_filter(imx*imx,window)-mux2
    sy2 = gaussian_filter(imy*imy,window)-muy2
    sxsy = np.sqrt(np.abs(sx2*sy2))
    

    #luminance, contrast and structure:
    l = (2*muxy+c1)/(mux2+muy2+c1)
    c = (2*sxsy + c2)/(sx2+sy2+c2)
    s = (sxy+c3)/(sxsy+c3)
    ssim = l*c*s
    
    return np.nanmean(ssim)

def LOG_SSIM(x,y):
    return SSIM(np.log(x+1e-32),np.log(y+1e-32))

def MSE(x,y):
    return np.mean((x-y)**2)

def LOG_MSE(x,y):
    return MSE(np.log(x+1e-32),np.log(y+1e-32))

def PSD(x):
    x = x-np.mean(x)
    N = x.shape[-1]
    fx = np.fft.fft(x,axis=-1)[:,1:N//2]
    psdx = (2/N)*np.abs(fx)**2
    return np.mean(10*np.log10(psdx),axis=0)
    
def LOG_PSD(x):
    return PSD(np.log(x+1E-32))