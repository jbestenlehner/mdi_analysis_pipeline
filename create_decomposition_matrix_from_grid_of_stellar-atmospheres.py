import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline as ip1d
import multiprocessing as mp
import my_functions_dev

#Example script to generate the grid input for the pipeline
#Path to grid of models 
path = ''
#Path to the pipeline model folder
out_path = ''
#Reference wavelenght grid
wave_ref=np.load('decomposed_wave.npy')
def model2grid(model):
    #Following lines to be update according to your parameter file and stellar atmosphere grid
    model=model.strip()
    model=model.split(',')
    vrot = float(model[-2])
    v_mac = float(model[-1])
    print(vrot,v_mac)
    #Read in stellar atmosphere model
    model_path=model[0]+'.h5'
    wave_norm, flux_norm, m_param = my_functions_dev.read_FW(model_path, 'VT010',2800,8500)
    #Transfer the model to the instrument resolution
    wave_inst1,flux_inst1=my_functions_dev.gauss_conv(wave_norm, flux_norm,0.65,2900.,5600.)
    wave_inst2,flux_inst2=my_functions_dev.gauss_conv(wave_norm, flux_norm,0.6,5500.,8400.)
    print(flux_inst1.min(),flux_inst1.max())
    print(flux_inst2.min(),flux_inst2.max())
    flux_inst = np.concatenate((flux_inst1[wave_inst1<5550],flux_inst2[wave_inst2>=5550]))
    wave_inst = np.concatenate((wave_inst1[wave_inst1<5550],wave_inst2[wave_inst2>=5550]))
    print(flux_inst.min(),flux_inst.max())
    #Broaden the model according to rotational and macro-turbulent velocities
    if vrot != 0:
        wave_rot,flux_rot=my_functions_dev.rot_conv(wave_inst,flux_inst, vrot, 5650., 3000., 8300.)
    if (v_mac != 0) and (vrot != 0):
        wave_rot,flux_rot=my_functions_dev.gauss_conv(wave_rot,flux_rot, v_mac/300000*5650, 3050.,8250.)
    elif (v_mac != 0) and (vrot == 0):
        wave_rot,flux_rot=my_functions_dev.gauss_conv(wave_inst,flux_inst, v_mac/300000*5650, 3050.,8250.)
    elif (v_mac == 0) and (vrot == 0):
        wave_rot = wave_inst
        flux_rot = flux_inst
    flux_ip=ip1d(wave_rot, flux_rot, k=5)
    print(flux_rot.min(),flux_rot.max())
    return flux_ip(wave_ref)

if __name__ == '__main__':
    #Load stellar parameter file for stellar atmosphere grid
    #Structure should match the model2grid function
    name_sm=pd.read_csv(path+'model_params.csv','w')
    pool = mp.Pool(processes=35)
    model_cube = np.asarray(pool.map(model2grid, name_sm.to_list()), dtype=np.float32)
    #Store model_cube
    np.save(path+'model_cube_flux_XShootU.npy',model_cube)
    #Store wavelength reference grid. This can be different to decomposed_wave.npy, if greater flexibility is required.
    np.save(path+'model_cube_wave_XShootU.npy',wave_ref)
    # Calculation of the decomposition matrix. Depending on available RAM this matrix might be based on a fraction of the grid. Example is given below.
    A = np.linalg.svd(model_cube[np.random.randint(model_cube.shape[0], size = int(model_cube.shape[0]*0.1)),:], full_matrices=0, compute_uv=1)[2]
    #Save entire matrix 
    np.save(path+'decomposed_data_cube.npy',A)
    #or only e.g 15 most important oders.
    A = np.transpose(A[:15])
    np.save(path+'reduced_decomposed_data_cube_XShootU_all_new_cutUVlessHalpha.npy',A)
    #Decompose and store model_cube
    reduced_spectra = model_cube.dot(A)
    np.save(path+'reduced_model_cube.npy',reduced_spectra)
