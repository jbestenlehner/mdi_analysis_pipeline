from a_lot_star import single_star_problem
import numpy as np
import pandas as pd
import time as ttt
from multiprocessing import Pool, Array, get_context
#import matplotlib
#import matplotlib.cm as cm
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
#import scipy.interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as ip1d
from scipy.constants import c as light_speed
from astropy.io import fits
#import my_functions_dev
import ctypes
import pickle

#Path to directory. Could be replaced with os command pwd?
path = '/local2/bestenlehner/XShootU_ift_sp_py3/'


class many_stars(object):
    
    def __init__(self,path,ncpus):
        """
        Reading in grid of stellar atmospheres and observational data.

        Note: all stellar spectra are read as they are analysed at once to 
        determince the model error. Size of grid and number stars to be analysed 
        are limit by the RAM available. The model is iteratively updated. 
        Therefore, computational time to analyses stars does not linearly scale.

        It is adivsable to group simialr objects to not only reduce the computational
        time but also to obtain a represenatative model error.

        Stellar atmosphere grid needs to be decomposed into principle components to
        reduce size in RAM. See example script, ....py.
        """

        self.path=path
        self.ncpus=ncpus

        ##############################################
        # read in model spectra and model wavelengths#
        ##############################################

        # read in filename and RV plus broadening if available
        self.data=pd.read_csv(self.path+'Input_RV.csv')
        #wavelength cube
        self.model_wave = np.load(self.path+'mod_dat/model_cube_wave_XShootU_SMC.npy')
        #read in model names
        self.flags = np.loadtxt(self.path+'mod_dat/model_params_XShootU_SMC.csv',dtype=str,skiprows=1,unpack=True)

        #read in decompose model data cube into shared mememory, so that the grid is not copied for each star.
        # Number in [:??] determined, how many most important components are used for the analysis. A value around 10
        # is usually sufficient. 
        m_shape = np.transpose(np.load(self.path+'mod_dat/decomposed_data_cube_XShootU_SMC.npy')[:12]).shape
        shared_array_base = Array(ctypes.c_double, m_shape[0]*m_shape[1])
        self.A = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.A = self.A.reshape(m_shape)
        self.A[:,:] = np.transpose(np.load(self.path+'mod_dat/decomposed_data_cube_XShootU_SMC.npy')[:12])
        print("Read in model spectra is done")

        #Read in decomposition matrix A into shared memory
        m_shape = np.transpose(np.load(self.path+'mod_dat/reduced_model_cube_XShootU_SMC.npy').T[:12]).shape
        shared_array_base = Array(ctypes.c_double, m_shape[0]*m_shape[1])
        self.reduced_spectra = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.reduced_spectra = self.reduced_spectra.reshape(m_shape)
        self.reduced_spectra[:,:] = np.transpose(np.load(self.path+'mod_dat/reduced_model_cube_XShootU_SMC.npy').T[:12])
        #Reference wavelength to which all model and observations are sampled on.
        #Usually the same as model_wave, but can be modified to ommit specific wavelength regions.
        self.wave = np.load(self.path+'mod_dat/decomposed_wave_XShootU_SMC.npy')
        print("Read in of A is completed")

        #Read in and prepare input spectra for analysis
        linalgtime = ttt.time()
        with Pool(self.ncpus) as pool:
            self.problems=pool.map(unwrap_self_prepare_input, zip([self]*self.data.shape[0], self.data.values))
        print("Prepare input took",ttt.time()-linalgtime,"seconds.")
        print("Read in data spectra is done")

        #Set convergence flags to False
        self.vU_convergence = [False] * self.data.shape[0]
        self.p_convergence = [False] * self.data.shape[0]
        print('Preparation done!')
        print('Begin the analysis')

    def prepare_input(self,star_input):
        """
        Preparation of input spectra and create an object for each spectra.
        Make sure that vacuum/air wavelength of grid and observations agree.

        NaN and negative fluxes should be removed from the observation, if
        not already done by the data reduction pipeline.

        Observations are radial velocity (RV) corrected and transformed on the 
        reference wavelength grid.

        Observational error should be provided. If not available use, the variance.
        """
        #Input file should contain name (unique ID), filename of the spectrum and radial velocity.
        #This line needs to be modified according to the input file.
        #vsini and vmac can be used to limit to the nearest value of the model grid, but currently
        #only used to check, if the derifed broadening of the pipeline is sensible. Large disagreement
        #is a good indicator of a poor fit.
        name,filename,vsini,vmac,rv=star_input
        print(name,filename,rv,vsini,vmac)
        #Read in observational spectra in fits format.
        #Example for ascii files:
        #obs_wave,obs_flux,obs_erro = np.loadtxt(self.path+'obs_dat/'+filename, unpack=True)
        #If observational error is not available the variance or signal to noise (SNR) of the
        # can be uses, e.g. var = np.sqrt(1/SNR) or var = np.std(obs_flux)
        fitsfile=fits.open(self.path+'obs_dat_eDR2/'+filename)
        obs_wave=fitsfile[1].data['WAVELENGTH_AIR']
        obs_flux=fitsfile[1].data['SCI_FLUX_NORM']
        obs_erro=fitsfile[1].data['SCI_FLUX_NORM_ERR']
        #Remove NaN, inf and negative fluxes/errors
        obs_wave=obs_wave[~np.isnan(obs_flux)]
        obs_erro=obs_erro[~np.isnan(obs_flux)]
        obs_flux=obs_flux[~np.isnan(obs_flux)]
        obs_wave=obs_wave[~np.isinf(obs_flux)]
        obs_erro=obs_erro[~np.isinf(obs_flux)]
        obs_flux=obs_flux[~np.isinf(obs_flux)]
        obs_wave=obs_wave[obs_erro >= 0]
        obs_flux=obs_flux[obs_erro >= 0]
        obs_erro=obs_erro[obs_erro >= 0]
        #Correct data for RV
        obs_wave=(1-rv/light_speed*1000)*obs_wave
        #Transform data to reference wavelenght
        data_inp=ip1d(obs_wave,obs_flux,k=5)
        dataobs=data_inp(self.wave)
        red_data = dataobs.dot(self.A)
        data_inp=ip1d(obs_wave,obs_erro,k=5)
        var = data_inp(self.wave)
    
        #Create object for each spectrum
        problem=single_star_problem(self.reduced_spectra,self.flags,red_data,name,var,31093,alpha=0.0,A=self.A)
        print('Preparation for '+filename+' is done')
        return problem


    def _update_v_U(self,convtol,alpha):
        """
        Updating of model error parameter v and U.
        Input: 
            convtol: convergence threshold (float)
            alpha: optional factor which can be used to limit unwanted behaviour,
            e.g mean v absorbs all the differences between observation and model, even though the fit is bad. For the first few iteration v should be set to zero or very small values.

        Note: seriel seems faster than parallel due to overheads
        """
        
        #Updating U using the information of all fits
        #Weights can be used to avoid that poor fits as a results of none stellar features dominate the model error.
        for i,problem in enumerate(self.problems):
            if i==0:
                weight = 1/np.sqrt(np.mean((problem.delta)**2))
                weights = weight
                delta = problem.delta*weight
                self.U = problem.D*weight
#                self.U += vecvec_operator(val=problem.delta)
#                self.U += np.outer(problem.delta,problem.delta)
                self.U +=np.outer(problem.delta,problem.delta)*weight
            else:
                weight = 1/np.sqrt(np.mean((problem.delta)**2))
                weights += weight
                delta += problem.delta*weight
                self.U += problem.D*weight
#                self.U += vecvec_operator(val=problem.delta)
#                self.U += np.outer(problem.delta,problem.delta)
                self.U +=np.outer(problem.delta,problem.delta)*weight
        #delta.val /= len(self.problems)
        #self.U.val /= len(self.problems)
        #delta /= len(self.problems)
        #self.U /= len(self.problems)
        delta /= weights
        self.U /=weights
           
        #Updating of v
        #This is a critical part which can impact the performance of the pipeline
        #Initially it is advisable to set v=0.
        #If the all spectra suffer from similar contamination, a global value for v can be adoppted.
        #For example if only subset are contaminated with nebular, v can be based on the individual self.problems[ii].delta. However, this only works if a reasonable fit has been obtained. A potential solution would be to let contribution of self.problems[ii].delta slowly increase after each interation.
        for ii in range(len(self.problems)):
#            v_new = 1e-5*(self.problems[ii].v+delta)
            v_new = 0.5*(0.01*self.problems[ii].v+0.01*delta)
#            v_new = 0.01*(0.2*self.problems[ii].v+0.8*self.problems[ii].delta)
#            v_new = 1e-5*(self.problems[ii].v+self.problems[ii].delta)
#            v_new = 0.1*(self.problems[ii].v+0.1*self.problems[ii].delta)
            vdist,Udist = self.problems[ii].update_v_U(v_new,self.U)
            #vdist,Udist = self.problems[ii].update_v_U(self.problems[ii].v,self.U)
            #Update convergency flag, if a consistant fit between iteration has been obtained
            if(vdist<convtol and Udist<convtol):
                self.vU_convergence[ii] = True
    
    def _update_para_singleCPU(self):

        pnumbers_old = [problem.pnumber+0. for problem in self.problems]
       
        for problem in self.problems:
            problem.solve_parameter_likelihood(self.reduced_spectra, self.flags, return_DeltaH=1)
       
        pnumbers_new = [problem.pnumber+0. for problem in self.problems]
       
        for ii in range(len(self.problems)):
            print(pnumbers_old[ii], pnumber_new[ii])
            if(pnumbers_new[ii]==pnumbers_old[ii]):
                self.p_convergence[ii] = True
            else:
                self.p_convergence[ii] = False
                self.vU_convergence[ii] = False

    def _update_para(self,problem_p_vU_conv):
        problem,p_conv,vU_conv=problem_p_vU_conv 
        pnumbers_old = problem.pnumber
        problem.solve_parameter_likelihood(self.reduced_spectra, self.flags, return_DeltaH=1)
        print(pnumbers_old, problem.pnumber)

        if(problem.pnumber==pnumbers_old):
            p_conv = True
        else:
            p_conv = False
            vU_conv = False
        return [problem,p_conv,vU_conv]
   
    def update_alpha(self,alpha):
        for problem in self.problems:
            problem.update_alpha(alpha) 
    
    def solve(self,convtol=0.1):
#        alpha_para=[0.01,0.1,0.5,1.0,1.5]
        f = open('outfile.txt','w')
        alpha = 1.e-5
        kk=0
        f.write('update alpha\n')
        self.update_alpha(alpha)
        f.write('start maximum likelihood calculation\n')
        f.close()
        while((not all(self.p_convergence)) and kk<5):
            f = open('outfile.txt','a')
            f.write(str(kk+1) + ' loop with alpha = '+str(alpha)+'\n')
            f.close()
#            print(self.p_convergence)
#            with Pool(self.ncpus, maxtasksperchild=10) as pool:
            with get_context("spawn").Pool(self.ncpus) as pool:
##            pool = Pool(self.ncpus)
#                results=pool.map_async(unwrap_self_updata_para, zip([self]*len(self.problems), list(zip(self.problems,self.p_convergence,self.vU_convergence))))
#                results.wait()
#                self.problems,self.p_convergence,self.vU_convergence=np.asarray(results.get()).T
                self.problems,self.p_convergence,self.vU_convergence=np.asarray(pool.map(unwrap_self_updata_para, zip([self]*len(self.problems), list(zip(self.problems,self.p_convergence,self.vU_convergence))))).T
#                pool.close()
##            print(self.p_convergence)
##            print(self.vU_convergence)
##            self._update_para_singleCPU()
            f = open('outfile.txt','a')
            f.write('update v and U\n')
            f.close()
            count = 0
            while((not all(self.vU_convergence)) and count < 5):
                self._update_v_U(convtol,alpha)
                count += 1
#            alpha = alpha_para[kk]
            if not all(self.vU_convergence): print('vU update did not converged')
            alpha += 0.35
            print("Alpha",alpha)
            kk += 1
            alpha = min(1.0,alpha)
            f = open('outfile.txt','a')
            f.write('update alpha\n')
            f.close()
            self.update_alpha(alpha) 
            f = open('outfile.txt','a')
            f.write('next loop or finished\n')
            f.close()
        
        print("Alpha",alpha)
        print('solved!!')
        f = open('outfile.txt','a')
        f.write('Solved\n')
        f.close()
#        print 'final parameter number', self.problem.pnumber 
#        print 'final input parameter number', self.targets[0]
#        return self.problem.U.val 
        
def unwrap_self_updata_para(arg, **kwarg):
    return many_stars._update_para(*arg, **kwarg)
def unwrap_self_prepare_input(arg, **kwarg):
    return many_stars.prepare_input(*arg, **kwarg)
    
if __name__ == '__main__':
    linalgtime = ttt.time()
    test = many_stars(path,20)
#    print("Calculation of many_stars took",ttt.time()-linalgtime,"seconds.")
#    linalgtime = ttt.time()
    test.solve()
#    print("Solve took",ttt.time()-linalgtime,"seconds.")
    print("Preparation of many stars and solving it took",(ttt.time()-linalgtime)/60.,"minutes.")
    
#    linalgtime = ttt.time()
#    test = many_stars(path,cut=False)
#    print("Calculation of many_stars took",ttt.time()-linalgtime,"seconds.")
#    linalgtime = ttt.time()
#    test.solve()
#    print("Solve took",ttt.time()-linalgtime,"seconds.")
    prob = np.zeros((test.flags.size,test.problems.size))
    g = open('XShootU_results_19sep23_a_SMC_eDR2.csv','w')
    g.write('ID,model,Teff,logg,Y,Mdot_t,Q,beta,vsini,vmac\n')
    for i,aproblem in enumerate(test.problems):
        print(aproblem.name)
        probability=np.exp(-aproblem.deltaH)/sum(np.exp(-aproblem.deltaH))
        prob[:,i] = probability
#Slice through the parmeter space
#Normalise probability distribution: y1=y/scipy.integrate.trapz(y,x)
#Cumulative integration over array including zero as initial condition: y_cum = scipy.integrate.cumtrapz(y1,x,initial=0.)
#Chose confidence levels and obtain parameters with uncertainties: np.interp(0.159,x,y_cum), np.interp(0.50,x,y_cum), np.interp(0.841,x,y_cum)
        #plt.figure(figsize=(11.69, 8.27), dpi=300)
        #obs_flux=aproblem.A.dot(aproblem.data.val)
        #mod_flux=aproblem.A.dot(aproblem.reduced_spectra[aproblem.pnumber])
        #plt.plot(test.wave,obs_flux,'b-',lw=1.0)
        #plt.plot(test.wave,mod_flux,'r--',lw=0.8)
        #plt.title(aproblem.name+': '+aproblem.flags[aproblem.pnumber])
        #pdf_output.savefig(orientation='landscape')
        #plt.clf()
        #plt.close()
        g.write(aproblem.name+','+test.flags[aproblem.pnumber]+ '\n')
    g.close()
    # Save the file
#    pickle.dump(test, file = open("test.pickle", "wb"))
#    test_loaded=pickle.load(open("test.pickle", "rb"))
    sig=test.problems[0].A.dot(test.problems[0].U)
    sig=sig.dot(test.problems[0].A.T)
    np.save("error_co-variance_19sep23_a_SMC_eDR2.npy", sig)
    np.save("probabilities_19sep23_a_SMC_eDR2.npy", prob)
    #sigma=np.sqrt(sig.diagonal())
    #pdf_output = PdfPages('plots.pdf')
    #plt.figure(figsize=(11.69, 8.27), dpi=300)
    #plt.fill_between(test.wave, 1-sigma, 1+sigma)
    #pdf_output.savefig(orientation='landscape')
    #plt.close()
    #pdf_output.close()
