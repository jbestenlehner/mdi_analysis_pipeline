#from nifty import sqrt
import numpy as np
#from .response import *
#from nifty_local import explicit_operator, field
import time as ttt



def minimize_parameters(Hamiltonian,parameter_flags,return_DeltaH=0):
    
    LL = len(parameter_flags)
    
    Hvals = np.ones(LL)
    
    for ii in range(LL):
        if(parameter_flags[ii]):
            
            Hvals[ii] = Hamiltonian(ii)
    
    ret = np.where(Hvals==Hvals.min())[0][0]

    if(return_DeltaH):
        Hvals = Hvals-Hvals.min()
        return ret, Hvals
    else:
        return ret


class single_star_problem(object):
    
#    def __init__(self,model_space,reduced_spectra,flags,reduced_data,name,var,pnumber,alpha=1.,A=None):
    def __init__(self,reduced_spectra,flags,reduced_data,name,var,pnumber,alpha=1.,A=None):
        
#        self.model_space = model_space
        self.var = var
#        self.reduced_spectra = reduced_spectra
#        self.flags = flags
        self.name = name
        if(flags is None):
#            self.flags = np.ones(self.reduced_spectra.shape[0])
            raise ValueError("Please provide synthetic model spectra identifier, e.g. array")
        
        self.A = A
        #convert data into nifty field
        #self.data = field(self.model_space,val=reduced_data)
        self.data = reduced_data

        #create uncorrelated error matrix using data variance A^daggerNA
        #if(self.A.shape[1] != self.model_space.dim()):
        if(self.A.shape[1] != reduced_spectra.shape[1]):
            raise ValueError("shape of A not compatible with grid shape.\ A.shape =%d, %d"%(A.shape[0],A.shape[1]))
        N = np.ones(self.A.shape[0])*self.var
#        N = np.ones(self.A.shape[0])*self.var + np.load('/local2/bestenlehner/XShootU_ift_sp_py3/mod_dat/v_vec.npy')
        self.N = np.einsum("ji,j,jk",A,N,A)
#        self.N = explicit_operator(self.model_space, sym=True, uni=False, matrix=N)
        
        self.stt = ttt.time()
#        self.j = self.N.inverse_times(self.data)
        self.j = np.linalg.inv(self.N).dot(self.data)
        self.stt = ttt.time()
                
        self.v = None
        self.U = None

        self.vdist = None
        self.Udist = None
        
        self.pnumber = None
        self.t = None
        self.deltaH = None
        
        self.delta = None
        self.D = None
        #self.div = None

        self.alpha = alpha
        
        self.update_parameter(pnumber,self.deltaH,reduced_spectra)        


    def update_alpha(self,alpha):
        self.alpha = alpha

    def update_parameter(self,pnumber,deltaH,reduced_spectra):
        
        self.pnumber = pnumber
        self.deltaH = deltaH
        #self.t = field(self.model_space,val=reduced_spectra[pnumber])
        self.t = reduced_spectra[pnumber]
        
        self._calc_delta_D()        
    
    
    def _initial_guess(self):

        #self.v = field(self.model_space,val=0)
        self.v = 0
#        self.v = 0.1*np.load('/local2/bestenlehner/XShootU_ift_sp_py3/mod_dat/v_vec.npy').dot(self.A)
        #self.U = explicit_operator(self.model_space,matrix=np.diag(np.ones(self.model_space.dim())))
        self.U = 0.5*self.N
    
    def _calc_delta_D(self):
        #calculates the posterior uncertainty
        
        if((self.U is None) or (self.v is None)):
            self._initial_guess()
        
#        self.D = (self.U.inverse()+self.N.inverse()).inverse()
        self.D = np.linalg.inv(np.linalg.inv(self.U)+np.linalg.inv(self.N))

        #self.delta = self.D(self.j-self.N.inverse_times(self.t+self.v))
        self.delta = self.D.dot(self.j-np.linalg.inv(self.N).dot(self.t+self.v))
        #self.div = (self.A.dot(self.data)/self.A.dot(self.t)-1).dot(self.A)

    
    def update_v_U(self,v,U):
        
        #self.vdist = np.sqrt((v-self.v).dot(v-self.v))     # distance
        #self.Udist = np.sqrt(((self.U-U).transpose()*(self.U-U)).tr()) #Frobenius distance
        self.vdist = np.sqrt((v-self.v).dot(v-self.v))     # distance
        self.Udist = np.sqrt(((self.U-U).transpose().dot(self.U-U)).trace()) #Frobenius distance
        
        self.v = v
        self.U = U
        
        self._calc_delta_D()
        
        return self.vdist,self.Udist
        
    
    def solve_parameter_likelihood(self, reduced_spectra, flags, return_DeltaH = 0):
        
        print('preparing Hamiltonian...')
        abctime = ttt.time()        
        #computes error covariance 
        print("Alpha:",self.alpha)
        linalgtime = ttt.time()
        M = self.U*self.alpha + self.N #U already multiplied by alpha in _v_U_update?
#        M.val *= self.alpha
        print("calculating M took",ttt.time()-linalgtime,"seconds")
        linalgtime = ttt.time()
        #Minv = M.inverse()
        Minv = np.linalg.inv(M)
        print("calculating M^-1 took",ttt.time()-linalgtime,"seconds.")
       
        #Minv_d = Minv(self.data)
        Minv_d = Minv.dot(self.data)
 
        print('done!')
        print('it took ',ttt.time()-abctime,' seconds.')
        
        def Hamiltonian(pnumber):
            #t = field(self.model_space,val=reduced_spectra[pnumber])
            t = reduced_spectra[pnumber]
            
        #    return 0.5*(self.v+t).dot(Minv(self.v+t)) - (self.v+t).dot(Minv_d)
            return 0.5*(self.v+t).dot(Minv.dot(self.v+t)) - (self.v+t).dot(Minv_d)
            
        print('solving Hamiltonian for all parameters...')
        abctime = ttt.time()
        if(return_DeltaH):
            pp,DeltaH = minimize_parameters(Hamiltonian,flags,return_DeltaH=1)
        else:
            pp = minimize_parameters(Hamiltonian,flags,return_DeltaH=0)
        print('done!')
        print('it took ',ttt.time()-abctime,' seconds.')
        
        self.update_parameter(pp,DeltaH,reduced_spectra)
        
        print("found parameter",pp,"Alpha:",self.alpha)
        
#        if(return_DeltaH):
#            return DeltaH
#        else:
#            return None

