from scipy.optimize import minimize, BFGS
from scipy.stats import sem, iqr, wilcoxon, mannwhitneyu, ttest_ind, median_absolute_deviation
from scipy.stats.distributions import chi2
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import NMF, PCA, SparsePCA 
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn import metrics
from scipy.stats import chisquare

import numpy as np, matplotlib.pyplot as plt, pandas as pd 
import csv, pdb, itertools, os, pickle, sys, random
import pingouin as pg
import multiprocessing as mp

import statsmodels.formula.api as smf 
from scipy.stats import pearsonr, spearmanr


eps=np.finfo(float).eps


def my_bincount(x,bins):
   hist=np.zeros((len(bins),))
   inds, val = np.unique(x,return_counts=True)
   hist[inds]=val
   
   return hist


def print_fit_scores(k, n, ll, acc, type):

   print(type)
   print('k=%d, N=%d'%(k, n))
   print('LL=%.6f'%ll)
   print('AIC_c=%.6f'%(-2*ll + 2*k+(2*k+1)/(n-k-1) ))
   print('BIC=%.6f'%(  -2*ll + k*np.log(n) ))
   print('ACC_c=%.6f'%( 100*acc ) )




def u(x,lambd,rho):

   if x > 0:
      return x**rho
   else:
      return -lambd*(-x)**rho


def p_gamble_accept(x_gain, x_loss, x_sure, mu, rho, lambd):


   return 1.0/(1.0 + np.exp( -mu*( 0.5*(    u(x_gain,lambd,rho) \
                                          + u(x_loss,lambd,rho))\
                                          - u(x_sure,lambd,rho) ) ) )


def neg_log_likelihood(params, args):
   global mufix

   X,model=args

   # Set parameters according to model

   if model == 'lambda_mu':
      rho   = 1
      mu,lambd=params
   
   elif model == 'rho_mu':
      lambd = 1
      mu,rho=params

   elif model == 'mu':
      rho   = 1
      lambd = 1
      mu = params[0]

   elif model == 'full':
      mu,rho,lambd=params      

   elif model == 'full_mufix':
      mu = mufix 
      rho,lambd =params

   elif model == 'lambda_mufix':
      mu = mufix
      lambd =params[0]
      rho=1

   elif model == 'rho_mufix':
      mu = mufix
      rho =params[0]
      lambd=1


   ll=0
   for x_ii in X:

      x_ii = np.squeeze(x_ii)
   
      x_gain = x_ii[0]
      x_loss = x_ii[1]
      x_sure = x_ii[2]
      y = x_ii[3]

      p_ii = p_gamble_accept(x_gain, x_loss, x_sure, mu, rho, lambd)

      ll += y*np.log(p_ii+eps) + (1-y)*np.log(1-p_ii+eps)


   return -ll




def log_likelihood_all(params,data,model,k):

   np.random.seed(11901)

   ll=[]
   aic=[]
   aic_c=[]
   bic=[]
   acc=[]

   if model == 'null_1' or model == 'null_2':
      params=np.zeros((len(data.keys()),))

   for ii,p_ii in zip(data.keys(),params):

      if np.any(np.isnan(p_ii)):
         ll.append(np.nan)
         aic.append(np.nan)
         aic_c.append(np.nan)
         bic.append(np.nan)
         acc.append(np.nan)
         continue

      # Set parameters according to model

      if model == 'lambda_mu':
         rho   = 1
         mu,lambd=p_ii
   
      elif model == 'rho_mu':
         lambd = 1
         mu,rho=p_ii

      elif model == 'mu':
         rho   = 1
         lambd = 1
         mu = p_ii[0]

      elif model == 'full':
         mu,rho,lambd=p_ii      

      elif model == 'full_mufix':
         mu = mufix
         rho,lambd =p_ii

      elif model == 'lambda_mufix':
         mu = mufix
         lambd =p_ii[0]
         rho=1

      elif model == 'rho_mufix':
         mu = mufix
         rho =p_ii[0]
         lambd=1


      X = np.array([jj[0][0] for jj in data[ii]]).astype('float')

      ll_ii = 0
      out_ii = []
      

      null_p1 = np.nanmean(X[:,3])

      for x_ii in X:


         x_ii = np.squeeze(x_ii)
   
         x_gain = x_ii[0]
         x_loss = x_ii[1]
         x_sure = x_ii[2]
         y = x_ii[3]


         if model == 'null_1':
            p_ii = null_p1
         elif model == 'null_2':
            p_ii = 0.5
         else:
            p_ii = p_gamble_accept(x_gain, x_loss, x_sure, mu, rho, lambd)


         ll_ii += (y*np.log(p_ii+eps) + (1-y)*np.log(1-p_ii+eps))

         out_ii.append(np.random.binomial(1, p=p_ii)==y)         


      n = len(X)
      
      ll.append(ll_ii)
      aic.append(-2*ll_ii + 2*k)
      aic_c.append(-2*ll_ii + 2*k + (2*k**2 + 2*k)/(n-k-1) )
      bic.append(-2*ll_ii + k*np.log(n))
      acc.append(np.nanmean(out_ii))
      

   return ll,aic,aic_c,bic,acc




def ml_fit_data(x0,data,solver,model):

   global mufix

   params=[]
   fun=[]
   groups=[]
   cond=[]

       
   if model == 'lambda_mu':
      bounds=[(0,np.inf),(0,np.inf)]

   elif model == 'rho_mu':
      bounds=[(0,np.inf),(0,1)]

   elif model == 'mu':
      bounds=[(0,np.inf)]

   elif model == 'full':
      bounds=[(0,np.inf),(0,1),(0,np.inf)]

   elif model == 'full_mufix':
      bounds=[(0,1),(0,np.inf)]

   elif model == 'lambda_mufix':
      bounds=[(0,np.inf)]

   elif model == 'rho_mufix':
      bounds=[(0,1)]


   for ii in data.keys():

      X = np.array([jj[0][0] for jj in data[ii]]).astype('float')
      res = minimize(fun=neg_log_likelihood, x0=x0, args=[X,model], method=solver, bounds=bounds)


      params += [res.x]
      
      if res.success and not np.any(np.isnan(res.x)):
         fun += [res.fun]
      else:
         fun += [np.nan]

      groups += data[ii][1]
      cond += data[ii][2]

   #params=np.array(params)
   #fun = np.array(fun)

   return params, fun, groups, cond




####################################################################################
# This part of the code was originally written to run on multiple processors. The
# multiprocessing part is commented below. If you run this on a single-processor
# it may take a lot of time.
####################################################################################

# Callback
def log_fits(out_fun):
  global out
  out.append(out_fun)


# Fit function
def hpc_fit(data, x0, solver, model, file):
   global out

   if not os.path.exists(file):

      out=[]
    
      pool = mp.Pool(processes=58)
      for ind,param in enumerate(x0):
         print(ind)
         pool.apply_async(ml_fit_data, (param, data, solver, model),callback=log_fits)
         #ml_fit_data(param, data, solver, model)
      pool.close()
      pool.join()

      pickle.dump(out, open(file,'wb'))

      print ('Saved file')

   else:
      out=pickle.load(open(file,'rb'),encoding='latin')

   return out



def get_val_data(grp, cond,offset=''):

   data = {}
   with open('raw_data_la1.csv') as file:

      reader=csv.reader(file)
      next(reader)
      for ii in reader:
         ii[0] = ii[0].strip() + offset

         if not ii[-1] in grp:
            continue

         if ii[1].strip() in cond:
            if ii[0] not in data:
               data[ii[0]]=[]
            data[ii[0]].append(([[ii[2]]+['-'+ii[3]]+[0]+[int(ii[5] == 'y')]], ii[1].strip(), ii[-1]))

   return data



def med_opt_params(fits, thr):


   out = [(ii[0],ii[1]) for ii in fits]

   params,f=zip(*out)

   print(type(f))
   
   #z=[len(ii.shape) for ii in f]
   #print(z)
   #print(all(z))

   try:
      f = np.array(f)
   except:
      pdb.set_trace()

   if len(f.shape) > 2:
      print(f.shape)
      f=np.squeeze(f)
      pdb.set_trace()
 


   params=np.array(params)
   opt_params = []
   sel=[]

   for kk in f.T:
      try:
         sel.append (kk <= np.nanpercentile(kk,thr))
      except:
         pdb.set_trace()   

   opt_params = np.array([np.nanmean(params[ii,ss],0) if len(ii) > 0 else np.nan*3 for ss,ii in enumerate(sel)])

   ll=[np.nanmean(f[:,ii][sel[ii]]) for ii in range(len(sel))]


   return opt_params, np.array(ll)



def get_data(grp, cond, offset=''):

   data = {}

   groups={}

   # Assign LA2 subjects to the right groups
   with open('previous_params.csv') as file:
      reader=csv.reader(file)
      next(reader)
      for ii in reader:
         ii[0] = ii[0].strip() + offset
         groups[ii[0]] = ii[1].strip()+offset

   with open('raw_data.csv') as file:

      reader=csv.reader(file)
      next(reader)
      for ii in reader:
         ii[0] = ii[0].strip() + offset
         if not groups[ii[0]] in grp:
            continue

         if ii[4].strip() in cond:
            if ii[0] not in data:
               data[ii[0]]=[]

            if int(ii[-1] == 'y') or int(ii[-1] == 'n') and 'no' in ii[9]:
               data[ii[0]].append(([ii[5:8]+[int(ii[-1] == 'y')]], ii[4].strip(), groups[ii[0]]))



   return data



if __name__ == '__main__' :
   
 global mufix

 solver = sys.argv[1] #nelder-mead, bfgs, lbfgs, etc
 task = sys.argv[2] #LA1, LA2

 # Create grid x0 of initial solutions for parameters
 mu_ini = np.linspace(0,5,10)
 rho_ini = np.linspace(0,1,10)
 lambd_ini = np.linspace(0,5,10)


 # Get the data
 data_healthy_safe_LA2    = get_data(['healthy'],['safe'])
 data_healthy_threat_LA2  = get_data(['healthy'],['threat'])
 data_patients_safe_LA2   = get_data(['patient'],['safe'])
 data_patients_threat_LA2 = get_data(['patient'],['threat'])

 data_healthy_safe_LA1    = get_val_data('healthy','safe',offset='LA1')
 data_healthy_threat_LA1  = get_val_data('healthy','threat',offset='LA1')
 data_patients_safe_LA1   = get_val_data('patient','safe',offset='LA1')
 data_patients_threat_LA1 = get_val_data('patient','threat',offset='LA1')


 configs = list(itertools.product(['full','lambda_mu','rho_mu','mu','full_mufix','lambda_mufix','rho_mufix','null_1','null_2'],
                  ['per_cond','both_cond']))

 #configs = list(itertools.product(['full_mufix','lambda_mufix','rho_mufix','null_1','null_2'],
 #                 ['per_cond','both_cond']))
 

 dic_ll_config = [{} for ii in range(len(configs))]
 dic_aic_config = [{} for ii in range(len(configs))]
 dic_aic_c_config = [{} for ii in range(len(configs))]
 dic_bic_config = [{} for ii in range(len(configs))]
 dic_acc_config = [{} for ii in range(len(configs))]


 # Load or determine mufix
 try:
         mufix = np.load('median_mu_%s_%s.npy'%(solver,task),allow_pickle=True)[0]
 except:
 
         print ('Mufix pickle not found. Running (mu,both_cond) configuration first to obtain it')

         
         if task == 'LA1':
            data_healthy_safe    = data_healthy_safe_LA1
            data_healthy_threat  = data_healthy_threat_LA1
            data_patients_safe   = data_patients_safe_LA1
            data_patients_threat = data_patients_threat_LA1

            data_healthy = {**data_healthy_safe_LA1, **data_healthy_threat_LA1}
            data_patients = {**data_patients_safe_LA1, **data_patients_threat_LA1}

         elif task == 'LA2':
            data_healthy_safe    = data_healthy_safe_LA2
            data_healthy_threat  = data_healthy_threat_LA2
            data_patients_safe   = data_patients_safe_LA2
            data_patients_threat = data_patients_threat_LA2

            data_healthy = {**data_healthy_safe_LA2, **data_healthy_threat_LA2}
            data_patients = {**data_patients_safe_LA2, **data_patients_threat_LA2}

         #Fit models
         print('fits_patients_%s_%s_m=%s.pkl'%(task,solver,'mu'))
         print('fits_healthy_%s_%s_m=%s.pkl'%(task,solver,'mu'))

         x0 = mu_ini
         pdb.set_trace()
         patients = hpc_fit(data_patients, x0, solver, 'mu', 'fits_patients_%s_%s_m=%s.pkl'%(task,solver,'mu'))
         healthy = hpc_fit(data_healthy, x0, solver, 'mu', 'fits_healthy_%s_%s_m=%s.pkl'%(task,solver,'mu'))

         params_patients,_  = med_opt_params(patients, 5)
         params_healthy,_   = med_opt_params(healthy, 5)

         mufix = np.nanmean(np.vstack([params_healthy,params_patients]),0)
         np.save('median_mu_%s_%s.npy'%(solver,task),mufix,allow_pickle=True)



 for ind,(model, cond) in enumerate(configs):


   print ('>>>>',model,cond)
   if model == 'null_1' or model ==  'null_2':

      if task == 'LA1':
         data_healthy_safe    = data_healthy_safe_LA1
         data_healthy_threat  = data_healthy_threat_LA1
         data_patients_safe   = data_patients_safe_LA1
         data_patients_threat = data_patients_threat_LA1

         data_healthy = {**data_healthy_safe_LA1, **data_healthy_threat_LA1}
         data_patients = {**data_patients_safe_LA1, **data_patients_threat_LA1}

      elif task == 'LA2':
         data_healthy_safe    = data_healthy_safe_LA2
         data_healthy_threat  = data_healthy_threat_LA2
         data_patients_safe   = data_patients_safe_LA2
         data_patients_threat = data_patients_threat_LA2

         data_healthy = {**data_healthy_safe_LA2, **data_healthy_threat_LA2}
         data_patients = {**data_patients_safe_LA2, **data_patients_threat_LA2}


      ll_h, aic_h, aic_c_h, bic_h, acc_h = log_likelihood_all(None,  data_healthy, model, 0)
      ll_a, aic_a, aic_c_a, bic_a, acc_a = log_likelihood_all(None, data_patients, model, 0)

      for ii,key in enumerate(data_healthy.keys()):
         dic_ll_config[ind][key] = ll_h[ii]
         dic_aic_config[ind][key] = aic_h[ii]
         dic_aic_c_config[ind][key] = aic_c_h[ii]
         dic_bic_config[ind][key] = bic_h[ii]
         dic_acc_config[ind][key] = acc_h[ii]

      for ii,key in enumerate(data_patients.keys()):
         dic_ll_config[ind][key] = ll_a[ii]
         dic_aic_config[ind][key] = aic_a[ii]
         dic_aic_c_config[ind][key] = aic_c_a[ii]
         dic_bic_config[ind][key] = bic_a[ii]
         dic_acc_config[ind][key] = acc_a[ii]

      continue



   if model == 'full':
      k = 3*2
      x0 = np.array(list(itertools.product(mu_ini,rho_ini,lambd_ini)))
   elif model == 'lambda_mu':
      k = 2*2
      x0 = np.array(list(itertools.product(mu_ini,lambd_ini)))
   elif model == 'rho_mu':
      k = 2*2
      x0 = np.array(list(itertools.product(mu_ini,rho_ini)))
   elif model == 'mu':
      k = 1*2
      x0 = mu_ini
   elif model == 'full_mufix':
      x0 = np.array(list(itertools.product(rho_ini,lambd_ini)))
      k = 2*2
   elif model == 'lambda_mufix':
      x0 = lambd_ini
      k = 1*2
   elif model == 'rho_mufix':
      x0 = rho_ini
      k = 1*2
   
   if cond == 'per_cond':
      k *= 2
   
   if 'mufix' in model:
      k +=1

   print('\n*model=%s, cond=%s, task=%s, k=%d\n'%(model, cond, task, k))


   if cond == 'per_cond':

         if task == 'LA1':
            data_healthy_safe    = data_healthy_safe_LA1
            data_healthy_threat  = data_healthy_threat_LA1
            data_patients_safe   = data_patients_safe_LA1
            data_patients_threat = data_patients_threat_LA1
         elif task == 'LA2':
            data_healthy_safe    = data_healthy_safe_LA2
            data_healthy_threat  = data_healthy_threat_LA2
            data_patients_safe   = data_patients_safe_LA2
            data_patients_threat = data_patients_threat_LA2

         print ('fits_patients_safe_%s_%s_m=%s.pkl'%(task,solver,model))
         print ('fits_patients_threat_%s_%s_m=%s.pkl'%(task,solver,model))
         print ('fits_healthy_safe_%s_%s_m=%s.pkl'%(task,solver,model))
         print ('fits_healthy_threat_%s_%s_m=%s.pkl'%(task,solver,model))

         # Fit models

         patients_safe = hpc_fit(data_patients_safe, x0, solver,model,
         'fits_patients_safe_%s_%s_m=%s.pkl'%(task,solver,model))

         patients_threat = hpc_fit(data_patients_threat, x0, solver,model,
         'fits_patients_threat_%s_%s_m=%s.pkl'%(task,solver,model))

         healthy_safe = hpc_fit(data_healthy_safe, x0, solver,model,
         'fits_healthy_safe_%s_%s_m=%s.pkl'%(task,solver,model))

         healthy_threat = hpc_fit(data_healthy_threat, x0, solver,model,
         'fits_healthy_threat_%s_%s_m=%s.pkl'%(task,solver,model))

         params_healthy_safe,_    = med_opt_params(healthy_safe, 5)
         params_healthy_threat,_  = med_opt_params(healthy_threat, 5)
         params_patients_safe,_   = med_opt_params(patients_safe, 5)
         params_patients_threat,_ = med_opt_params(patients_threat, 5)

 
         # Model fits (2 sets of model fits per participant)

         ll_hs, aic_hs, aic_c_hs, bic_hs, acc_hs = log_likelihood_all(params_healthy_safe,  data_healthy_safe, model, k)
         ll_ht, aic_ht, aic_c_ht, bic_ht, acc_ht = log_likelihood_all(params_healthy_threat,data_healthy_threat, model, k)
         ll_as, aic_as, aic_c_as, bic_as, acc_as = log_likelihood_all(params_patients_safe,  data_patients_safe, model, k)
         ll_at, aic_at, aic_c_at, bic_at, acc_at = log_likelihood_all(params_patients_threat,data_patients_threat, model, k)


         for ii,key in enumerate(data_healthy_safe.keys()):
            dic_ll_config[ind][key] = ll_hs[ii] + ll_ht[ii]
            dic_aic_config[ind][key] = aic_hs[ii] + aic_ht[ii] 
            dic_aic_c_config[ind][key] = aic_c_hs[ii] + aic_c_ht[ii] 
            dic_bic_config[ind][key] = bic_hs[ii] + bic_ht[ii]
            dic_acc_config[ind][key] = (acc_hs[ii] + acc_ht[ii])/2

         for ii,key in enumerate(data_patients_safe.keys()):
            dic_ll_config[ind][key] = ll_as[ii] + ll_at[ii]
            dic_aic_config[ind][key] = aic_as[ii] + aic_at[ii] 
            dic_aic_c_config[ind][key] = aic_c_as[ii] + aic_c_at[ii] 
            dic_bic_config[ind][key] = bic_as[ii] + bic_at[ii]
            dic_acc_config[ind][key] = (acc_as[ii] + acc_at[ii])/2


   elif cond == 'both_cond':

         if task == 'LA1':
            data_healthy_safe    = data_healthy_safe_LA1
            data_healthy_threat  = data_healthy_threat_LA1
            data_patients_safe   = data_patients_safe_LA1
            data_patients_threat = data_patients_threat_LA1

            data_healthy = {**data_healthy_safe_LA1, **data_healthy_threat_LA1}
            data_patients = {**data_patients_safe_LA1, **data_patients_threat_LA1}

         elif task == 'LA2':
            data_healthy_safe    = data_healthy_safe_LA2
            data_healthy_threat  = data_healthy_threat_LA2
            data_patients_safe   = data_patients_safe_LA2
            data_patients_threat = data_patients_threat_LA2

            data_healthy = {**data_healthy_safe_LA2, **data_healthy_threat_LA2}
            data_patients = {**data_patients_safe_LA2, **data_patients_threat_LA2}


         #Fit models
         print('fits_patients_%s_%s_m=%s.pkl'%(task,solver,model))
         print('fits_healthy_%s_%s_m=%s.pkl'%(task,solver,model))

         patients = hpc_fit(data_patients, x0, solver,model,'fits_patients_%s_%s_m=%s.pkl'%(task,solver,model))
         healthy = hpc_fit(data_healthy, x0, solver,model,'fits_healthy_%s_%s_m=%s.pkl'%(task,solver,model))


         params_patients,_  = med_opt_params(patients, 5)
         params_healthy,_   = med_opt_params(healthy, 5)


         # Model fits (1 sets of model fits per participant)
         ll_h, aic_h, aic_c_h, bic_h, acc_h = log_likelihood_all(params_healthy,  data_healthy, model, k)
         ll_a, aic_a, aic_c_a, bic_a, acc_a = log_likelihood_all(params_patients, data_patients, model, k)

         for ii,key in enumerate(data_healthy.keys()):
            dic_ll_config[ind][key] = ll_h[ii]
            dic_aic_config[ind][key] = aic_h[ii]
            dic_aic_c_config[ind][key] = aic_c_h[ii]
            dic_bic_config[ind][key] = bic_h[ii]
            dic_acc_config[ind][key] = acc_h[ii]

         for ii,key in enumerate(data_patients.keys()):
            dic_ll_config[ind][key] = ll_a[ii]
            dic_aic_config[ind][key] = aic_a[ii]
            dic_aic_c_config[ind][key] = aic_c_a[ii]
            dic_bic_config[ind][key] = bic_a[ii]
            dic_acc_config[ind][key] = acc_a[ii]



   ###############################################################################################################
 
 z_ll=np.vstack([list(ii.values()) for ii in dic_ll_config])
 z_aic=np.vstack([list(ii.values()) for ii in dic_aic_config])
 z_aic_c=np.vstack([list(ii.values()) for ii in dic_aic_c_config])
 z_bic=np.vstack([list(ii.values()) for ii in dic_bic_config])
 z_acc=np.vstack([list(ii.values()) for ii in dic_acc_config])



 print('LL')
 print(my_bincount(np.nanargmin(-z_ll,0),range(len(configs))))

 print('AIC')
 print (my_bincount(np.nanargmin(z_aic,0),range(len(configs))))
 print('AIC_c')
 print (my_bincount(np.nanargmin(z_aic_c,0),range(len(configs))))
 print('BIC')
 print(my_bincount(np.nanargmin(z_bic,0),range(len(configs))))
 print('ACC')
 print(my_bincount(np.nanargmax(z_acc,0),range(len(configs))))

 print ('------------')


 print ('LL     = ' ,np.nansum(z_ll,1))
 print ('AIC    = ' ,np.nansum(z_aic,1))
 print ('AIC_c  = ' ,np.nansum(z_aic_c,1))
 print ('BIC    = ' ,np.nansum(z_bic,1))
 print ('ACC    = ' ,np.nanmean(z_acc,1))

 print ('------------')

 print( 'N H     = ',len(params_healthy))
 print( 'N A     = ',z_ll[:,len(params_healthy):].shape[1])
 print ('LL H    = ' ,np.nansum(z_ll[:,0:len(params_healthy)],1))
 print ('LL A    = ' ,np.nansum(z_ll[:,len(params_healthy):],1))
 print ('AIC H   = ' ,np.nansum(z_aic[:,0:len(params_healthy)],1))
 print ('AIC A   = ' ,np.nansum(z_aic[:,len(params_healthy):],1))
 print ('AIC_c H = ' ,np.nansum(z_aic_c[:,0:len(params_healthy)],1))
 print ('AIC_c A = ' ,np.nansum(z_aic_c[:,len(params_healthy):],1))
 print ('BIC H   = ' ,np.nansum(z_bic[:,0:len(params_healthy)],1))
 print ('BIC A   = ' ,np.nansum(z_bic[:,len(params_healthy):],1))
 print ('ACC H   = ' ,np.nanmean(z_acc[:,0:len(params_healthy)],1))
 print ('ACC A   = ' ,np.nanmean(z_acc[:,len(params_healthy):],1))


 print('Healthy table')
 htable = np.vstack((np.nansum(z_ll[:,0:len(params_healthy)],1),
            np.nansum(z_aic[:,0:len(params_healthy)],1),
            np.nansum(z_aic_c[:,0:len(params_healthy)],1),
            np.nansum(z_bic[:,0:len(params_healthy)],1),
            100*np.nanmean(z_acc[:,0:len(params_healthy)],1),
            100*np.nanstd(z_acc[:,0:len(params_healthy)],1))) 

 print(np.array2string(htable.T[:,0:-2], precision=0, separator='\t'))
 print(np.array2string(100*np.nanmean(z_acc[:,0:len(params_healthy)],1), precision=1, separator='\n'))
 print(np.array2string(100*np.nanstd(z_acc[:,0:len(params_healthy)],1), precision=1, separator='\n'))

 print('Anx table')
 atable = np.vstack((np.nansum(z_ll[:,len(params_healthy):],1),
            np.nansum(z_aic[:,len(params_healthy):],1),
            np.nansum(z_aic_c[:,len(params_healthy):],1),
            np.nansum(z_bic[:,len(params_healthy):],1),
            100*np.nanmean(z_acc[:,len(params_healthy):],1),
            100*np.nanstd(z_acc[:,len(params_healthy):],1)))

 print(np.array2string(atable.T[:,0:-2], precision=1, separator='\t'))
 print(np.array2string(100*np.nanmean(z_acc[:,len(params_healthy):],1), precision=1, separator='\n'))
 print(np.array2string(100*np.nanstd(z_acc[:,len(params_healthy):],1), precision=1, separator='\n'))


 pdb.set_trace()
 print('****************************************************')
 print('Table %d'%(5 if task == 'LA1' else 6))
 for ii in zip(htable.T[:,0:-2],htable.T[:,-2:],atable.T[:,0:-2],atable.T[:,-2:]):
    print(np.array2string(ii[0],precision=0, separator='\t &').replace('.','').replace('[','').replace(']','') + '\t &%.1f (%.1f)'%(ii[1][0],ii[1][1]) +'\t &'+\
          np.array2string(ii[2],precision=0, separator='\t &').replace('.','').replace('[','').replace(']','') + '\t &%.1f (%.1f)'%(ii[3][0],ii[3][1]))



 print ('------------')


 print(configs)


 if task == 'LA1':
    data_healthy_safe    = data_healthy_safe_LA1
    data_healthy_threat  = data_healthy_threat_LA1
    data_patients_safe   = data_patients_safe_LA1
    data_patients_threat = data_patients_threat_LA1

    data_healthy = {**data_healthy_safe_LA1, **data_healthy_threat_LA1}
    data_patients = {**data_patients_safe_LA1, **data_patients_threat_LA1}

 elif task == 'LA2':
    data_healthy_safe    = data_healthy_safe_LA2
    data_healthy_threat  = data_healthy_threat_LA2
    data_patients_safe   = data_patients_safe_LA2
    data_patients_threat = data_patients_threat_LA2

    data_healthy = {**data_healthy_safe_LA2, **data_healthy_threat_LA2}
    data_patients = {**data_patients_safe_LA2, **data_patients_threat_LA2}

