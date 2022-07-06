import csv, os, pickle, sys, random, pdb
import numpy as np, matplotlib.pyplot as plt, pingouin as pg, pandas as pd
from scipy.stats import wilcoxon, pearsonr, ttest_ind

import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.proportion import proportions_ztest

eps=np.finfo(float).eps

def get_demographics(task, codes):

   #Participant ID,Group,Initial,Dx,CB,Gender,Age ,Race,Handedness?,IQ,STAIS1.Total,STAIS2.Total,BAI.Total,Shock Level

   gender=[]
   age=[]
   iq=[]
   stai1=[]
   stai2=[]
   bai=[]
   shock=[]
   masq_gda=[]
   masq_aa=[]
   masq_gdd=[]
   masq_ad=[]
   rt_safe=[]
   rt_threat=[]

   codes=[ii.replace('LA1','') for ii in codes]

   if task == 'LA2':

      with open('Masterfile_AM_122917.csv') as file:

         reader=csv.reader(file)
         next(reader)

         for ii in reader:

            ii[0] = ii[0].strip()

            if not ii[0] in codes:
               continue

            gender.append(0 if ii[5].strip() == 'M' else 1)
            age.append(float(ii[6].strip()))
            iq.append(float(ii[9].strip()))
            stai1.append(float(ii[31].strip()))
            stai2.append(float(ii[32].strip()))
            bai.append(float(ii[33].strip()))
            shock.append(float(ii[34].strip()))
            masq_gda.append(float(ii[17].strip()) if float(ii[17].strip()) < 999 else np.nan)
            masq_aa.append(float(ii[18].strip())  if float(ii[18].strip()) < 999 else np.nan)
            masq_gdd.append(float(ii[19].strip()) if float(ii[19].strip()) < 999 else np.nan)
            masq_ad.append(float(ii[20].strip())  if float(ii[20].strip()) < 999 else np.nan)
            rt_safe.append(float(ii[41].strip()))
            rt_threat.append(float(ii[42].strip()))


   else:

      dic_gender={}
      dic_age={}
      dic_iq={}
      dic_stai1={}
      dic_stai2={}
      dic_shock={}
      dic_bai={}
      dic_masq_gda={}
      dic_masq_aa={}
      dic_masq_gdd={}
      dic_masq_ad={}
      dic_rt_safe={}
      dic_rt_threat={}

      with open('loss_aversion_r.csv') as file:

         reader=csv.reader(file)
         next(reader)

         for ii in reader:

            ii[0] = ii[0].strip()

            if not ii[0] in codes:
               continue

            dic_gender[ii[0]]=float(ii[7].strip())
            dic_age[ii[0]]=float(ii[8].strip())
            
            dic_iq[ii[0]]=float(ii[10].strip()) if ii[10].strip().isnumeric() else np.nan

            dic_stai1[ii[0]]=float(ii[11].strip())
            dic_stai2[ii[0]]=float(ii[12].strip())
            dic_shock[ii[0]]=np.nan
            dic_bai[ii[0]]=np.nan
            dic_masq_gda[ii[0]]=np.nan
            dic_masq_aa[ii[0]]=np.nan
            dic_masq_gdd[ii[0]]=np.nan
            dic_masq_ad[ii[0]]=np.nan
            if int(ii[2]) == 0:
               dic_rt_safe[ii[0]] = float(ii[6].strip())/1000
            else:
               dic_rt_threat[ii[0]] = float(ii[6].strip())/1000

      with open('LA1_master_leo.csv') as file:

         reader=csv.reader(file)
         next(reader)

         for ii in reader:

            if not ii[0].replace('LA1','') in codes:
               continue

            try:
               dic_shock[ii[0]]=float(ii[21].strip())
               dic_bai[ii[0]]=float(ii[9].strip())
               dic_masq_gda[ii[0]]=float(ii[31].strip())  if ii[31].isnumeric() else np.nan
               dic_masq_aa[ii[0]] =float(ii[32].strip())  if ii[32].isnumeric() else np.nan
               dic_masq_gdd[ii[0]]=float(ii[33].strip())  if ii[33].isnumeric() else np.nan
               dic_masq_ad[ii[0]] =float(ii[34].strip())  if ii[34].isnumeric() else np.nan
            except:
               pass
               #pdb.set_trace()

         gender = list(dic_gender.values())
         age = list(dic_age.values())
         iq = list(dic_iq.values())
         stai1 = list(dic_stai1.values())
         stai2 = list(dic_stai2.values())
         bai = list(dic_bai.values())
         shock = list(dic_shock.values())
         masq_gda = list(dic_masq_gda.values())
         masq_aa = list(dic_masq_aa.values())
         masq_gdd = list(dic_masq_gdd.values())
         masq_ad = list(dic_masq_ad.values())
         rt_safe = list(dic_rt_safe.values())
         rt_threat = list(dic_rt_threat.values())


   return gender,age,iq,stai1,stai2,bai,shock,masq_gda,masq_aa,masq_gdd,masq_ad,rt_safe,rt_threat

def barplot_marginal_triple(mean_gda,mean_gdd,mean_ad,sem_gda,sem_gdd,sem_ad,pval_gda,pval_gdd,pval_ad,alpha,\
ylim,ystep=0.75,xlabels=[None,None,None],stat_label=''):

   plt.gcf().patch.set_facecolor('w')
   b=plt.bar([0.2,0.3,0.4],
   height=[mean_gda,mean_gdd,mean_ad],
   width=0.05,yerr=[sem_gda,sem_gdd,sem_ad],
   align='center',alpha=0.5,edgecolor='black',
   ecolor='black',linewidth=1,error_kw=dict(lw=1, capsize=10, capthick=1))

   plt.xticks([0.2,0.3,0.4],xlabels,fontsize=12)
   plt.xlim([0.1,0.5])
   plt.ylabel('%s'%stat_label,fontsize=16)
   plt.tight_layout()

   b[0].set_color('orange')
   b[1].set_color('orange')
   b[2].set_color('orange')

   if ylim == None:
      y = 0.5*plt.ylim()[1]
   else:
      plt.ylim(ylim)
      y= 0.5*ylim[1]

   resy = (ylim[1]-ylim[0])/100.0
   y = ystep*(np.abs(ylim[1])+np.abs(ylim[0]))

   plt.text(0.175,(mean_gda-2.5*se_gda) ,'p=%.3f'%pval_gda + ' (*)' if pval_gda < alpha else 'p=%.3f'%pval_gda + ' (NS)',alpha=0.5,fontsize=10)
   plt.text(0.275,(mean_gdd-2.5*se_gdd),'p=%.3f'%pval_gdd + ' (*)' if pval_gdd < alpha else 'p=%.3f'%pval_gdd + ' (NS)',alpha=0.5,fontsize=10)
   plt.text(0.375,(mean_ad-2.5*se_ad),'p=%.3f'%pval_ad+ ' (*)' if pval_ad < alpha else 'p=%.3f'%pval_ad + ' (NS)',alpha=0.5,fontsize=10)


def barplot_marginal(mean_h,mean_a,sem_h,sem_a,pval,pval_h,pval_a,alpha,ylim,ystep=0.75,xlabels=[None,None],stat_label=''):

   plt.gcf().patch.set_facecolor('w')
   b=plt.bar([0.2,0.3],
   height=[mean_h,mean_a],
   width=0.05,yerr=[sem_h,sem_a],
   align='center',alpha=0.5,edgecolor='black',
   ecolor='black',linewidth=1,error_kw=dict(lw=1, capsize=10, capthick=1))

   plt.xticks([0.2,0.3],xlabels,fontsize=14)
   plt.xlim([0.1,0.4])
   plt.ylabel('%s'%stat_label,fontsize=16)
   plt.tight_layout()

   b[0].set_color('blue')
   b[1].set_color('orange')

   if ylim == None:
      y = 0.5*plt.ylim()[1]
   else:
      plt.ylim(ylim)
      y= 0.5*ylim[1]

   resy = (ylim[1]-ylim[0])/100.0
   y = ystep*(np.abs(ylim[1])+np.abs(ylim[0]))

   plt.plot([0.2, 0.3],[y, y],'-',color='black',alpha=0.1)
   plt.text(0.225,y+resy,'p=%.3f'%pval + ' (*)' if pval < alpha else 'p=%.3f'%pval + ' (NS)',alpha=0.35,fontsize=10) 
 
   try: 
      if pval_h < 0.001:
         pval_h_txt = 'p<0.001 (*)'
      else:
         pval_h_txt = 'p=%.3f'%pval_h + ' (*)' if pval_h < alpha else 'p=%.3f'%pval_h + ' (NS)'
   except:
      pval_h_txt=pval_h

   try: 
      if pval_a < 0.001:
         pval_a_txt = 'p<0.001 (*)'
      else:
         pval_a_txt = 'p=%.3f'%pval_a + ' (*)' if pval_a < alpha else 'p=%.3f'%pval_a + ' (NS)'
   except:
      pval_a_txt=pval_a


   plt.text(0.175,mean_h-1.95*se_h,pval_h_txt,alpha=0.5,fontsize=10)
   plt.text(0.275,mean_a-2.5*se_a,pval_a_txt,alpha=0.5,fontsize=10)



def bootstrapped_correlation(counts, learn, N, subsample):

   rnd=np.random.RandomState(seed=11901)

   rho=[]
   pl=[]

   for ii in range(N):


      ind_nan=np.isnan(learn) + np.isnan(counts)
      counts_tmp=counts[~ind_nan]
      learn_tmp=learn[~ind_nan]

      inds=rnd.choice(range(len(counts_tmp)),subsample)

      counts_tmp=counts_tmp[inds]
      learn_tmp=learn_tmp[inds]


      rho.append(pearsonr(counts_tmp,learn_tmp)[0])
      pl.append(np.polyfit(counts_tmp, learn_tmp, 1))

   return np.array(rho),pl




def plot_sensitivity(data,codes,params,model,color,marker,label,ytrue_ref=None,yhat_ref=None):

   np.random.seed(11901)

   ytrue_all=[]
   yhat_all=[]
   corr_all=[]

   for ii,p_ii in zip(data.keys(),params):

      if np.any(np.isnan(p_ii)):
         continue

      mu,rho,lambd=p_ii


      X = np.array([jj[0][0] for jj in data[ii]]).astype('float')

      yhat_ii=[]
      ytrue_ii=[]
      for x_ii in X:

         x_gain = x_ii[0]
         x_loss = x_ii[1]
         x_sure = x_ii[2]
         y = x_ii[3]

         p_ii = p_gamble_accept(x_gain, x_loss, x_sure, mu, rho, lambd)
         yhat_ii.append(np.random.binomial(1, p=p_ii))

         ytrue_ii.append(y)

      yhat_all.append(yhat_ii)
      ytrue_all.append(ytrue_ii)
      corr_all.append(pearsonr(yhat_ii,ytrue_ii)[0])


   if ytrue_ref != None and yhat_ref != None: 

      for ind,vals in enumerate(zip(ytrue_all, yhat_all, ytrue_ref, yhat_ref)):
      
         ii=vals[0]
         jj=vals[1]
         ii_hat=vals[2]
         jj_hat=vals[3]

         if ind == 0:
            plt.plot(100*np.nanmean(ii),100*np.nanmean(jj),color=color,marker=marker,label=label,alpha=0.75,linestyle="None") 
            plt.plot([np.nanmean(ii),np.nanmean(ii_hat)],[np.nanmean(jj),np.nanmean(jj_hat)],color='k',linestyle='-',alpha=0.1,markerfacecolor='none',marker=marker)
         else:
            plt.plot(100*np.nanmean(ii),100*np.nanmean(jj),color=color,marker=marker,alpha=0.75) 
            plt.plot([np.nanmean(ii),np.nanmean(ii_hat)],[np.nanmean(jj),np.nanmean(jj_hat)],color='k',linestyle='-',alpha=0.1,markerfacecolor='none',marker=marker)

   else:
      for ind,vals in enumerate(zip(ytrue_all,yhat_all)):
      
         ii=vals[0]
         jj=vals[1]

         if ind == 0:
            plt.plot(100*np.nanmean(ii),100*np.nanmean(jj),color=color,marker=marker,label=label,alpha=0.75) 
         else:
            plt.plot(100*np.nanmean(ii),100*np.nanmean(jj),color=color,marker=marker,alpha=0.75) 


   plt.plot([0,100],[0,100],'k--',alpha=0.1)

   plt.gcf().text(0,110, 'A',fontsize=40)
   plt.xlabel('Observed (%)',fontsize=14)
   plt.ylabel('Simulated (%)',fontsize=14)

   bid_ytrue = [np.nanmean(ii) for ii in ytrue_all]
   bid_yhat = [np.nanmean(ii) for ii in yhat_all]

   plt.gca().set_aspect('equal', 'box')
   plt.xlim(0,100)
   plt.ylim(0,100)

   return corr_all, bid_ytrue, bid_yhat, ytrue_all, yhat_all


def set_box_color(bp, color):

   plt.setp(bp['boxes'], color=color)
   plt.setp(bp['whiskers'], color=color)
   plt.setp(bp['caps'], color=color)
   plt.setp(bp['medians'], color=color)


def u(x,lambd,rho):

   if x > 0:
      return x**rho
   else:
      return -lambd*(-x)**rho


def p_gamble_accept(x_gain, x_loss, x_sure, mu, rho, lambd):


   return 1.0/(1.0 + np.exp( -mu*( 0.5*( u(x_gain,lambd,rho) \
                                          + u(x_loss,lambd,rho))\
                                          - u(x_sure,lambd,rho) ) ) )


def med_opt_params(fits, thr):


   out = [(ii[0],ii[1]) for ii in fits]

   params,f=zip(*out)

   #z=[len(ii.shape) for ii in f]
   #print(z)
   #print(all(z))

   try:
      f = np.array(f)
   except:
      pass
      #pdb.set_trace()

   if len(f.shape) > 2:
      print(f.shape)
      f=np.squeeze(f)
      #pdb.set_trace()
 


   params=np.array(params)
   opt_params = []
   sel=[]

   for kk in f.T:
      try:
         sel.append (kk <= np.nanpercentile(kk,thr))
      except:
         #pdb.set_trace()
         pass

   opt_params = np.array([np.nanmean(params[ii,ss],0) if len(ii) > 0 else np.nan*3 for ss,ii in enumerate(sel)])

   ll=[np.nanmean(f[:,ii][sel[ii]]) for ii in range(len(sel))]


   return opt_params, np.array(ll)





def get_val_data(cohort, cond,offset=''):

   data = {}
   with open('raw_data_la1.csv') as file:

      reader=csv.reader(file)
      next(reader)
      for ii in reader:
         ii[0] = ii[0].strip() + offset

         if not ii[-1] in cohort:
            continue

         if ii[1].strip() in cond:
            if ii[0] not in data:
               data[ii[0]]=[]
            data[ii[0]].append(([[ii[2]]+['-'+ii[3]]+[0]+[int(ii[5] == 'y')]], ii[1].strip(), ii[-1]))


   return data


def get_data(cohort, cond, offset=''):

   data = {}
   groups={}

   # Assign subjects to the right groups
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
         if not groups[ii[0]] in cohort:
            continue

         if ii[4].strip() in cond:
            if ii[0] not in data:
               data[ii[0]]=[]

            if int(ii[-1] == 'y') or int(ii[-1] == 'n') and 'no' in ii[9]:
               data[ii[0]].append(([ii[5:8]+[int(ii[-1] == 'y')]], ii[4].strip(), groups[ii[0]]))


   return data



def plot_propensities(healthy_obs, healthy_sim, patients_obs, patients_sim, colors, xlab):

   color_healthy, color_patients = colors

   obs_hp= pg.mwu(healthy_obs, patients_obs).values[0][[0,2,4]]

   sim_hp= pg.mwu(healthy_sim, patients_sim).values[0][[0,2,4]]

   obs_sim_h= pg.wilcoxon(healthy_obs, healthy_sim).values[0][[0,2,4]]
   obs_sim_p= pg.wilcoxon(patients_obs, patients_sim).values[0][[0,2,4]]

   healthy_obs=list(map(lambda x: 100*x,healthy_obs))
   patients_obs=list(map(lambda x: 100*x,patients_obs))
   healthy_sim=list(map(lambda x: 100*x,healthy_sim))
   patients_sim=list(map(lambda x: 100*x,patients_sim))

   healthy  = [healthy_obs, healthy_sim]
   patients = [patients_obs, patients_sim]

   bh = plt.boxplot(healthy, positions=[1,2],widths=[0.6]*2,notch=True,showfliers=False,bootstrap=10000)
   ba = plt.boxplot(patients, positions=[4,5],widths=[0.6]*2,notch=True,showfliers=False,bootstrap=10000)

   plt.scatter(np.ones((len(healthy_obs),)),healthy_obs,color=color_healthy,marker='o',label='H',alpha=0.1)
   plt.scatter(2*np.ones((len(healthy_sim),)),healthy_sim,color=color_healthy,marker='o',alpha=0.1)
   plt.scatter(4*np.ones((len(patients_obs),)),patients_obs,color=color_patients,marker='o',label='A',alpha=0.1)
   plt.scatter(5*np.ones((len(patients_sim),)),patients_sim,color=color_patients,marker='o',alpha=0.1)

   plt.ylim([0,100])

   set_box_color(bh,color_healthy)
   set_box_color(ba,color_patients)

   plt.legend([bh["boxes"][0], ba["boxes"][0]], ['H', 'A'], loc='upper center')

   plt.xticks([1,2,4,5],xlab,fontsize=14)

   plt.plot([1,2],[4, 4],'-',color='black',alpha=0.1)
   plt.plot([4,5],[4, 4],'-',color='black',alpha=0.1)
   plt.text(1,5,'p=%.3f'%obs_sim_h[1] + ' (*)' if obs_sim_h[1] < 0.05 else 'p=%.3f'%obs_sim_h[1] + ' (NS)' ,alpha=0.35,color='black', fontsize=10)
   plt.text(4,5,'p=%.3f'%obs_sim_p[1] + ' (*)' if obs_sim_p[1] < 0.05 else 'p=%.3f'%obs_sim_p[1] + ' (NS)' ,alpha=0.35,color='black', fontsize=10)

   plt.ylabel('Gambling propensity (%)',fontsize=14)




if __name__ == '__main__' :
   
   solver='L-BFGS-B'
   #solver='nelder-mead'

   data_healthy_safe_LA1    = get_val_data('healthy','safe',offset='LA1')
   data_healthy_threat_LA1  = get_val_data('healthy','threat',offset='LA1')
   data_patients_safe_LA1   = get_val_data('patient','safe',offset='LA1')
   data_patients_threat_LA1 = get_val_data('patient','threat',offset='LA1')

   data_healthy_safe_LA2    = get_data(['healthy'],['safe'])
   data_healthy_threat_LA2  = get_data(['healthy'],['threat'])
   data_patients_safe_LA2   = get_data(['patient'],['safe'])
   data_patients_threat_LA2 = get_data(['patient'],['threat'])


   data_healthy_la1 = {**data_healthy_safe_LA1, **data_healthy_threat_LA1}
   data_healthy_la2 = {**data_healthy_safe_LA2, **data_healthy_threat_LA2}
   data_patients_la1 = {**data_patients_safe_LA1, **data_patients_threat_LA1}
   data_patients_la2 = {**data_patients_safe_LA2, **data_patients_threat_LA2}


   h_file_la1 = 'fits_healthy_LA1_%s_m=lambda_mu.pkl'%solver
   a_file_la1 = 'fits_patients_LA1_%s_m=lambda_mu.pkl'%solver

   h_file_la2 = 'fits_healthy_LA2_%s_m=lambda_mu.pkl'%solver
   a_file_la2 = 'fits_patients_LA2_%s_m=lambda_mu.pkl'%solver

   healthy_la1 = pickle.load(open(h_file_la1,'rb'))
   patients_la1 = pickle.load(open(a_file_la1,'rb'))
   healthy_la2 = pickle.load(open(h_file_la2,'rb'))
   patients_la2 = pickle.load(open(a_file_la2,'rb'))

   healthy_codes_la1 = data_healthy_la1.keys()
   patients_codes_la1 = data_patients_la1.keys()
   healthy_codes_la2 = data_healthy_la2.keys()
   patients_codes_la2 = data_patients_la2.keys()




   # Display demographics and other data Task 1
   gender_h_la1,age_h_la1,iq_h_la1,stai1_h_la1,stai2_h_la1,bai_h_la1,shock_h_la1,masq_gda_h_la1,masq_aa_h_la1,masq_gdd_h_la1,masq_ad_h_la1,\
   rt_safe_h_la1,rt_threat_h_la1 = get_demographics('LA1', healthy_codes_la1)
   gender_a_la1,age_a_la1,iq_a_la1,stai1_a_la1,stai2_a_la1,bai_a_la1,shock_a_la1,masq_gda_a_la1,masq_aa_a_la1,masq_gdd_a_la1,masq_ad_a_la1,\
   rt_safe_a_la1,rt_threat_a_la1 = get_demographics('LA1', patients_codes_la1)

   # Display demographics and other data Task 2
   gender_h_la2,age_h_la2,iq_h_la2,stai1_h_la2,stai2_h_la2,bai_h_la2,shock_h_la2,masq_gda_h_la2,masq_aa_h_la2,masq_gdd_h_la2,masq_ad_h_la2,\
   rt_safe_h_la2,rt_threat_h_la2 = get_demographics('LA2', healthy_codes_la2)
   gender_a_la2,age_a_la2,iq_a_la2,stai1_a_la2,stai2_a_la2,bai_a_la2,shock_a_la2,masq_gda_a_la2,masq_aa_a_la2,masq_gdd_a_la2,masq_ad_a_la2,\
   rt_safe_a_la2,rt_threat_a_la2 = get_demographics('LA2', patients_codes_la2)


   age_a_la1       = np.array(age_a_la1)
   age_a_la2       = np.array(age_a_la2)
   iq_a_la1        = np.array(iq_a_la1)
   iq_a_la2        = np.array(iq_a_la2)
   masq_gda_a_la1  = np.array(masq_gda_a_la1)
   masq_gda_a_la2  = np.array(masq_gda_a_la2)
   masq_aa_a_la1   = np.array(masq_aa_a_la1)
   masq_aa_a_la2   = np.array(masq_aa_a_la2)
   masq_gdd_a_la1  = np.array(masq_gdd_a_la1)
   masq_gdd_a_la2  = np.array(masq_gdd_a_la2)
   masq_ad_a_la1   = np.array(masq_ad_a_la1)
   masq_ad_a_la2   = np.array(masq_ad_a_la2)
   shock_a_la1     = np.array(shock_a_la1)
   shock_a_la2     = np.array(shock_a_la2)
   stai1_a_la1     = np.array(stai1_a_la1)
   stai1_a_la2     = np.array(stai1_a_la2)
   stai2_a_la1     = np.array(stai2_a_la1)
   stai2_a_la2     = np.array(stai2_a_la2)
   

   # Final fit parameters
   healthy_la1_params,_ = med_opt_params(healthy_la1,5)
   healthy_la2_params,_ = med_opt_params(healthy_la2,5)
   patients_la1_params,_ = med_opt_params(patients_la1,5)
   patients_la2_params,_ = med_opt_params(patients_la2,5)

   # Add risk column (rho=1) to healthy and patients in task 1
   patients_la1_params=np.vstack((patients_la1_params[:,0],np.ones((len(patients_la1_params,))),patients_la1_params[:,1])).T
   healthy_la1_params=np.vstack((healthy_la1_params[:,0],np.ones((len(healthy_la1_params,))),healthy_la1_params[:,1])).T

   # Add risk column (rho=1) only to patients in task 2
   patients_la2_params=np.vstack((patients_la2_params[:,0],np.ones((len(patients_la2_params,))),patients_la2_params[:,1])).T
   healthy_la2_params=np.vstack((healthy_la2_params[:,0],np.ones((len(healthy_la2_params,))),healthy_la2_params[:,1])).T


   ###############################################################################################################
   # Figure 3
   ###############################################################################################################

   # Sensitivity plots
   plt.figure(figsize=(10,8))
   plt.subplot(2,2,1)

   corr_healthy_la1, obs_h_la1, sim_h_la1, _,_ = plot_sensitivity(data_healthy_la1, healthy_codes_la1, healthy_la1_params, 'lambda_mu', 'blue','o','H')
   corr_anxious_la1, obs_a_la1, sim_a_la1, _,_  = plot_sensitivity(data_patients_la1, patients_codes_la1, patients_la1_params, 'lambda_mu', 'orange','o','A')

   plt.title('Task 1',fontsize=16)
   plt.legend(fontsize=14)
   plt.subplot(2,2,2)

   corr_healthy_la2, obs_h_la2, sim_h_la2, _,_ = plot_sensitivity(data_healthy_la2, healthy_codes_la2, healthy_la2_params, 'full', 'blue','o','H')
   corr_anxious_la2, obs_a_la2, sim_a_la2, _,_  = plot_sensitivity(data_patients_la2, patients_codes_la2, patients_la2_params, 'full', 'orange','o','A')

   plt.title('Task 2',fontsize=16)
   plt.legend(fontsize=14)
   plt.tight_layout()

   # Gambling propensities obs vs sim
   plt.subplot(2,2,3)
   plot_propensities(obs_h_la1, sim_h_la1, obs_a_la1, sim_a_la1, colors=['blue', 'orange'],xlab=['Obs','Sim','Obs','Sim'])
   plt.title('Task 1',fontsize=16)

   plt.subplot(2,2,4)
   plot_propensities(obs_h_la2, sim_h_la2, obs_a_la2, sim_a_la2, colors=['blue', 'orange'],xlab=['Obs','Sim','Obs','Sim'])
   plt.title('Task 2',fontsize=16)

   plt.tight_layout()

   plt.gca().text(-1.35,2.35,'A',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')
   plt.gca().text(-0.15,2.35,'B',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')
   plt.gca().text(-1.35,1.05,'C',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')
   plt.gca().text(-0.15,1.05,'D',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')

   rho_h_la1,_ = bootstrapped_correlation(np.array(obs_h_la1), np.array(sim_h_la1), 10000, len(obs_h_la1))
   rho_h_la2,_ = bootstrapped_correlation(np.array(obs_h_la2), np.array(sim_h_la2), 10000, len(obs_h_la2))
   rho_a_la1,_ = bootstrapped_correlation(np.array(obs_a_la1), np.array(sim_a_la1), 10000, len(obs_a_la1))
   rho_a_la2,_ = bootstrapped_correlation(np.array(obs_a_la2), np.array(sim_a_la2), 10000, len(obs_a_la2))

   plt.gca().text(-0.75,1.45,'H 95%%CI r=[%.2f, %.2f]'%(np.percentile(rho_h_la1,2.5), np.percentile(rho_h_la1,97.5)),transform=plt.gca().transAxes, color='blue')
   plt.gca().text(-0.75,1.37,'A 95%%CI r=[%.2f, %.2f]'%(np.percentile(rho_a_la1,2.5), np.percentile(rho_a_la1,97.5)),transform=plt.gca().transAxes,color='orange')

   plt.gca().text(0.45,1.45,'H 95%%CI r=[%.2f, %.2f]'%(np.percentile(rho_h_la2,2.5), np.percentile(rho_h_la2,97.5)),transform=plt.gca().transAxes,color='blue')
   plt.gca().text(0.45,1.37,'A 95%%CI r=[%.2f, %.2f]'%(np.percentile(rho_a_la2,2.5), np.percentile(rho_a_la2,97.5)),transform=plt.gca().transAxes,color='orange')


   ###############################################################################################################
   # Figure 4 (R results)
   ###############################################################################################################

   print('Task1: H=%d,A=%d'%(len(healthy_codes_la1),len(patients_codes_la1)))
   print('Task2: H=%d,A=%d'%(len(healthy_codes_la2),len(patients_codes_la2)))

   '''
   This comes from outputs of la.R
   and are manually assigned to the variables below

   Task = b'Task1':
   Dx         emmean    SE df null z.ratio p.value
   b'Anxiety'   1.78 0.187 NA    1   4.181  0.0001
   b'Healthy'   2.08 0.161 NA    1   6.698  <.0001

   Task = b'Task2':
   Dx         emmean    SE df null z.ratio p.value
   b'Anxiety'   1.04 0.192 NA    1   0.216  0.9707
   b'Healthy'   1.77 0.187 NA    1   4.137  0.0001

   Results are averaged over the levels of: Gender 
   P value adjustment: sidak method for 2 tests 

   contrasts
   Task = b'Task1':
   contrast                estimate    SE df z.ratio p.value
   b'Anxiety' - b'Healthy'   -0.300 0.256 NA  -1.174  0.2406

   Task = b'Task2':
   contrast                estimate    SE df z.ratio p.value
   b'Anxiety' - b'Healthy'   -0.732 0.283 NA  -2.588  0.0097

   '''

   alpha=0.025
   plt.figure(figsize=(10,8))
   plt.subplot(2,2,1)
   mean_h= 2.08
   se_h = 0.161
   mean_a = 1.78
   se_a = 0.187
   p_contrast= 0.2406
   p_h = 'p<0.0001 (*)'
   p_a = 0.0001
   barplot_marginal(mean_h, mean_a, se_h, se_a, p_contrast,p_h,p_a,alpha, [0,2.5], 0.88, 
   ['Healthy','Anxiety'], 'Marginal $\\bar{\\lambda}$')
 
   plt.plot([0, 0.4],[1, 1],'--',color='black',alpha=0.3) 
   plt.text(0.107,0.2,'Gain seeking →',rotation=-90,fontsize=10,alpha=0.3)
   plt.text(0.11,1.1,'Loss averse →',rotation=90,fontsize=10,alpha=0.3)

   plt.title('Task 1',fontsize=16)
   plt.subplot(2,2,2)
   mean_h= 1.77
   se_h = 0.187
   mean_a = 1.04
   se_a = 0.192
   p_contrast= 0.0097
   p_h = 0.0001
   p_a = 0.9707
   barplot_marginal(mean_h, mean_a, se_h, se_a, p_contrast,p_h, p_a, alpha,[0,2.5], 0.88, 
   ['Healthy','Anxiety'], 'Marginal $\\bar{\\lambda}$')

   plt.plot([0, 0.4],[1, 1],'--',color='black',alpha=0.3) 
   plt.text(0.107,0.2,'Gain seeking →',rotation=-90,fontsize=10,alpha=0.3)
   plt.text(0.11,1.1,'Loss averse →',rotation=90,fontsize=10,alpha=0.3)
   plt.title('Task 2',fontsize=16)


   '''
   This comes from outputs of la.R
   and are manually assigned to the variables below

   Task = b'Task1':
   Dx         MASQ_AA.trend     SE df z.ratio p.value
   b'Anxiety'      -0.00119 0.0154 NA  -0.077  0.9385
   b'Healthy'       0.00740 0.0520 NA   0.142  0.8869

   Task = b'Task2':
   Dx         MASQ_AA.trend     SE df z.ratio p.value
   b'Anxiety'       0.04015 0.0113 NA   3.546  0.0004
   b'Healthy'      -0.14831 0.0642 NA  -2.310  0.0209

   $contrasts
   Task = b'Task1':
   contrast                estimate     SE df z.ratio p.value
   b'Anxiety' - b'Healthy' -0.00859 0.0541 NA  -0.159  0.8740

   Task = b'Task2':
   contrast                estimate     SE df z.ratio p.value
   b'Anxiety' - b'Healthy'  0.18846 0.0652 NA   2.890  0.0038
   '''

   alpha = 0.0125
   plt.subplot(2,2,3)
   mean_h=0.00740
   se_h = 0.0520
   mean_a = -0.00119
   se_a = 0.0154
   p_contrast= 0.8740
   p_h=0.8869
   p_a=0.9385
   plt.title('Task 1',fontsize=16)
   barplot_marginal(mean_h,mean_a,se_h,se_a,p_contrast,p_h, p_a,alpha,[-0.3,0.2], 0.2,
   ['Healthy','Anxiety'], '$\\beta_{MASQ~AA}$')
   plt.plot([0, 0.4],[0, 0],'--',color='black',alpha=0.3) 

   plt.subplot(2,2,4)
   mean_h= -0.14831
   se_h = 0.0642
   mean_a = 0.04015
   se_a = 0.0113
   p_contrast=0.0038
   p_h = 0.0209
   p_a = 0.0004
   plt.title('Task 2',fontsize=16)
   barplot_marginal(mean_h,mean_a,se_h,se_a,p_contrast,p_h, p_a,alpha,[-0.3,0.2],0.2,
   ['Healthy','Anxiety'], '$\\beta_{MASQ~AA}$')
   plt.plot([0, 0.4],[0,0],'--',color='black',alpha=0.3) 


   plt.tight_layout()

   plt.gca().text(-1.35,2.28,'A',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')
   plt.gca().text(-0.15,2.28,'B',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')
   plt.gca().text(-1.35,1.05,'C',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')
   plt.gca().text(-0.15,1.05,'D',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')


   '''
   This comes from outputs of la.R
   and are manually assigned to the variables below

   Task = b'Task1':
   Dx         MASQ_GDA.trend     SE df z.ratio p.value
   b'Anxiety'        0.00889 0.0148 NA   0.599  0.5490

   Task = b'Task2':
   Dx         MASQ_GDA.trend     SE df z.ratio p.value
   b'Anxiety'        0.02709 0.0158 NA   1.714  0.0865

   $emtrends
   Task = b'Task1':
   Dx         MASQ_AD.trend      SE df z.ratio p.value
   b'Anxiety'      -0.01161 0.00789 NA  -1.471  0.1412

   Task = b'Task2':
   Dx         MASQ_AD.trend      SE df z.ratio p.value
   b'Anxiety'       0.01588 0.00995 NA   1.596  0.1104

   $emtrends
   Task = b'Task1':
   Dx         MASQ_GDD.trend     SE df z.ratio p.value
   b'Anxiety'       1.39e-02 0.0115 NA   1.206  0.2278

   Task = b'Task2':
   Dx         MASQ_GDD.trend     SE df z.ratio p.value
   b'Anxiety'       7.41e-05 0.0165 NA   0.004  0.9964

   '''

   alpha = 0.05

   mean_gda=0.00889
   se_gda =0.0148
   mean_gdd =1.39e-02
   se_gdd =0.0115
   mean_ad =-0.01161
   se_ad =0.00789

   pval_gda=0.5490
   pval_gdd=0.2278
   pval_ad=0.1412

   plt.figure(figsize=(10,4))
   plt.subplot(1,2,1)
   plt.title('Task 1',fontsize=16)
   barplot_marginal_triple(mean_gda,mean_gdd,mean_ad,se_gda,se_gdd,se_ad,pval_gda,pval_gdd,pval_ad,alpha,\
   [-0.1,0.1],0.2,['$\\beta_{MASQ~GDA}$','$\\beta_{MASQ~GDD}$','$\\beta_{MASQ~AD}$'], 
   'Non anxiety-specific $\\beta_{MASQ}$')
   plt.plot([0, 0.6],[0, 0],'--',color='black',alpha=0.3) 
   plt.tight_layout()

   mean_gda= 0.02709
   se_gda = 0.0158
   mean_gdd = 7.41e-05
   se_gdd = 0.0165
   mean_ad = 0.01588
   se_ad = 0.00995

   pval_gda= 0.0865
   pval_gdd= 0.9964
   pval_ad= 0.1104

   plt.subplot(1,2,2)
   plt.title('Task 2',fontsize=16)
   barplot_marginal_triple(mean_gda,mean_gdd,mean_ad,se_gda,se_gdd,se_ad,pval_gda,pval_gdd,pval_ad,alpha,\
   [-0.1,0.1],0.2,['$\\beta_{MASQ~GDA}$','$\\beta_{MASQ~GDD}$','$\\beta_{MASQ~AD}$'], 'Non anxiety-specific $\\beta_{MASQ}$')
   plt.plot([0, 0.6],[0, 0],'--',color='black',alpha=0.3) 

   plt.tight_layout()

   plt.gca().text(-1.35,1.05,'E',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')
   plt.gca().text(-0.15,1.05,'F',transform=plt.gca().transAxes,fontsize=24,fontweight='bold')


   # Group and task
   X = np.empty((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),2), dtype='S10')
   X[:,0] = 'Task1'
   X[:,1] = 'Healthy'
   X[len(age_h_la1)+len(age_a_la1):,0]='Task2' # LA2 = 1
   X[len(age_h_la1):len(age_h_la1)+len(age_a_la1),1]='Anxiety' #Anx = 1 (LA1)
   X[len(age_h_la1)+len(age_a_la1)+len(age_h_la2):,1]='Anxiety'#Anx = 1 (LA2)

   X_Task1 = np.ones((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),1))
   X_Task1[len(age_h_la1)+len(age_a_la1):,0]=0

   X_Task2 = np.zeros((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),1))
   X_Task2[len(age_h_la1)+len(age_a_la1):,0]=1

   X_Dx_H = np.ones((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),1))
   X_Dx_H[len(age_h_la1):len(age_h_la1)+len(age_a_la1)]=0
   X_Dx_H[len(age_h_la1)+len(age_a_la1)+len(age_h_la2):]=0

   X_Dx_A = np.zeros((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),1))
   X_Dx_A[len(age_h_la1):len(age_h_la1)+len(age_a_la1)]=1
   X_Dx_A[len(age_h_la1)+len(age_a_la1)+len(age_h_la2):]=1

   '''
   X = np.zeros((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),2))
   X[:,1]=1
   X[len(age_h_la1)+len(age_a_la1):,0]=1
   X[len(age_h_la1):len(age_h_la1)+len(age_a_la1),1]=1
   X[len(age_h_la1)+len(age_a_la1)+len(age_h_la2):,1]=1
   '''

   #Gender
   X_gender    = np.hstack((gender_h_la1,gender_a_la1,gender_h_la2,gender_a_la2))
   X_g = np.empty((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),1), dtype='S10')
   X_g[X_gender == 0]='M'
   X_g[X_gender == 1]='F'


   X_gf = np.zeros((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),1))
   X_gf[X_gender == 1]=1

   X_gm = np.zeros((len(age_h_la1) + len(age_a_la1) + len(age_h_la2) + len(age_a_la2),1))
   X_gm[X_gender == 0]=1


   #X_g=X_gender

   #Age
   X_age      = np.hstack((age_h_la1,age_a_la1,age_h_la2,age_a_la2))
   #IQ
   X_iq       = np.hstack((iq_h_la1,iq_a_la1,iq_h_la2,iq_a_la2))
   # State anxiety
   X_stai1     = np.hstack((stai1_h_la1,stai1_a_la1,stai1_h_la2,stai1_a_la2))
   # State anxiety
   X_stai2     = np.hstack((stai2_h_la1,stai2_a_la1,stai2_h_la2,stai2_a_la2))
   # Shock ratings
   X_shock    = np.hstack((shock_h_la1,shock_a_la1,shock_h_la2,shock_a_la2))
   # MASQ Anxiety
   X_masq_gda = np.hstack((masq_gda_h_la1,masq_gda_a_la1,masq_gda_h_la2,masq_gda_a_la2))
   X_masq_aa  = np.hstack((masq_aa_h_la1,masq_aa_a_la1,masq_aa_h_la2,masq_aa_a_la2))
   # MASQ Depression
   X_masq_gdd = np.hstack((masq_gdd_h_la1,masq_gdd_a_la1,masq_gdd_h_la2,masq_gdd_a_la2))
   X_masq_ad  = np.hstack((masq_ad_h_la1,masq_ad_a_la1,masq_ad_h_la2,masq_ad_a_la2))

   loss_av = np.hstack([healthy_la1_params[:,2],patients_la1_params[:,2],healthy_la2_params[:,2],patients_la2_params[:,2]])
   consist = np.hstack([healthy_la1_params[:,0],patients_la1_params[:,0],healthy_la2_params[:,0],patients_la2_params[:,0]])
   gamb_obs    = np.hstack([obs_h_la1,obs_a_la1,obs_h_la2,obs_a_la2])
   gamb_sim    = np.hstack([sim_h_la1,sim_a_la1,sim_h_la2,sim_a_la2])


   df = pd.DataFrame(data=X[:,0:2],columns=['Task','Dx'])
   df.insert(2,"Task_1",X_Task1)
   df.insert(3,"Task_2",X_Task2)
   df.insert(4,"Dx_H",X_Dx_H)
   df.insert(5,"Dx_A",X_Dx_A)
   df.insert(6,"Gender",X_g)
   df.insert(7,"Gender_F",X_gf)
   df.insert(8,"Gender_M",X_gm)
   df.insert(9,"Age",X_age)
   df.insert(10,"IQ",X_iq)
   df.insert(11,"Shock",X_shock)
   df.insert(12,"STAI_bef",X_stai1)
   df.insert(13,"STAI_aft",X_stai2)
   df.insert(14,"MASQ_GDA",X_masq_gda)
   df.insert(15,"MASQ_AA", X_masq_aa)
   df.insert(16,"MASQ_GDD",X_masq_gdd)
   df.insert(17,"MASQ_AD", X_masq_ad)
   df.insert(18,"LAMBDA", loss_av)
   df.insert(19,"MU", consist)
   df.insert(20,"LOG_LAMBDA", np.log(loss_av+eps))
   df.insert(21,"GAMB_OBS", gamb_obs)
   df.insert(22,"GAMB_SIM", gamb_sim)

   df[['Dx', 'Gender', 'Task']] = df[['Dx', 'Gender', 'Task']].apply(lambda x: x.astype('category'))


   # Generates csv file needed for R
   df.to_csv('df.csv')

   age_a_la1       = age_a_la1[~np.isnan(age_a_la1)]
   age_a_la2       = age_a_la2[~np.isnan(age_a_la2)]
   iq_a_la1        = iq_a_la1[~np.isnan(iq_a_la1)]
   iq_a_la2        = iq_a_la2[~np.isnan(iq_a_la2)]
   masq_gda_a_la1  = masq_gda_a_la1[~np.isnan(masq_gda_a_la1)]
   masq_gda_a_la2  = masq_gda_a_la2[~np.isnan(masq_gda_a_la2)]
   masq_aa_a_la1   = masq_aa_a_la1[~np.isnan(masq_aa_a_la1)]
   masq_aa_a_la2   = masq_aa_a_la2[~np.isnan(masq_aa_a_la2)]
   masq_gdd_a_la1  = masq_gdd_a_la1[~np.isnan(masq_gdd_a_la1)]
   masq_gdd_a_la2  = masq_gdd_a_la2[~np.isnan(masq_gdd_a_la2)]
   masq_ad_a_la1   = masq_ad_a_la1[~np.isnan(masq_ad_a_la1)]
   masq_ad_a_la2   = masq_ad_a_la2[~np.isnan(masq_ad_a_la2)]
   shock_a_la1     = shock_a_la1[~np.isnan(shock_a_la1)]
   shock_a_la2     = shock_a_la2[~np.isnan(shock_a_la2)]
   stai1_a_la1     = stai1_a_la1[~np.isnan(stai1_a_la1)]
   stai1_a_la2     = stai1_a_la2[~np.isnan(stai1_a_la2)]
   stai2_a_la1     = stai2_a_la1[~np.isnan(stai2_a_la1)]
   stai2_a_la2     = stai2_a_la2[~np.isnan(stai2_a_la2)]

   age = ([np.nanmean(age_a_la1),np.nanstd(age_a_la1),np.nanmean(age_a_la2),np.nanstd(age_a_la2)] + list(ttest_ind(age_a_la1,age_a_la2)))
   iq = ([np.nanmean(iq_a_la1),np.nanstd(iq_a_la1),np.nanmean(iq_a_la2),np.nanstd(iq_a_la2)] + list(ttest_ind(iq_a_la1,iq_a_la2)))
   gda = ([np.nanmean(masq_gda_a_la1),np.nanstd(masq_gda_a_la1),np.nanmean(masq_gda_a_la2),np.nanstd(masq_gda_a_la2)] + list(ttest_ind(masq_gda_a_la1,masq_gda_a_la2)))
   aa = ([np.nanmean(masq_aa_a_la1),np.nanstd(masq_aa_a_la1),np.nanmean(masq_aa_a_la2),np.nanstd(masq_aa_a_la2)] + list(ttest_ind(masq_aa_a_la1,masq_aa_a_la2)))
   gdd = ([np.nanmean(masq_gdd_a_la1),np.nanstd(masq_gdd_a_la1),np.nanmean(masq_gdd_a_la2),np.nanstd(masq_gdd_a_la2)] + list(ttest_ind(masq_gdd_a_la1,masq_gdd_a_la2)))
   ad = ([np.nanmean(masq_ad_a_la1),np.nanstd(masq_ad_a_la1),np.nanmean(masq_ad_a_la2),np.nanstd(masq_ad_a_la2)] + list(ttest_ind(masq_ad_a_la1,masq_ad_a_la2)))
   shock = ([np.nanmean(shock_a_la1),np.nanstd(shock_a_la1),np.nanmean(shock_a_la2),np.nanstd(shock_a_la2)] + list(ttest_ind(shock_a_la1,shock_a_la2)))
   stai1 = ([np.nanmean(stai1_a_la1),np.nanstd(stai1_a_la1),np.nanmean(stai1_a_la2),np.nanstd(stai1_a_la2)] + list(ttest_ind(stai1_a_la1,stai1_a_la2)))
   stai2 = ([np.nanmean(stai2_a_la1),np.nanstd(stai2_a_la1),np.nanmean(stai2_a_la2),np.nanstd(stai2_a_la2)] + list(ttest_ind(stai2_a_la1,stai2_a_la2)))


   print('*****************************************************')
   print('Table S1')

   tab=[age,iq,gda,aa,gdd,ad,shock,stai1,stai2]
   print('\n'.join(list(map(lambda x : '%.2f (%.2f)\t%.2f (%.2f)\t%.2f\t%.6f'%(x[0],x[1],x[2],x[3],x[4],x[5]),tab))))


   print('*****************************************************')
   print('Gender proportion tests (See results)')
   counts_la1 = np.array([np.sum(gender_a_la1),np.sum(gender_h_la1)])
   nobs_la1   = np.array([len(gender_a_la1),len(gender_h_la1)])
   print (proportions_ztest(counts_la1,nobs_la1))

   counts_la2 = np.array([np.sum(gender_a_la2),np.sum(gender_h_la2)])
   nobs_la2   = np.array([len(gender_a_la2),len(gender_h_la2)])
   print (proportions_ztest(counts_la2,nobs_la2))
  
