import numpy as np
import matplotlib.pyplot as plt 
from path_read import *
import pyabf
import pyabf.filter
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

dname_xls = '/Users/closeyes/Library/Mobile Documents/com~apple~CloudDocs/[02]CURRENT_and_FUTURE/____URGENT_____/EPSC_analysis/'
f_n_xls='data_org.xlsx'
fname_xls = dname_xls+f_n_xls
global x
x=pd.read_excel(fname_xls, sheet_name = 0, header=0)
rownum = x.shape[0]

dname = '/Users/closeyes/Library/Mobile Documents/com~apple~CloudDocs/[02]CURRENT_and_FUTURE/____URGENT_____/EPSC_analysis/raw_data/' # Arrangement file

def exp_func2(X,A,tau) :
    return A*(1-np.exp(-X/tau))

def exp_func(X,A,tau,B,tau2,C,D) :
    return A-D*np.exp(-X/tau)-B*np.exp((X-C)/tau2)

def risekin_cal(a,i) :
    t = a.sweepX*1000
    t_local = t[4000:8000]
    signal_tot = np.array([])
    swpL = np.array(x.iloc[i,5:15].dropna(),dtype=np.int32)
    use_swpL_num = np.shape(swpL)[0]
    for amm,sweepNum in enumerate(swpL):
        a.setSweep(sweepNum)

        if amm == 0 :
            signal_tot = a.sweepY[:]
            
        else :
            signal_tot = np.vstack((signal_tot,a.sweepY))

    signal_local = signal_tot[:,4000:8000]
    baseline = np.mean(signal_tot[:,4500:5500], axis=1)
    b_adj_local = signal_local - np.resize(baseline,(np.size(baseline),1))    
    cm = plt.get_cmap("hsv")
    colors = [cm(x/use_swpL_num) for x in swpL]

    rise_t = np.array([])

    #plt.figure()
    n_fit_cnt=0
    for m in range(use_swpL_num) :
        st_t_id = np.abs(b_adj_local[m,2010:2250]).argmin() + 2010
        yhat = savgol_filter(b_adj_local[m, st_t_id: (st_t_id+300)], 51, 3) 

        amp = np.min(yhat)
        amp_t_id = np.argmin(yhat)
        
        t_new = t[(4000+st_t_id):(4001+st_t_id+amp_t_id)]-t[st_t_id+4000]
        signal_new = np.abs(b_adj_local[m,(st_t_id):(st_t_id+amp_t_id+1)])
        #print(i,m)
        #try:
        #    popt,pcov = curve_fit(exp_func, t_new, signal_new)
        #    if (popt[0]<=0) or (popt[1]<=0):
        #        popt,pcov = curve_fit(exp_func2, t_new, signal_new)
            
        #except :
        popt,pcov = curve_fit(exp_func2, t_new, signal_new)
        
        print(popt)
        if m == 0 :
            rise_t = popt[1]
        else :
            rise_t = np.append(rise_t,popt[1])

    #print(popt)

        #plt.plot(t_new, signal_new, color=colors[m], alpha=0.7, label='sweep'+str(swpL[m]), lw=0.5)
        #plt.plot(t_new, exp_func2(t_new, *popt), color=colors[m], alpha=0.7, label='fit'+str(swpL[m]), lw=0.7)
        #plt.plot(t_new, np.abs(yhat[0:amp_t_id+1]),color=colors[m], alpha=0.7, label='sweep'+str(swpL[m]), lw=0.7)
        #plt.plot(t[4000+st_t_id+amp_t_id]-t[st_t_id+4000],np.abs(amp), 'k*')

    #plt.legend()

    return np.mean(rise_t), np.std(rise_t)/np.sqrt(use_swpL_num)
    

r_t = np.array([])

for i in range(rownum) :

    f_n = x['File'].iloc[i]
    fname = dname+f_n
    abf = pyabf.ABF(fname) 
    r,r_std = risekin_cal(abf,i)
    if i==0:
        r_t = r
    else :
        r_t = np.append(r_t,r)


id_e = np.arange(0,20,2)
id_o = np.arange(1,20,2)

far = r_t[id_e]
near = r_t[id_o]

tot_res = pd.DataFrame({'Far':far, 'Near':near})
tot_res.to_excel(dname+'result.xlsx')

#    abf = pyabf.ABF(fname)
#    num_rep = abf.sweepCount
#    t=abf.sweepX*1000
#    num_t_pts = len(t)
#cm = plt.get_cmap("hsv")
#colors = [cm(x/num_rep) for x in abf.sweepList]
#
#t_local = t[4000:8000]
#
#signal_tot = np.array([])
#
#for sweepNum in abf.sweepList :
#    abf.setSweep(sweepNum)
#    if sweepNum == 0 :
#        signal_tot = abf.sweepY[:]
#    else :
#        signal_tot = np.vstack((signal_tot,abf.sweepY))
#
#signal_local = signal_tot[:,4000:8000]
#baseline = np.mean(signal_tot[:,4500:5500], axis=1)
#b_adj_local = signal_local - np.resize(baseline,(np.size(baseline),1))
#
#plt.figure()
#
#for sweepNum in abf.sweepList :
#
#    plt.plot(t_local,b_adj_local[sweepNum,:],color=colors[sweepNum], alpha=.7, label='sweep'+str(sweepNum), lw=0.5)
#
#plt.plot(t_local,np.zeros((np.shape(signal_local)[1],)),'k--',lw=2,alpha = 0.7)
#plt.ylim([-200, 200])
#plt.legend()
#
#

