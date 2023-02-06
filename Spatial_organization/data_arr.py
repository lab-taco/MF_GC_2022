from configurations import *
from spatanalysis import *
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

# Theoretical G and F functions for Poisson point process
def G_F_poisson (lmbda, r) :
    return 1-np.exp(-lmbda*np.pi*(r**2))

# Integration of G, F function data across several images
def G_F_data(im_num, im_dict_all, im_dict_gr) :

    G_d = np.array([])
    G_r = np.array([])
    G_c = np.array([])
    F_d = np.array([])
    F_r = np.array([])
    F_c = np.array([])
    
    arena_area_collection = np.array([])
    arena_cell_collection = np.array([])
    fluo_collection = np.array([])

    for im in range(im_num):

        arena_crd = im_dict_all[im]
        fluo_crd = im_dict_gr[im]

        I = spat_arr(arena_crd, fluo_crd, 200)

        for i in range(0,10):
            I.clst_config(4,2000)
            I.G()
            I.F()

            if ((im == 0) and (i == 0)):
                G_r = I.G_nn_r
                G_c = I.G_nn_cl
                F_r = I.F_nn_r
                F_c = I.F_nn_cl
            else :
                G_r = np.append(G_r,I.G_nn_r)
                G_c = np.append(G_c,I.G_nn_cl)
                F_r = np.append(F_r,I.F_nn_r)
                F_c = np.append(F_c,I.F_nn_cl)

        if im == 0 :
            G_d = I.G_nn
            F_d = I.F_nn
            arena_area_collection = I.arena_area
            arena_cell_collection = I.sitenum
            fluo_collection = I.exprnum

        else :
            G_d = np.append(G_d, I.G_nn)
            F_d = np.append(F_d, I.F_nn)
            arena_area_collection = np.append(arena_area_collection, I.arena_area)
            arena_cell_collection = np.append(arena_cell_collection, I.sitenum)
            fluo_collection = np.append(fluo_collection, I.exprnum)

    Gcdf_d = ECDF(G_d)
    Gcdf_r = ECDF(G_r)
    Gcdf_c = ECDF(G_c)
    Fcdf_r = ECDF(F_r)
    Fcdf_c = ECDF(F_c)
    Fcdf_d = ECDF(F_d)

    maxr = np.max(np.array([np.max(Gcdf_d.x), np.max(Gcdf_r.x), np.max(Gcdf_c.x), np.max(Fcdf_d.x), np.max(Fcdf_r.x), np.max(Fcdf_c.x)]))
    r = np.arange(0,np.int8(maxr),0.05)

    G_r_i = interp1d(Gcdf_r.x, Gcdf_r.y, bounds_error=False, fill_value=1.0)
    G_d_i = interp1d(Gcdf_d.x, Gcdf_d.y, bounds_error=False, fill_value=1.0)
    G_c_i = interp1d(Gcdf_c.x, Gcdf_c.y, bounds_error=False, fill_value=1.0)

    F_r_i = interp1d(Fcdf_r.x, Fcdf_r.y, bounds_error=False, fill_value=1.0)
    F_c_i = interp1d(Fcdf_c.x, Fcdf_c.y, bounds_error=False, fill_value=1.0)
    F_d_i = interp1d(Fcdf_d.x, Fcdf_d.y, bounds_error=False, fill_value=1.0)

    info = [arena_area_collection, arena_cell_collection, fluo_collection, np.int8(maxr)]
    cdfs = [Gcdf_d, Gcdf_r, Gcdf_c, Fcdf_d, Fcdf_r, Fcdf_c]
    interps = [G_d_i(r), G_r_i(r), G_c_i(r), F_d_i(r), F_r_i(r), F_c_i(r)]

    return info, cdfs, interps

# Integration of K function data across several images
def K_data(im_num, im_dict_all, im_dict_gr) :
    
    K_d_all = np.array([])
    K_r_all = np.array([])
    K_c_all = np.array([])

    for im in range(im_num):

        arena_crd = im_dict_all[im]
        fluo_crd = im_dict_gr[im]

        I = spat_arr(arena_crd, fluo_crd, 200)

        for i in range(0,10):
            I.clst_config(8,50)
            I.K()

            if ((im == 0) and (i == 0)):
                K_r_all = I.K_d_r
                K_c_all = I.K_d_c
            else :
                K_r_all = np.vstack((K_r_all,I.K_d_r))
                K_c_all = np.vstack((K_c_all,I.K_d_c))
            
        if im == 0 :
            K_d_all = I.K_d
        else :
            K_d_all = np.vstack((K_d_all, I.K_d))
            
    r = I.d_for_K
    raw_data = [K_d_all, K_r_all, K_c_all]
    mean_curve = [np.mean(K_d_all, axis=0), np.mean(K_r_all, axis=0), np.mean(K_c_all, axis=0)]
    
    return r,raw_data, mean_curve
