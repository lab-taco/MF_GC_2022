import numpy as np
from spatanalysis import *

class spat_arr :

    def __init__(self, arena_crd, fluo_crd, pt_num):
        
        self.arena_c = arena_crd
        self.expr_c = fluo_crd

        self.sitenum = np.shape(arena_crd)[0]
        self.exprnum = np.shape(fluo_crd)[0]
        
        self.site_list = np.arange(self.sitenum)

        # Whole GC layer in the image ROI : Arena        
        self.arena_h = hull_gen(self.arena_c)
        self.pol_arena = Polygon(self.arena_h)
        self.arena_area = self.pol_arena.area 
        
        # G function of the data
        self.G_nn, self.Gcdf = G_fn(self.expr_c)
        
        # Function of the data
        self.Fptnum = pt_num
        self.F_nn, self.Fcdf = F_fn(self.arena_h, self.expr_c, self.Fptnum)
        
        # K function of the data
        self.d_for_K, self.K_d = K_fn(self.arena_h, self.expr_c)

    def rnd_config(self) :

        self.rnd_conf_list = np.random.choice(self.site_list, size=self.exprnum, replace=False)
        self.rnd_conf_crd = self.arena_c[self.rnd_conf_list,:]

    def clst_config(self, cl_r, rep) :

        self.rnd_config()
        fl_list = self.rnd_conf_list[:]
        nfl_list = np.setdiff1d(self.site_list,fl_list)

        self.perm_exc_num = 0

        for i in range(rep) :

            f_pick = np.random.choice(fl_list)
            nf_pick = np.random.choice(nfl_list)

            dist_ffp = dist_vec (self.arena_c[fl_list,:], self.arena_c[f_pick,:])
            dist_fnfp = dist_vec (self.arena_c[fl_list,:], self.arena_c[nf_pick,:])
            dist_nffp = dist_vec (self.arena_c[nfl_list,:], self.arena_c[f_pick,:])
            dist_nfnfp = dist_vec (self.arena_c[nfl_list,:], self.arena_c[nf_pick,:])

            if dist_nffp[dist_nffp<=cl_r].size :
                score_f = dist_ffp[dist_ffp<=cl_r].shape[0]/dist_nffp[dist_nffp<=cl_r].shape[0]
            else :
                score_f = 50000000

            if dist_nfnfp[dist_nfnfp<=cl_r].size :
                score_nf = dist_fnfp[dist_fnfp<=cl_r].shape[0]/dist_nfnfp[dist_nfnfp<=cl_r].shape[0]
            else :
                score_nf = 50000000

            if score_f < score_nf :
                
                fl_list[fl_list==f_pick] = nf_pick
                nfl_list[nfl_list==nf_pick] = f_pick
                
                self.perm_exc_num+=1

        self.clst_conf_list = fl_list[:]
        self.clst_conf_crd = self.arena_c[self.clst_conf_list,:]


    def G(self) :
        self.G_nn_r, self.Gcdf_r = G_fn(self.rnd_conf_crd)
        self.G_nn_cl, self.Gcdf_cl = G_fn(self.clst_conf_crd)

    def F(self) :
        self.F_nn_r, self.Fcdf_r = F_fn(self.arena_h, self.rnd_conf_crd, self.Fptnum)
        self.F_nn_cl, self.Fcdf_cl = F_fn(self.arena_h, self.clst_conf_crd, self.Fptnum)
        
    def K(self) :
        self.d_for_K_r, self.K_d_r = K_fn(self.arena_h, self.rnd_conf_crd)
        self.d_for_K_c, self.K_d_c = K_fn(self.arena_h, self.clst_conf_crd)
