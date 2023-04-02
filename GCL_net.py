import numpy as np
import copy
import networkx as nx
#import matplotlib.pyplot as plt
#from CalStats import *

class GC_MF_net :

    def __init__(self, GC_num, MF_num) :

        self.GC_n = GC_num
        self.MF_n = MF_num

        self.GC_node_color = []
        self.MF_node_color = []
        self.e_color = []

        self.e_list = []
        self.e_list_rand = []
        self.GC_list_in_MF_node = []
        self.GC_list_in_MF_node_rand = []
        self.netmodi_log = ''

        self.set_random_net()
        #self.GC_act = []
        #self.MF_act = []
        #self.weight = []

    def set_init(self):
        self.__init__(self.GC_n, self.MF_n)

    def set_random_net(self, **kwrds) : 

        if self.netmodi_log != '' :
            print("Reinitializing everything")
            previous = int(self.netmodi_log[0:3])
            self.set_init()
            previous+=1
            self.netmodi_log = str(previous).zfill(3)
        else :
            self.netmodi_log = '001'

        if ('gc_d_num' in kwrds) :
            gcdnum = np.int32(kwrds['gc_d_num'])
        else :
            gcdnum = 4

        for x in range(self.GC_n):
            ch = np.random.choice(np.arange(1, self.MF_n + 1),size=gcdnum, replace=False)

            for i in range(gcdnum) :
                self.e_list.append([x+1,ch[i]]) # [GC, MF]
        
        el = np.array(self.e_list)

        for m in range(self.MF_n) :
            self.GC_list_in_MF_node.append(el[np.where(el[:,1]==(m+1)),0].tolist()[0])

        self.e_num = np.shape(el)[0]
        self.e_list_in_array = el

        self.e_list_rand = copy.deepcopy(self.e_list)
        self.GC_list_in_MF_node_rand = copy.deepcopy(self.GC_list_in_MF_node)

        print('Random network : ' + self.netmodi_log + ' is generated.')
    
    def gc_swap(self,edge_pick_idx) :
        
        gc1=self.e_list[edge_pick_idx[0]][0]
        gc2=self.e_list[edge_pick_idx[1]][0]
        mf1=self.e_list[edge_pick_idx[0]][1]
        mf2=self.e_list[edge_pick_idx[1]][1]

        self.e_list[edge_pick_idx[0]][0]=gc2
        self.e_list[edge_pick_idx[1]][0]=gc1
        
        self.e_list.sort()

        self.GC_list_in_MF_node[mf1-1][self.GC_list_in_MF_node[mf1-1].index(gc1)]=gc2
        self.GC_list_in_MF_node[mf2-1][self.GC_list_in_MF_node[mf2-1].index(gc2)]=gc1
        self.GC_list_in_MF_node[mf1-1].sort()
        self.GC_list_in_MF_node[mf2-1].sort()

    @staticmethod
    def score_func(gcl, gc_pick) :

        gc1=gcl[0][:]
        gc2=gcl[1][:]
        gc1.remove(gc_pick[0])
        gc2.remove(gc_pick[1])
        gcl_1 = np.array(gc1)
        gcl_2 = np.array(gc2)

        gcl_1_before = np.mean(np.abs(gcl_1-gc_pick[0]))
        gcl_2_before = np.mean(np.abs(gcl_2-gc_pick[1]))
        gcl_1_after = np.mean(np.abs(gcl_1-gc_pick[1]))
        gcl_2_after = np.mean(np.abs(gcl_2-gc_pick[0]))

        before_score = gcl_1_before + gcl_2_before
        after_score = gcl_1_after + gcl_2_after

        return before_score, after_score

    def rewire(self, mode, iter_num) :

        if mode[1] =='.' :
            self.e_list = copy.deepcopy(self.e_list_rand)
            self.GC_list_in_MF_node = copy.deepcopy(self.GC_list_in_MF_node_rand)

        for _ in range(iter_num) :

            ch_2edge_num = np.random.choice(np.arange(0,self.e_num),size=2,replace=False)
            ch_2edge = [self.e_list[ch_2edge_num[0]], self.e_list[ch_2edge_num[1]]]
            ch_2gc = [ch_2edge[0][0], ch_2edge[1][0]]
            ch_2mf = [ch_2edge[0][1], ch_2edge[1][1]]
            ch_gcs = [self.GC_list_in_MF_node[ch_2mf[0]-1][:], self.GC_list_in_MF_node[ch_2mf[1]-1][:]]

            if ((ch_2mf[0] != ch_2mf[1]) and (ch_2gc[0] != ch_2gc[1])):
                
                before, after = GC_MF_net.score_func(ch_gcs, ch_2gc)

                target_check_1 = not (ch_2gc[1] in ch_gcs[0])
                target_check_2 = not (ch_2gc[0] in ch_gcs[1])
                target_check = (target_check_1) & (target_check_2)

                if mode[0] == 'c' : check_do=(before>after)
                elif mode[0] == 'a' : check_do=(before<after)                
                else : check_do = False

                if check_do and target_check : self.gc_swap(ch_2edge_num)

        if mode[1] == '+' :
            history = self.netmodi_log[3:]
            self.netmodi_log = self.netmodi_log + '_' + mode[0] + str(iter_num).zfill(10)
            print('Random network : ' + self.netmodi_log[0:3] + ' , Rewire mode : ' + mode[0] + ' , iteration : ' + str(iter_num) + ' , with history : ' + history)
        elif mode[1] == '.' :
            self.netmodi_log = self.netmodi_log[0:3] + '_' + mode[0] + str(iter_num).zfill(10)
            print('Random network : ' + self.netmodi_log[0:3] + ' , Rewire mode : ' + mode[0] + ' , iteration : ' + str(iter_num))
        
    def labeling(self, target, n_dual, r_st, r_ed, g_st, g_ed) :

        n_dual_h=np.int32(np.round(n_dual/2))

        r_label = ['']*self.GC_n
        g_label = ['']*self.GC_n
    
        r_label[(r_st-1):(r_ed-n_dual_h)]=['r']*(r_ed-r_st+1-2*n_dual_h)+['y']*n_dual_h
        g_label[(g_st-1+n_dual_h):(g_ed)]=['y']*n_dual_h+['g']*(g_ed-g_st+1-2*n_dual_h)

        color = ['']*self.GC_n

        for i in range(self.GC_n):

            if ((r_label[i] == '') and (g_label[i] =='')):
                color[i] = 'k'

            elif ((r_label[i] != '') and (g_label[i] == '')):
                color[i] = r_label[i]

            elif ((r_label[i] == '') and (g_label[i] != '')):
                color[i] = g_label[i]

            else:
                color[i] = 'y'

        self.GC_node_color = color
        self.MF_node_color = ['b']*self.MF_n

        if target == 'r' :

            for _, x in enumerate(self.e_list_rand) :

                if self.GC_node_color[x[0]-1] == 'r' :
                    self.e_color.append('r')
                elif self.GC_node_color[x[0]-1] == 'g' :
                    self.e_color.append('g')
                elif self.GC_node_color[x[0]-1] == 'y' :
                    self.e_color.append('y')
                else :
                    self.e_color.append('k')

        else : 
            for _, x in enumerate(self.e_list) :

                if self.GC_node_color[x[0]-1] == 'r' :
                    self.e_color.append('r')
                elif self.GC_node_color[x[0]-1] == 'g' :
                    self.e_color.append('g')
                elif self.GC_node_color[x[0]-1] == 'y' :
                    self.e_color.append('y')
                else :
                    self.e_color.append('k')

    def clear_labeling(self) :
        self.GC_node_color=['k'] * self.GC_n
        self.MF_node_color=['k'] * self.MF_n 
        self.e_color=['k'] * self.e_num

    def name_and_position(self):

        self.pos=dict()
    
        a=np.arange(self.GC_n) + 1
        self.GC_names = ['GC'+ str(x).zfill(3) for x in a]
        self.pos.update((n,(i,1)) for i,n in enumerate(self.GC_names))

        b=np.arange(self.MF_n)+1
        self.MF_names = ['MF'+str(x).zfill(3) for x in b]
        self.pos.update((n2,((j+0.5)*3,0)) for j,n2 in enumerate(self.MF_names))

    def net_drawing(self,target) :

        self.name_and_position()
        NT = nx.Graph()
        NT.add_nodes_from(self.GC_names, bipartite=0)
        NT.add_nodes_from(self.MF_names, bipartite=1)

        if target == 'r' :
            NT.add_edges_from([('GC'+str(n[0]).zfill(3),'MF'+str(n[1]).zfill(3)) for _,n in enumerate(self.e_list_rand)])
        else :
            NT.add_edges_from([('GC'+str(n[0]).zfill(3),'MF'+str(n[1]).zfill(3)) for _,n in enumerate(self.e_list)])
        
        options = {
            'with_labels' : False ,
            'nodelist' : self.GC_names+self.MF_names, #,
            'node_color' : self.GC_node_color+self.MF_node_color ,
            'edge_color' : self.e_color
        }

        nx.draw_networkx(NT, pos=self.pos, **options)
        

    def stats(self, **kwd) :
        
        if ('sigma' in kwd) :
            sigma = kwd['sigma']
        else :
            sigma = 0

        #----- Degree -----#

        if ('degree' in kwd) and (kwd['degree'] == True) :
            self.cnt_in_mf = np.zeros((self.MF_n,5))

            for _,x in enumerate(self.e_list) :
        
                if self.GC_node_color[x[0]-1] == 'r' :
                    self.cnt_in_mf[(x[1]-1),0]+=np.random.normal(1.0,sigma)
                
                elif self.GC_node_color[x[0]-1] == 'g' :
                    self.cnt_in_mf[(x[1]-1),1]+=np.random.normal(1.0,sigma)
                
                elif self.GC_node_color[x[0]-1] == 'y' :
                    a=np.random.normal(1.0,sigma)
                    self.cnt_in_mf[(x[1]-1),0]+=a
                    self.cnt_in_mf[(x[1]-1),1]+=a
                    self.cnt_in_mf[(x[1]-1),3]+=a            
                
                else :
                    self.cnt_in_mf[(x[1]-1),2]+=np.random.normal(1.0,sigma)
            
            self.tot_deg = self.cnt_in_mf[:,0] + self.cnt_in_mf[:,1] - self.cnt_in_mf[:,3] + self.cnt_in_mf[:,2]
        
        #---- Ratio ----#

        if ('ratio' in kwd) and (kwd['ratio']==True) :
            self.stat_ratio = (self.cnt_in_mf[:,0] + np.random.lognormal(0.3,0.1,self.MF_n))/(self.cnt_in_mf[:,1] + np.random.lognormal(0.3,0.1,self.MF_n))
        
        



#        MF_node_gc_color_cnt = np.zeros((MF_num,4))
#
#        all_edge_color = []
#
#        for m,x in enumerate(all_edge_list) :
#
#            if GC_node_color[x[0]-1] == 'r' :
#                MF_node_gc_color_cnt[x[1]-1][0]+=np.random.normal(1.0,sigma)
#                all_edge_color.append('r')
#            elif GC_node_color[x[0]-1] == 'g' :
#                MF_node_gc_color_cnt[x[1]-1][1]+=np.random.normal(1.0,sigma)
#                all_edge_color.append('g')
#            elif GC_node_color[x[0]-1] == 'y' :
#                a=np.random.normal(1.0,sigma)
#                MF_node_gc_color_cnt[x[1]-1][0]+=a
#                MF_node_gc_color_cnt[x[1]-1][1]+=a
#                MF_node_gc_color_cnt[x[1]-1][3]+=a            
#                all_edge_color.append('y')
#            else :
#                MF_node_gc_color_cnt[x[1]-1][2]+=np.random.normal(1.0,sigma)
#                all_edge_color.append('k')    
#
#    def clear_labeling() :
    
#    def stats() :

