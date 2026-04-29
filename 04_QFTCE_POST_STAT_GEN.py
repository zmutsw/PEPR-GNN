
# coding: utf-8

# In[1]:


import numpy as np
import scanpy as sc
import scvi as sv
import pandas as pd
import pickle as pkl
import os
import seaborn as sb
import scipy.sparse as csr
from itertools import combinations
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from scipy.stats import gaussian_kde
import scipy.stats as ss
import math
from timeit import default_timer
import multiprocessing
from multiprocessing import Pool
from scipy.stats import percentileofscore
import sys
import random
import networkx as nx

# set a working directory for saving plots
os.chdir('/project/GCRB/Hon_lab/s437603/data/ghmt_multiome/analysis')


# ## LOAD IN FILES

# In[2]:


# lib = 'bl'
lib = 'ln'

#ORIGINAL FILTERED LIBS
rna_lib = sc.read(lib+'rna_filtered.h5ad')
atac_lib = sc.read(lib+'atac_filtered.h5ad')
pert_lib = pkl.load(open(lib+'pert_filtered.pkl','rb'))
ccc = pkl.load(open(lib+'ccc.pkl','rb'))

# [tss_ind,rna_ind,[atac_inds]] for enhol
# [tss_ind,rna_ind,atac_ind] for promol
# load in new prom_overlaps and promenh_overlaps with 1000bp offset
promol = pkl.load(open(lib+'1000ce_prom_overlaps.pkl','rb'))
promenhol = pkl.load(open(lib+'1000ce_promenh_overlaps.pkl','rb'))
enhol = pkl.load(open(lib+'1000ce_enh_overlaps.pkl','rb'))

# LOAD IN CE LIBS
ce_rna = pkl.load(open(lib+'nce_rna.pkl','rb'))                   #ce_rna but CPM normalized
zce_rna = pkl.load(open(lib+'zce_rna.pkl','rb'))                  #nce_rna but z-scored
ce_atac = pkl.load(open(lib+'nce_atac.pkl','rb'))                 #ce_atac but binarized and then CPM normalized
ce_tss = pkl.load(open(lib+'ce_tss.pkl','rb'))

# LOAD IN FULL STATS FILES,    each element is in form [pv,qv,fc]
unflat_te = pkl.load(open('total_'+lib+'_perturbationenhancer_stats.pkl','rb'))
te = [[stats for pert in enh for stats in pert] for enh in unflat_te] #flatten by one dimension
tp = pkl.load(open('total_'+lib+'_perturbationpromoter_stats.pkl','rb'))
tr = pkl.load(open('total_'+lib+'_perturbationrna_stats.pkl','rb'))
ep = pkl.load(open('total_'+lib+'_enhancerpromoter_stats.pkl','rb'))
er = pkl.load(open('total_'+lib+'_enhancerrna_stats.pkl','rb'))
pr = pkl.load(open('total_'+lib+'_promoterrna_stats.pkl','rb'))


# In[5]:


#for making list of all used genes and enhancers

all_tss = list(ce_tss.loc[:,'gid'])
gl = []
for entry in promol:
    tss_ind = entry[0]
    gene = all_tss[tss_ind]
    gl.append(gene)

all_e = list(ce_atac.var_names)
el = []
for entry in enhol:
    this_tss = []
    for enh in entry[2]:
        e_ind = enh
        this_tss.append(all_e[e_ind])
    el.append(this_tss)

with open(lib+'genelist.pkl','wb') as file:
    pkl.dump(gl,file)
with open(lib+'enhlist.pkl','wb') as file:
    pkl.dump(el,file)


# ## PREPARE PERBOOL AND IDXBOOL

# In[6]:


# returns a boolean array for every comparison in a given matrix, with a given statistic ('pval' or 'qval'), and a given threshold value
def edge_bool_getter(mat,stat,thresh):
    
    #convert mat to 2d array, where every row is a comparison, and columns are [pval,qval,fc]
    matl = []
    for tss in mat:
        tssl = []
        for comparison in tss:
            tssl.append(comparison)
        matl+=tssl
    mata = np.asarray(matl)

    #identify comparison column
    if stat == 'p':
        stat_col = 0
    if stat == 'q':
        stat_col = 1
        
    #set threshold if bonferroni is specified
    if thresh == 'bonferroni':
        thresh = 0.05/len(mata)
    
    #initialize and populate edge_bool
    edge_bool = np.zeros(shape=(len(mata),1))
    for row in range(len(mata)):
        if mata[row,stat_col] < thresh:
            edge_bool[row,0] = 1
    
    return edge_bool


# In[7]:


stat,thresh = 'p','bonferroni'
te_bool = edge_bool_getter(te,stat,thresh)
tp_bool = edge_bool_getter(tp,stat,thresh)
tr_bool = edge_bool_getter(tr,stat,thresh)
ep_bool = edge_bool_getter(ep,stat,thresh)
er_bool = edge_bool_getter(er,stat,thresh)
pr_bool = edge_bool_getter(pr,stat,thresh)


# In[ ]:


#make fullbool, scanning for all full networks
idxbool = np.zeros(shape=(len(te_bool),6))
fullbool = np.zeros(shape=(len(te_bool),1))

elist = [len(elem[2]) for elem in enhol]
pr_num = 0
split = 16 #change to 16 if doing full pert or 5 if doing degree pert
for per in range(len(te_bool)):
    te_num = per
    ep_num = int(per/split)
    if ep_num == np.sum(elist[0:pr_num+1]) or elist[pr_num] == 0:
        pr_num += 1
    tp_num = pr_num*split+per%split
    tr_num = pr_num*split+per%split
    er_num = int(per/split)
    
    idxbool[per,0] = te_num
    idxbool[per,1] = tp_num
    idxbool[per,2] = tr_num
    idxbool[per,3] = ep_num
    idxbool[per,4] = er_num
    idxbool[per,5] = pr_num
    
#     if te_bool[te_num] and tp_bool[tp_num] and tr_bool[tr_num] and ep_bool[ep_num] and er_bool[er_num] and pr_bool[pr_num]:
    if ep_bool[ep_num] and er_bool[er_num] and pr_bool[pr_num]:
        fullbool[per] = 1

with open(lib+'_'+stat+str(thresh)+'_eprbool.pkl','wb') as pfile:
    pkl.dump(fullbool,pfile)
with open(lib+'_'+stat+str(thresh)+'_eprbool_idx.pkl','wb') as pfile:
    pkl.dump(idxbool,pfile)


# In[37]:


# lib = 'bl'
lib = 'ln'
cb = pkl.load(open(lib+'_pbonferroni_completebool.pkl','rb'))
cb_idx = pkl.load(open(lib+'_pbonferroni_completebool_idx.pkl','rb'))
promol = pkl.load(open(lib+'1000ce_prom_overlaps.pkl','rb'))
enhol = pkl.load(open(lib+'1000ce_enh_overlaps.pkl','rb'))
elist = [len(elem[2]) for elem in enhol]
print(np.sum(cb))
gl = [ce_rna.var_names[promol[int(cb_idx[cell][5])][1]] for cell in range(len(cb_idx)) if cb[cell]]
print(len(gl),len(set(gl)))
com_idx = np.array([cb_idx[which_elem] for which_elem in range(len(cb_idx)) if cb[which_elem]])


# In[10]:


#for plotting perturbation position
sb.histplot(com_idx[:,0]%16,bins=16,binrange=(0,16))


# In[4]:


arr = np.asarray(tr)


# In[48]:


#NOTA BENE: distributions are skewed for each pert of course, but nothing crazy, mu is surprisingly reasonable looking.
pert_num = 15
fcl_full = []
fcl_bon = []
for tss in arr:
    fcl_full.append(np.log2(tss[pert_num][2]+2**-5))
    if tss[pert_num][0] < 0.05/len(arr):
        fcl_bon.append(np.log2(tss[pert_num][2]+2**-5))
fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(50,15))
print(len(fcl_full),' out of ',len(arr),' with mu=',2**np.mean(fcl_full))
print(len(fcl_bon),' out of ',len(arr),' with mu=',2**np.mean(fcl_bon))
sb.histplot(fcl_full,ax=axs[0])
sb.histplot(fcl_bon,ax=axs[1])


# In[62]:


pert_num = 12
fcl = []
for tss in arr:
    if tss[pert_num][0] < 0.05/len(arr):
        fcl.append(np.log2(tss[pert_num][2]+2**-5))
print(len(fcl),' out of ',len(arr))
sb.histplot(fcl)


# In[63]:


fcl = []
for tss in arr:
    if tss[pert_num][0] < 1:
        fcl.append(np.log2(tss[pert_num][2]+2**-5))
print(len(fcl),' out of ',len(arr))
sb.histplot(fcl)


# In[104]:


elist = [len(elem[2]) for elem in enhol]


# In[110]:


eidx = []
for elem in range(len(com_idx)):
    


# In[104]:


#for plotting enhancer position
sb.histplot(com_idx[:,3]%24,bins=100,binrange=(0,100))


# ## LOOK AT SPECIFIC GRAPHS

# In[45]:


# lib = 'bl'
# lib = 'ln'
lib = 'co'

#ORIGINAL FILTERED LIBS
rna_lib = sc.read(lib+'rna_filtered.h5ad')
atac_lib = sc.read(lib+'atac_filtered.h5ad')
pert_lib = pkl.load(open(lib+'pert_filtered.pkl','rb'))
ccc = pkl.load(open(lib+'ccc.pkl','rb'))
#nc = 1426, ghmt = 626

# [tss_ind,rna_ind,[atac_inds]] for enhol
# [tss_ind,rna_ind,atac_ind] for promol
# load in new prom_overlaps and promenh_overlaps with 1000bp offset
promol = pkl.load(open(lib+'1000ce_prom_overlaps.pkl','rb'))
promenhol = pkl.load(open(lib+'1000ce_promenh_overlaps.pkl','rb'))
enhol = pkl.load(open(lib+'1000ce_enh_overlaps.pkl','rb'))

# LOAD IN CE LIBS
ce_rna = pkl.load(open(lib+'nce_rna.pkl','rb'))                   #ce_rna but CPM normalized
zce_rna = pkl.load(open(lib+'zce_rna.pkl','rb'))                  #nce_rna but z-scored
ce_atac = pkl.load(open(lib+'nce_atac.pkl','rb'))                 #ce_atac but binarized and then CPM normalized
ce_tss = pkl.load(open(lib+'ce_tss.pkl','rb'))

# LOAD IN FULL STATS FILES,    each element is in form [pv,qv,fc]
unflat_te = pkl.load(open('total_'+lib+'_perturbationenhancer_stats.pkl','rb'))
# te = [[stats for pert in enh for stats in pert] for enh in unflat_te] #flatten by one dimension
te = unflat_te
tp = pkl.load(open('total_'+lib+'_perturbationpromoter_stats.pkl','rb'))
tr = pkl.load(open('total_'+lib+'_perturbationrna_stats.pkl','rb'))
ep = pkl.load(open('total_'+lib+'_enhancerpromoter_stats.pkl','rb'))
er = pkl.load(open('total_'+lib+'_enhancerrna_stats.pkl','rb'))
pr = pkl.load(open('total_'+lib+'_promoterrna_stats.pkl','rb'))


# In[18]:


cbool = pkl.load(open('bl_q5_completebool.pkl','rb'))
cbool_idx = pkl.load(open('bl_q5_completebool_idx.pkl','rb'))

idx_counter = 0
full_graph_stats = []
for which_graph in range(len(cbool)):
    if cbool[which_graph]:
        ids = cbool_idx[idx_counter]
        

np.sum(cbool)


# In[3]:


#total regualtory element number
e_total = [len(enh[2]) for enh in enhol]
np.sum(e_total)+len(pr)


# In[4]:


smh1 = pkl.load(open('bl_pbonferroni_completebool.pkl','rb'))
smh2 = pkl.load(open('ln_pbonferroni_completebool.pkl','rb'))

print(np.sum(smh1),np.sum(smh2))


# ## SET UP GENE MAPS

# In[46]:


def tree_assembler(gene_name,t_split):
    #define gene and gather needed indices
    tss_pos = np.where(ce_tss['gid'] == gene_name)[0]
    gene_promol = [elem for elem in promol if elem[0] in tss_pos][0] #specifies first TSS for a given gene
    gene_enhol = [elem for elem in enhol if elem[0] in tss_pos][0] #specifies first TSS for a given gene
    promol_ind = promol.index(gene_promol)
    rna_ind = gene_promol[1]
    prom_ind = gene_promol[2]
    enh_inds = gene_enhol[2]
#     print(ce_atac.var_names[prom_ind])
#     for ind in enh_inds:
#         print(ce_atac.var_names[ind])
    elist = [len(elem[2]) for elem in enhol]

    #for handling genes with no associated enhancers
    if len(enh_inds) == 0:
        return [],[]
    
    #subset pqfc files
    te_sub,tp_sub,tr_sub = te[promol_ind],np.array(tp[promol_ind]),np.array(tr[promol_ind])
    te_sub = np.array([edge for enh in te_sub for edge in enh])     #flatten te_sub
    ep_sub,er_sub,pr_sub = np.array(ep[promol_ind]),np.array(er[promol_ind]),np.array(pr[promol_ind])
    
    print('so far so good')
    
    all_trees = []
    #assemble into p and fc matrices_________________________________________________________________________________________________________________
    for val in range(2): #0=p, 1=fc
        e_split = len(enh_inds)
        te_slice,tp_slice,tr_slice = te_sub[:,val],tp_sub[:,val],tr_sub[:,val]
        ep_slice,er_slice,pr_slice = ep_sub[:,val],er_sub[:,val],pr_sub[:,val]
        #extend all non-t by t_split
        ep_slice = np.repeat(ep_slice,t_split)
        er_slice = np.repeat(er_slice,t_split)
        pr_slice = np.repeat(pr_slice,t_split)
        #extend all non-e by e_split
        tp_slice = np.repeat(tp_slice,e_split)
        tr_slice = np.repeat(tr_slice,e_split)
        pr_slice = np.repeat(pr_slice,e_split)
        #concat all slices 
        slices = np.array([te_slice,tp_slice,tr_slice,ep_slice,er_slice,pr_slice])
        new_tree = np.transpose(slices)
        
        #transform if using fc
        if val == 1:
            new_tree = np.log2(new_tree+2**-5)
        
        #add to final output
        all_trees.append(new_tree)
    
    #                p           fc
    return all_trees[0],all_trees[1]

def plot_getter():
    num_enh = int(len(p_tree)/16)
    edge_names = ['te','tp','tr','ep','er','pr']
    fig,ax = plt.subplots(figsize=(30,50),ncols=3,nrows=2)
    p_t = p_tree[0::num_enh]
    fc_t = fc_tree[0::num_enh]

    #optional, for filtering
#     if fc_t[15,1] < 0 or fc_t[15,2] < 0:
#         return
    
    p_e = p_tree[0::16]
    fc_e = fc_tree[0::16]
    coloring = 'seismic'

    sb.heatmap(data=fc_tree[:,0:1],xticklabels=edge_names[0:1],yticklabels=16,ax=ax[0,0],vmax=2,vmin=-2,center=0,linewidths=0,linecolor='w',cmap=coloring)
    sb.heatmap(data=p_tree[:,0:1],xticklabels=edge_names[0:1],yticklabels=16,ax=ax[1,0],center=0,linewidths=0,linecolor='w',cmap=coloring)

    sb.heatmap(data=fc_t[:,1:3],xticklabels=edge_names[1:3],yticklabels=1,ax=ax[0,2],vmax=2,vmin=-2,center=0,linewidths=0,linecolor='w',cmap=coloring)
    sb.heatmap(data=p_t[:,1:3],xticklabels=edge_names[1:3],yticklabels=1,ax=ax[1,2],center=0,linewidths=0,linecolor='w',cmap=coloring)

    sb.heatmap(data=fc_e[:,3:6],xticklabels=edge_names[3:6],yticklabels=1,ax=ax[0,1],vmax=2,vmin=-2,center=0,linewidths=0,linecolor='w',cmap=coloring)
    sb.heatmap(data=p_e[:,3:6],xticklabels=edge_names[3:6],yticklabels=1,ax=ax[1,1],center=0,linewidths=0,linecolor='w',cmap=coloring)
    plt.show()

def ep_bar_plotter():
    ep_p = p_tree[::16,3]
    ep_f = fc_tree[::16,3]
    strong_fc = [ep_f[elem] for elem in range(len(ep_f)) if ep_p[elem] > thresh]
    strong_fc.sort(reverse=True)
    
#     if len(strong_fc) < 1 or np.mean(strong_fc) >-0.1:
#         return

    er_fc = []
    for elem in strong_fc:
        this_ind = np.where(ep_f == elem)[0][0]
        if p_tree[this_ind*16,4] > thresh:
            this_er = fc_tree[this_ind*16,4]
            er_fc.append(this_er)
        else:
            er_fc.append(0)

    if np.min(er_fc) >0:
        return
    
    fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(15,5))
    axs.set_title(gene_name+' Promoter Accessibility',fontsize=25)
    axs.set_ylabel('Foldchange (log2)',fontsize=15)
    axs.set_xlabel('High-confidence Enhancers',fontsize=15)
    sb.barplot(x=list(range(len(strong_fc))),y=strong_fc,ax=axs)
    plt.show()
            
#     er_p = p_tree[::16,4]
#     er_f = fc_tree[::16,4]
#     strong_fc = [er_f[elem] for elem in range(len(er_f)) if er_p[elem] < thresh]
#     strong_fc.sort(reverse=True)
#     sb.barplot(x=list(range(len(er_fc))),y=er_fc)


# In[47]:


#v0 interesting genes = [Ryr2,Hcn1,Hcn4,Des,Tpm1,Actc1,Lmna,Pkp2,Dsp,Dsc2,Tmem43,Scn5a,Apob,Fbn1,Tgfbr1,Tgfbr2,Smad3,Mylk,Braf,Notch1,Ptpn11,Sox1,Tbx1,Jag1,Kras,Map2k1,Kcnh2,Pdlim3]
#notes:
#'Zfpm1'
#Tnnt2,Dsp,Hcn4
#define gene and perturbation split
# gene_name,t_split = list(set(gl))[x],16
gene_name,t_split = 'Xkr4',16
p_tree,fc_tree = tree_assembler(gene_name,t_split)
p_tree = -np.log10(p_tree+10**-15)
plot_getter()

#for plotting enhancer FCs
e_list = [len(tss) for tss in ep]
e_sum = sum(e_list)
thresh = -np.log10(0.05/e_sum+10**-15)

# ep_bar_plotter()


# In[48]:


ce_rna.var_names


# In[49]:


te[0][0]


# In[50]:


fc_tree[0:16,0]


# In[27]:


flat = np.array([np.log2(edge[2]+1e-2) for tss in er for edge in tss])
sigflat = np.array([np.log2(edge[2]+1e-2) for tss in er for edge in tss if edge[0] < 0.05/len(flat)])
negsig = [edge for edge in sigflat if edge < 0]


# In[45]:


# flat = np.array([np.log2(edge[2]+1e-2) for tss in er for edge in tss])
-np.log10(0.05/len(flat))


# In[34]:


flat = np.array([np.log2(edge[2]+1e-2) for tss in pr for edge in tss])
sigflat = np.array([np.log2(edge[2]+1e-2) for tss in pr for edge in tss if edge[0] < 0.05/len(flat)])
negsig = [edge for edge in sigflat if edge < 0]


# In[ ]:


gl = [ce_rna.var_names[entry[1]] for entry in promol]
for gene_name in list(set(gl)):
    t_split = 16
    p_tree,q_tree,fc_tree = tree_assembler(gene_name,t_split)
    
    #if gene has no enhancers, skips to next gene
    if len(p_tree) == 0:
        continue
        
    p_tree = -np.log10(p_tree+10**-15)
    q_tree = -np.log10(q_tree+10**-2)
    plot_getter()

    #for plotting enhancer FCs
    e_list = [len(tss) for tss in ep]
    e_sum = sum(e_list)
    thresh = -np.log10(0.05/e_sum+10**-15)
#     ep_bar_plotter()


# In[ ]:


# gl = ['Dsp','Ablim1','Tnnt2','Ppp2r2c','Pdlim4','Schip1','Ube2h','Fhod3','Ctsd','Limch1']
perts = ['','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt']
gl = [ce_rna.var_names[entry[1]] for entry in promol]
# gl = ['Lingo1']
# gl = fgl
gene_hits = []
total,count1,count2,count3 = 0,0,0,0
print(len(gl))
start = default_timer()
p_thresh,atac_fc_thresh,rna_fc_thresh = -np.log10(0.05+10e-15),np.log2(1.5+2e-5),np.log2(1.1+2e-5)
for x in range(len(set(gl))):
    #define gene and perturbation split
    gene_name,t_split = list(set(gl))[x],16
    p_tree,fc_tree = tree_assembler(gene_name,t_split)
    if type(p_tree)!=type(np.array([])):
        continue
    if x%1000==0:
        print(x,'out of ',len(set(gl)))
    p_tree = -np.log10(p_tree+10**-15)
#     q_tree = -np.log10(q_tree+10**-2)

    #filtering for open+recruit synergy
    #filter 1, EUP->R
    num_perts = 16
    num_enhancers = int(len(p_tree)/num_perts)
    
    EUP_pvals = np.array([p_tree[0,5]]+[enh for enh in p_tree[::num_perts,4]])
    EUP_fcs = np.array([fc_tree[0,5]]+[enh for enh in fc_tree[::num_perts,4]])
    
    total+=len(EUP_pvals)
    for reg_elem in range(len(EUP_pvals)):
        if EUP_pvals[reg_elem]>=p_thresh and EUP_fcs[reg_elem]>=atac_fc_thresh:
            count1+=1
            #filter 2, T{a,b}->EUP and T{a,b}->R
            if reg_elem == 0: #if regulatory element is a promoter
                reg_elem_indices = (np.array([0]),np.array([5]))
                reg_col,rna_col = 1,2
                TFsuperset_regelem_pvals = p_tree[num_enhancers::num_enhancers,reg_col]
                TFsuperset_regelem_fcs = fc_tree[num_enhancers::num_enhancers,reg_col]
                TFsuperset_rna_pvals = p_tree[num_enhancers::num_enhancers,rna_col]
                TFsuperset_rna_fcs = fc_tree[num_enhancers::num_enhancers,rna_col]
            else: #if regulatory element is an enhancer
                reg_elem_indices = (np.array(list(range((reg_elem-1)*num_perts,reg_elem*num_perts))),np.array([4]*num_perts))
                reg_col,rna_col = 0,2
                TFsuperset_regelem_pvals = p_tree[reg_elem_indices[0][1:],reg_col]
                TFsuperset_regelem_fcs = fc_tree[reg_elem_indices[0][1:],reg_col]
                TFsuperset_rna_pvals = p_tree[reg_elem_indices[0][1:],rna_col]
                TFsuperset_rna_fcs = fc_tree[reg_elem_indices[0][1:],rna_col]

            for TFsuperset in range(len(TFsuperset_regelem_pvals)):
                regelem_is_strong = (TFsuperset_regelem_pvals[TFsuperset] >= p_thresh) and (
                                     TFsuperset_regelem_fcs[TFsuperset] >= atac_fc_thresh)
                rna_is_strong = (TFsuperset_rna_pvals[TFsuperset] >= p_thresh) and (
                                 TFsuperset_rna_fcs[TFsuperset] >= rna_fc_thresh)
                if regelem_is_strong and rna_is_strong:
                    count2+=1
                    #filter 3, T{a or b}->EUP and not T{a or b}->R
                    TFsuperset_regelem_indices = np.where(p_tree == TFsuperset_regelem_pvals[TFsuperset])
                    if reg_elem == 0: #if regulatory element is a promoter
                        reg_col,rna_col = 1,2
                        TFsuperset_regelem_indices = (np.array(list(range(num_enhancers,len(p_tree),num_enhancers))),np.array([reg_col]))
                        TFsub_regelem_pvals = p_tree[:num_enhancers*TFsuperset+1:num_enhancers,reg_col]
                        TFsub_regelem_fcs = fc_tree[:num_enhancers*TFsuperset+1:num_enhancers,reg_col]
                        TFsub_rna_pvals = p_tree[:num_enhancers*TFsuperset+1:num_enhancers,rna_col]
                        TFsub_rna_fcs = fc_tree[:num_enhancers*TFsuperset+1:num_enhancers,rna_col]
                    else: #if regulatory element is an enhancer
                        reg_col,rna_col = 0,2
                        TFsuperset_regelem_indices = (np.array(list(range(reg_elem*num_perts+1,(reg_elem+1)*num_perts+1))),np.array([reg_col]))
                        TFsub_regelem_pvals = p_tree[reg_elem_indices[0][:TFsuperset+1],reg_col]
                        TFsub_regelem_fcs = fc_tree[reg_elem_indices[0][:TFsuperset+1],reg_col]
                        TFsub_rna_pvals = p_tree[reg_elem_indices[0][:TFsuperset+1],rna_col]
                        TFsub_rna_fcs = fc_tree[reg_elem_indices[0][:TFsuperset+1],rna_col]
                    superset_pert = perts[TFsuperset]
                    
#_________________________________________________________________________________________________________________________________________
                    #if creating super=sub of all constitutive singles (plus NC if super is already single)
                    subset_perts = [perts.index(single) for single in list(superset_pert)]
                    if len(subset_perts)<2:
                        subset_perts.insert(0,0)
                    at_least_one_sub_to_regelem = np.sum([(TFsub_regelem_pvals[single]>p_thresh and TFsub_regelem_fcs[single]>(
                                                    atac_fc_thresh/len(subset_perts))) for single in subset_perts])
                    no_subs_to_rna = np.prod([(TFsub_rna_pvals[single]>p_thresh and TFsub_rna_fcs[single]<=TFsuperset_rna_fcs[TFsuperset]
                                               ) for single in subset_perts]) and (np.sum(
                                                [TFsub_rna_fcs[single] for single in subset_perts])<=TFsuperset_rna_fcs[TFsuperset])
                    
                    if at_least_one_sub_to_regelem and no_subs_to_rna:
                        gene_hits.append(gene_name)
                        count3+=1
#                         print('*************************We got one!*************************')
#                         print('Gene: ',gene_name,
#                               '\nRegulatory Element Indices: ',reg_elem_indices,
#                               '\nPerturbation superset and subsets:',superset_pert,subset_perts)
                    
    
#_________________________________________________________________________________________________________________________________________
#                     #if creating any possible super=sub1+sub2
#                     subset_perts = [(perts.index(combo[0]),perts.index(combo[1])) for combo in combinations(perts,2)
#                                     if ''.join(sorted(combo[0] + combo[1])) == superset_pert]
#                     for subset in subset_perts:
#                         suba,subb = subset
#                         at_least_one_sub_to_regelem = (TFsub_regelem_pvals[suba]>p_thresh and TFsub_regelem_fcs[suba]>atac_fc_thresh) or (
#                                                        TFsub_regelem_pvals[subb]>p_thresh and TFsub_regelem_fcs[subb]>atac_fc_thresh)

#                         #set p thresh to 0 here for now, since the hypergeom. isn't as applicable
#                         no_subs_to_rna = (TFsub_rna_pvals[suba]>0 and TFsub_rna_fcs[suba]<=TFsuperset_rna_fcs[TFsuperset]) and (
#                                           TFsub_rna_pvals[subb]>0 and TFsub_rna_fcs[subb]<=TFsuperset_rna_fcs[TFsuperset]) and (
#                                           TFsub_rna_fcs[suba]+TFsub_rna_fcs[subb]<=TFsuperset_rna_fcs[TFsuperset])
#                         nc_must_open = True
#                         if suba == 0 or subb == 0:
#                             nc_must_open = (TFsub_regelem_pvals[0]>p_thresh and TFsub_regelem_fcs[0]>0)
# #                             #need to add cpm-based cut-off in NC cells here, since fc will always be low for nc
# #                             nc_must_open = (TFsub_regelem_pvals[0]>p_thresh and TFsub_regelem_fcs[0]>XXXXXXXXX)

#                         if at_least_one_sub_to_regelem and no_subs_to_rna and nc_must_open:
#                             gene_hits.append(gene_name)
#                             count3+=1
#                             print('*************************We got one!*************************')
#                             print('Gene: ',gene_name,
#                                   '\nRegulatory Element Indices: ',reg_elem_indices,
#                                   '\nPerturbation superset and subsets:',superset_pert,perts[suba],perts[subb])
#_________________________________________________________________________________________________________________________________________


print(default_timer() - start)
with open('open_recruit_'+lib+'_lowp_single_super.pkl','wb') as file:
    pkl.dump(gene_hits,file)
print('regulatory elements checked:',total,
      '\ngood EUP->Rs:',count1,
      '\ngood supersets for EUP->Rs:',count2,
      '\ngood subsets for supersets',count3)


# In[5]:


# fgl = pkl.load(open('open_recruit_ln_lowp_single_super.pkl','rb'))
fgl = pkl.load(open('open_recruit_co_single_super.pkl','rb'))


# In[6]:


len(fgl)


# In[28]:


'Ryr2' in set(fgl)


# In[6]:


fgl = pkl.load(open('co_single_super_EUPers.pkl','rb'))


# In[31]:


#for grouping cell features as IEAS
# gl = ['Dsp','Ablim1','Tnnt2','Ppp2r2c','Pdlim4','Schip1','Ube2h','Fhod3','Ctsd','Limch1']
perts = ['','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt']
singlepert_ids = list(range(1,5))
superpert_ids = list(range(5,len(perts)))
gl = [ce_rna.var_names[entry[1]] for entry in promol]
# gl = ['Lingo1']
# gl = fgl
IEAS_elist,IEAS_plist,IEAS_rlist = [],[],[]
total,count1,count2,count3 = 0,0,0,0
print(len(gl))
start = default_timer()
p_thresh,atac_fc_thresh,rna_fc_thresh = -np.log10(0.05/len(gl)+10e-15),np.log2(1.5+2e-5),np.log2(1.1+2e-5)
indep_cut = 0.5 #(percent foldchange accepted for 'independent' features)/100
epi_cut = 0.1 #(percent foldchange difference accepted for 'epistatic' features)/100
adi_cut = 0.1 #(percent foldchange difference accepted for 'additive' features)/100
for x in range(len(set(gl))):
    #define gene and perturbation split
    gene_name,t_split = list(set(gl))[x],16
    p_tree,fc_tree = tree_assembler(gene_name,t_split)
    if type(p_tree)!=type(np.array([])):
        continue
    if x%1000==0:
        print(x,'out of ',len(set(gl)))
    p_tree = -np.log10(p_tree+10**-15)

    #define subsets of each tree to group
    num_perts = len(perts)
    num_enhancers = int(len(p_tree)/num_perts)
    TE_pvals,TP_pvals,TR_pvals = p_tree[:,0],p_tree[::num_enhancers,1],p_tree[::num_enhancers,2]
    TE_fcs,TP_fcs,TR_fcs = fc_tree[:,0],fc_tree[::num_enhancers,1],fc_tree[::num_enhancers,2]
    
    #scan TsupE
    for this_te in range(len(TE_pvals)):
        if this_te%num_perts in superpert_ids: #only bother with tests if a superset is being looked at
            highfc = TE_fcs[this_te] > atac_fc_thresh #optional for highfc
            if TE_pvals[this_te] > p_thresh and highfc:
                this_pert = perts[this_te%num_perts]
                constituent_pert_ids = [perts.index(constituent) for constituent in this_pert]
                const_look_good = True
#                 for which_pert in constituent_pert_ids:
#                     this_const = this_te - (this_te%num_perts - which_pert) #define sub index by subtracting from super's index
#                     if not (TE_pvals[this_const]>p_thresh or np.log2(1-indep_cut+2e-5)<TE_fcs[this_const]<np.log2(1+indep_cut+2e-5)):
#                         const_look_good = False
                if const_look_good: #only bother with tests if all subsets look good
                    super_fc = TE_fcs[this_te]
                    sub_fcs = [TE_fcs[which_pert] for which_pert in constituent_pert_ids]
                    if (super_fc+np.log2(1+adi_cut+2e-5))<np.sum(sub_fcs): #subsets sum (log) to above additive cut percent of superset
                        which_IAS = 'I'
                    elif (super_fc+np.log2(1-adi_cut+2e-5))>np.sum(sub_fcs): #subsets sum (log) to below additive cut percent of superset
                        which_IAS = 'S'
                    else: #subsets sum (log) to within additive cut percent of superset
                        which_IAS = 'A'

                    for which_pert in constituent_pert_ids:
                        this_const = this_te - (this_te%num_perts - which_pert) #define sub index by subtracting from super's index
                        is_epistatic = (super_fc+np.log2(1-epi_cut+2e-5))<TE_fcs[this_const]<(super_fc+np.log2(1+epi_cut+2e-5))

                        IEAS_str = 'E'*is_epistatic+which_IAS
                        IEAS_elist.append([this_pert,perts[which_pert],IEAS_str,gene_name,(this_te/16)])
    #scan TsupP
    for this_tp in range(len(TP_pvals)):
        if this_tp%num_perts in superpert_ids: #only bother with tests if a superset is being looked at
            highfc = TP_fcs[this_tp] > atac_fc_thresh #optional for highfc
            if TP_pvals[this_tp] > p_thresh and highfc:
                this_pert = perts[this_tp%num_perts]
                constituent_pert_ids = [perts.index(constituent) for constituent in this_pert]
                const_look_good = True
#                 for which_pert in constituent_pert_ids:
#                     this_const = this_tp - (this_tp%num_perts - which_pert) #define sub index by subtracting from super's index
#                     if not (TP_pvals[this_const]>p_thresh or np.log2(1-indep_cut+2e-5)<TP_fcs[this_const]<np.log2(1+indep_cut+2e-5)):
#                         const_look_good = False
                if const_look_good: #only bother with tests if all subsets look good
                    super_fc = TP_fcs[this_tp]
                    sub_fcs = [TP_fcs[which_pert] for which_pert in constituent_pert_ids]
                    if (super_fc+np.log2(1+adi_cut+2e-5))<np.sum(sub_fcs): #subsets sum (log) to above additive cut percent of superset
                        which_IAS = 'I'
                    elif (super_fc+np.log2(1-adi_cut+2e-5))>np.sum(sub_fcs): #subsets sum (log) to below additive cut percent of superset
                        which_IAS = 'S'
                    else: #subsets sum (log) to within additive cut percent of superset
                        which_IAS = 'A'

                    for which_pert in constituent_pert_ids:
                        this_const = this_tp - (this_tp%num_perts - which_pert) #define sub index by subtracting from super's index
                        is_epistatic = (super_fc+np.log2(1-epi_cut+2e-5))<TP_fcs[this_const]<(super_fc+np.log2(1+epi_cut+2e-5))

                        IEAS_str = 'E'*is_epistatic+which_IAS
                        IEAS_plist.append([this_pert,perts[which_pert],IEAS_str,gene_name,(this_tp/16)])
    #scan TsupR
    for this_tr in range(len(TR_pvals)):
        if this_tr%num_perts in superpert_ids: #only bother with tests if a superset is being looked at
            highfc = TR_fcs[this_tr] > rna_fc_thresh #optional for highfc
            if TR_pvals[this_tr] > p_thresh and highfc:
                this_pert = perts[this_tr%num_perts]
                constituent_pert_ids = [perts.index(constituent) for constituent in this_pert]
                const_look_good = True
#                 for which_pert in constituent_pert_ids:
#                     this_const = this_tr - (this_tr%num_perts - which_pert) #define sub index by subtracting from super's index
#                     if not (TR_pvals[this_const]>p_thresh or np.log2(1-indep_cut+2e-5)<TR_fcs[this_const]<np.log2(1+indep_cut+2e-5)):
#                         const_look_good = False
                if const_look_good: #only bother with tests if all subsets look good
                    super_fc = TR_fcs[this_tr]
                    sub_fcs = [TR_fcs[which_pert] for which_pert in constituent_pert_ids]
                    if (super_fc+np.log2(1+adi_cut+2e-5))<np.sum(sub_fcs): #subsets sum (log) to above additive cut percent of superset
                        which_IAS = 'I'
                    elif (super_fc+np.log2(1-adi_cut+2e-5))>np.sum(sub_fcs): #subsets sum (log) to below additive cut percent of superset
                        which_IAS = 'S'
                    else: #subsets sum (log) to within additive cut percent of superset
                        which_IAS = 'A'

                    for which_pert in constituent_pert_ids:
                        this_const = this_tr - (this_tr%num_perts - which_pert) #define sub index by subtracting from super's index
                        is_epistatic = (super_fc+np.log2(1-epi_cut+2e-5))<TR_fcs[this_const]<(super_fc+np.log2(1+epi_cut+2e-5))

                        IEAS_str = 'E'*is_epistatic+which_IAS
                        IEAS_rlist.append([this_pert,perts[which_pert],IEAS_str,gene_name,(this_tr/16)])


                    
print(default_timer() - start)
with open('IEAS_'+lib+'_enhs_noindep.pkl','wb') as file:
    pkl.dump(IEAS_elist,file)
with open('IEAS_'+lib+'_proms_noindep.pkl','wb') as file:
    pkl.dump(IEAS_plist,file)
with open('IEAS_'+lib+'_rnas_noindep.pkl','wb') as file:
    pkl.dump(IEAS_rlist,file)


# In[33]:


# fel = pkl.load(open('co_IEAS_enhs.pkl','rb'))
# fpl = pkl.load(open('co_IEAS_proms.pkl','rb'))
# frl = pkl.load(open('co_IEAS_rnas.pkl','rb'))
# fel = pkl.load(open('co_IEAS_enhs_lowp.pkl','rb'))
# fpl = pkl.load(open('co_IEAS_proms_lowp.pkl','rb'))
# frl = pkl.load(open('co_IEAS_rnas_lowp.pkl','rb'))
# fel = pkl.load(open('co_IEAS_enhs_highfc.pkl','rb'))
# fpl = pkl.load(open('co_IEAS_proms_highfc.pkl','rb'))
# frl = pkl.load(open('co_IEAS_rnas_highfc.pkl','rb'))
# fel = pkl.load(open('co_IEAS_enhs_highfc_lowp.pkl','rb'))
# fpl = pkl.load(open('co_IEAS_proms_highfc_lowp.pkl','rb'))
# frl = pkl.load(open('co_IEAS_rnas_highfc_lowp.pkl','rb'))
# fel = pkl.load(open('co_IEAS_enhs_highfc_lowp_noindep.pkl','rb'))
# fpl = pkl.load(open('co_IEAS_proms_highfc_lowp_noindep.pkl','rb'))
# frl = pkl.load(open('co_IEAS_rnas_highfc_lowp_noindep.pkl','rb'))
fel = pkl.load(open('IEAS_ln_enhs_noindep.pkl','rb'))
fpl = pkl.load(open('IEAS_ln_proms_noindep.pkl','rb'))
frl = pkl.load(open('IEAS_ln_rnas_noindep.pkl','rb'))


# In[34]:


len(fel),len(fpl),len(frl)


# In[52]:


frl[0]


# In[53]:


fwl = frl
desig = 'I'
ge = [row for row in fwl if row[1] == 'g' and desig in row[2] and row[0] == 'ghmt']
he = [row for row in fwl if row[1] == 'h' and desig in row[2] and row[0] == 'ghmt']
me = [row for row in fwl if row[1] == 'm' and desig in row[2] and row[0] == 'ghmt']
te = [row for row in fwl if row[1] == 't' and desig in row[2] and row[0] == 'ghmt']
print(len(ge),len(he),len(me),len(te))
desig = 'A'
ge = [row for row in fwl if row[1] == 'g' and desig in row[2] and row[0] == 'ghmt']
he = [row for row in fwl if row[1] == 'h' and desig in row[2] and row[0] == 'ghmt']
me = [row for row in fwl if row[1] == 'm' and desig in row[2] and row[0] == 'ghmt']
te = [row for row in fwl if row[1] == 't' and desig in row[2] and row[0] == 'ghmt']
print(len(ge),len(he),len(me),len(te))
desig = 'S'
ge = [row for row in fwl if row[1] == 'g' and desig in row[2] and row[0] == 'ghmt']
he = [row for row in fwl if row[1] == 'h' and desig in row[2] and row[0] == 'ghmt']
me = [row for row in fwl if row[1] == 'm' and desig in row[2] and row[0] == 'ghmt']
te = [row for row in fwl if row[1] == 't' and desig in row[2] and row[0] == 'ghmt']
print(len(ge),len(he),len(me),len(te))
desig = 'E'
ge = [row for row in fwl if row[1] == 'g' and desig in row[2] and row[0] == 'ghmt']
he = [row for row in fwl if row[1] == 'h' and desig in row[2] and row[0] == 'ghmt']
me = [row for row in fwl if row[1] == 'm' and desig in row[2] and row[0] == 'ghmt']
te = [row for row in fwl if row[1] == 't' and desig in row[2] and row[0] == 'ghmt']
print(len(ge),len(he),len(me),len(te))

gtot = [row for row in fwl if row[1] == 'g' and row[0] == 'ghmt']
htot = [row for row in fwl if row[1] == 'h' and row[0] == 'ghmt']
mtot = [row for row in fwl if row[1] == 'm' and row[0] == 'ghmt']
ttot = [row for row in fwl if row[1] == 't' and row[0] == 'ghmt']
print(len(gtot),len(htot),len(mtot),len(ttot))


# In[51]:


perts = ['','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt']
rl = [[]]*16
fwl = frl
rel = 'I'
for combo_id in range(len(perts)):
    rl[combo_id] = [row for row in fwl if row[0] == perts[combo_id]]
portion = np.array([len([row for row in rl_pert if rel in row[2]])/len(rl_pert) for rl_pert in rl if len(rl_pert) != 0])
full_lists = np.array([[row for row in rl_pert if rel in row[2]] for rl_pert in rl if len(rl_pert) != 0])
for row_id in range(len(rl)):
    if rl[row_id] == []:
        portion = np.insert(portion,row_id,0)
sb.barplot(orient='v',x=perts,y=portion*100)


# In[21]:


g=[]
for pert in full_lists:
    for row in pert:
        g.append(row[3])
g = set(g)
full_str = ''
for elem in g:
    full_str+=elem+','
full_str


# In[22]:


ghmt_g = set([row[3] for row in full_lists[10]])
list(ghmt_g)
full_str = ''
for elem in ghmt_g:
    full_str+=elem+'\n'
print(full_str)


# In[9]:


trows = [row for row in frl if 'S' in row[2]]
x = [len([row for row in trows if row[0] == perts[pert_num]])/len(perts[pert_num]) for pert_num in range(1,len(perts))]
x


# In[ ]:


trows = [row for row in frl if 'S' in row[2] and row[0] == 'gmt']
gl = set([row[3] for row in trows])
gl


# In[52]:


tel = [row[3] for row in fel if 'S' in row[2]]
'Myh6' in set(tel)


# In[37]:


tel = [row[3] for row in frl if 'S' in row[2]]
tel = [row for row in frl if row[3] == 'Cacna1c']
# set(tel)
tel


# In[124]:


16*6.9375


# In[41]:


reg_elem_indices = np.where(p_tree == 15)
reg_elem_indices[0][0:1]


# In[21]:


x=8
np.log2(1.1),np.log2(x),np.log2(0.9)


# In[82]:


t=np.log2(1.5+2**-5)
x,y=1.2,1.45
t=np.log2(y+2**-5)
print(np.log2(x+2**-5)*2<t,x**2<y)


# In[16]:


perts = ['','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt']
superset_pert = perts[15]
possible_subset_perts = [(perts.index(combo[0]),perts.index(combo[1])) for combo in combinations(perts,2)
                         if ''.join(sorted(combo[0] + combo[1])) == superset_pert]
possible_subset_perts
suba,subb = possible_subset_perts[1]
print(suba,subb)


# In[99]:


type(np.array([]))


# In[24]:


p_tree[0,1]


# In[21]:


p_tree[::21,1]


# In[52]:


print(count,total)


# In[66]:


EUP_pvals = np.array([p_tree[0,5]]+[enh for enh in p_tree[::16,4]])
EUP_pvals[0]


# In[71]:


np.shape(p_tree)


# In[100]:


[x,y] = np.where(p_tree == EUP_pvals[0])
p_tree[21::21,1]


# In[92]:


x


# In[46]:


p_tree[0,5]


# In[47]:


p_tree[::16,4]


# In[18]:


print(p_tree[0,5],fc_tree[0,5])


# In[43]:


gene_name = 'Pkp1'
tss_pos = np.where(ce_tss['gid'] == gene_name)[0]
prom = [elem[2] for elem in promol if elem[0] in tss_pos][0]
ce_atac.var_names[prom]


# In[10]:


gene_name = 'Pkp1'
tss_pos = np.where(ce_tss['gid'] == gene_name)[0]
enhs = [elem[2] for elem in enhol if elem[0] in tss_pos][0]
print(ce_atac.var_names[enhs[10]]+'\n')
for enh in enhs[0:22]:
    print(ce_atac.var_names[enh])


# In[45]:


gene_name = 'Tnnt2'
tss_pos = np.where(ce_tss['gid'] == gene_name)[0]
prom = [elem[2] for elem in promol if elem[0] in tss_pos][0]
ce_atac.var_names[prom]


# In[12]:


gene_name = 'Tnnt2'
tss_pos = np.where(ce_tss['gid'] == gene_name)[0]
enhs = [elem[2] for elem in enhol if elem[0] in tss_pos][0]
print(ce_atac.var_names[enhs[16]]+'\n',ce_atac.var_names[enhs[21]]+'\n',ce_atac.var_names[enhs[27]]+'\n')
for enh in enhs[-22:]:
    print(ce_atac.var_names[enh])


# In[104]:


e=43

print(p_tree[:,3][::16][e])

print(2**fc_tree[:,3][::16][e])


# # SPLIT ADATAS BY PERT

# In[9]:


#create grouping of all NC cells
perturbed_cells = []
for grouping in ccc[1][1:]:
    perturbed_cells += grouping
nc_cells = [cell for cell in ce_rna.obs_names if cell not in perturbed_cells]

#initialize adata list for nc cells
cerna_list = [ce_rna[nc_cells].copy()]
ceatac_list = [ce_atac[nc_cells].copy()]

#append adatas for every other perturbation group
cerna_list += [ce_rna[grouping].copy() for grouping in ccc[1][1:]]
ceatac_list += [ce_atac[grouping].copy() for grouping in ccc[1][1:]]


# # MAIN

# ## FILTER PEAKS BY FC OR P, AND LOCATE ASSOCIATED GENE

# In[91]:


#MAKE ENH LISTS
e_fcon_l,enhp_l = [],[]
for tss_ind in range(len(enhpvals)):
    for peak_ind in range(len(enhpvals[tss_ind])):
        e_fcon_l.append(efcon[tss_ind][peak_ind])
        enhp_l.append(enhpvals[tss_ind][peak_ind])
for index in range(len(enhp_l)):
    if enhp_l[index] > 0.5:
        enhp_l[index] = 1 - enhp_l[index]

#MAKE PROM LISTS
p_fcon_l,promp_l = pfcon,prompvals
for index in range(len(promp_l)):
    if promp_l[index] > 0.5:
        promp_l[index] = 1 - promp_l[index]


# In[128]:


#LOOK AT PROMS
newere = [np.log10(1/(elem+1e-15)) for elem in promp_l]
newf = [np.log2([elem[0]+1e-2]) for elem in p_fcon_l]
data = np.column_stack([newf,newere])

pden = gaussian_kde(data.T,bw_method='silverman')
pdval=pden(data.T)
pdlist = list(pdval)
#pdlist = np.log2(pdlist)


# In[132]:


#LOOK AT ENHS
newere = [np.log10(1/(elem+1e-15)) for elem in enhp_l]
newf = [np.log2([elem+1e-2]) for elem in e_fcon_l]
data = np.column_stack([newf,newere])

eden = gaussian_kde(data.T)
edval = eden(data.T)
edlist = list(edval)


# In[18]:


prrna_stats = pkl.load(open('total_promoterrna_stats.pkl','rb'))


# In[36]:


p1 = pkl.load(open('total_pd_perturbationpromoter_stats1.pkl','rb'))
p2 = pkl.load(open('total_pd_perturbationpromoter_stats2.pkl','rb'))
p3 = pkl.load(open('total_pd_perturbationpromoter_stats3.pkl','rb'))
p4 = pkl.load(open('total_pd_perturbationpromoter_stats4.pkl','rb'))
p5 = pkl.load(open('total_pd_perturbationpromoter_stats5.pkl','rb'))
p6 = pkl.load(open('total_pd_perturbationpromoter_stats6.pkl','rb'))
# p11 = pkl.load(open('total_pd_perturbationenhancer_stats11.pkl','rb'))
# p12 = pkl.load(open('total_pd_perturbationenhancer_stats12.pkl','rb'))
# p13 = pkl.load(open('total_pd_perturbationenhancer_stats13.pkl','rb'))
# p14 = pkl.load(open('total_pd_perturbationenhancer_stats14.pkl','rb'))
# p15 = pkl.load(open('total_pd_perturbationenhancer_stats15.pkl','rb'))
# p16 = pkl.load(open('total_pd_perturbationenhancer_stats16.pkl','rb'))
# p17 = pkl.load(open('total_pd_perturbationenhancer_stats17.pkl','rb'))
# p18 = pkl.load(open('total_pd_perturbationenhancer_stats18.pkl','rb'))
# p19 = pkl.load(open('total_pd_perturbationenhancer_stats19.pkl','rb'))
# p20 = pkl.load(open('total_pd_perturbationenhancer_stats20.pkl','rb'))
# p21 = pkl.load(open('total_pd_perturbationenhancer_stats21.pkl','rb'))
# p22 = pkl.load(open('total_pd_perturbationenhancer_stats22.pkl','rb'))
# p23 = pkl.load(open('total_pd_perturbationenhancer_stats23.pkl','rb'))


# In[16]:


print(len(p1),len(p2),len(p3),len(p4),len(p5),len(p10),len(p11),len(p12),len(p13),len(p14),len(p15),len(p16),len(p17),len(p18),len(p19),len(p20),len(p21),len(p22),len(p23))


# In[37]:


pt = p1+p2+p3+p4+p5+p6     #+p11+p12+p13+p14+p15+p16+p17+p18+p19+p20+p21+p22+p23
len(pt)


# In[38]:


with open('total_pd_perturbationpromoter_stats.pkl','wb') as file:
    pkl.dump(pt,file)


# In[39]:


p22 = pkl.load(open('total_pd_perturbationenhancer_stats.pkl','rb'))
p23 = pkl.load(open('total_pd_perturbationpromoter_stats.pkl','rb'))


# ## 1. ATAC and RNA, P-Val and FC Generation, by PERT

# In[10]:


#want to make 3 data arrays, corresponding to proms, rnas, and enhs

#prom and rna arrays will match structure of promol but contain [pval, fcon] pairs per pert, per promol entry
#enh array will match structure of enhol but contain [pval, fcon] pairs per enhol entry

#promol is list of index triples [tss_ind,rna_ind,atac_ind] for all available tss
#enhol is list of index triples [tss_ind,rna_ind,[multiple atac_inds]] for all available tss


# In[10]:


cell_lists = [nc_cells]+ccc[1][1:]
inverse_cell_lists = []
for cell_list in cell_lists:
    inverse_list = [cell for cell in ce_rna.obs_names if cell not in cell_list]
    inverse_cell_lists.append(inverse_list)


# ## Generate Heat Map, filter for genes passing all thresholds

# In[46]:


# pertenh_stats = pkl.load(open('total_perturbationenhancer_stats.pkl','rb'))     #for perts fully split     
# pertprom_stats = pkl.load(open('total_perturbationpromoter_stats.pkl','rb'))    #for perts fully split
# pertrna_stats = pkl.load(open('total_perturbationrna_stats.pkl','rb'))          #for perts fully split
pertenh_stats = pkl.load(open('total_pd_perturbationenhancer_stats.pkl','rb'))     #for perts split by degree
pertprom_stats = pkl.load(open('total_pd_perturbationpromoter_stats.pkl','rb'))    #for perts split by degree
pertrna_stats = pkl.load(open('total_pd_perturbationrna_stats.pkl','rb'))          #for perts split by degree
enhprom_stats = pkl.load(open('total_enhancerpromoter_stats.pkl','rb'))
enhrna_stats = pkl.load(open('total_enhancerrna_stats.pkl','rb'))
promrna_stats = pkl.load(open('total_promoterrna_stats.pkl','rb'))

peen = pertenh_stats
pepr = pertprom_stats
pern = pertrna_stats
enpr = enhprom_stats
enrn = enhrna_stats
prrn = promrna_stats

print(len(peen),len(pepr),len(pern),len(enpr),len(enrn),len(prrn))

te = peen
tp = pepr
tr = pern
ep = enpr
er = enrn
pr = prrn


# ## PREPARE PERBOOL AND IDXBOOL

# In[47]:


#PERT TO ENH

#if using peen, need to perform an extra layer of list conversion
peenl = []
for tss in peen:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    peenl+=tssl
    peen = peenl

#specify which statset to look at
mat = peen
#convert to 2d array, where every row is a comparison, and columns are [pval,qval,fc]
matl = []
for tss in mat:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    matl+=tssl
mata = np.asarray(matl)

pbool = np.zeros(shape=(len(mata),1))
for row in range(len(mata)):
    if mata[row,1] < 1:
        pbool[row,0] = 1

print(len(mata),pbool.sum())


# In[48]:


#PERT TO PROM

#specify which statset to look at
mat = pepr
#convert to 2d array, where every row is a comparison, and columns are [pval,qval,fc]
matl = []
for tss in mat:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    matl+=tssl
mata = np.asarray(matl)

tpbool = np.zeros(shape=(len(mata),1))
for row in range(len(mata)):
    if mata[row,1] < 1:
        tpbool[row,0] = 1

print(len(mata),tpbool.sum())


# In[49]:


#PERT TO RNA

#specify which statset to look at
mat = pern
#convert to 2d array, where every row is a comparison, and columns are [pval,qval,fc]
matl = []
for tss in mat:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    matl+=tssl
mata = np.asarray(matl)

trbool = np.zeros(shape=(len(mata),1))
for row in range(len(mata)):
    if mata[row,1] < 1:
        trbool[row,0] = 1

print(len(mata),trbool.sum())


# In[50]:


#ENH TO PROM

#specify which statset to look at
mat = enpr
#convert to 2d array, where every row is a comparison, and columns are [pval,qval,fc]
matl = []
for tss in mat:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    matl+=tssl
mata = np.asarray(matl)

ebool = np.zeros(shape=(len(mata),1))
for row in range(len(mata)):
    if mata[row,1] < 1:
        ebool[row,0] = 1

print(len(mata),ebool.sum())


# In[51]:


#ENH TO RNA

#specify which statset to look at
mat = enrn
#convert to 2d array, where every row is a comparison, and columns are [pval,qval,fc]
matl = []
for tss in mat:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    matl+=tssl
mata = np.asarray(matl)

erbool = np.zeros(shape=(len(mata),1))
for row in range(len(mata)):
    if mata[row,1] < 1:
        erbool[row,0] = 1

print(len(mata),erbool.sum())


# In[52]:


#PROM TO RNA

#specify which statset to look at
mat = prrn
#convert to 2d array, where every row is a comparison, and columns are [pval,qval,fc]
matl = []
for tss in mat:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    matl+=tssl
mata = np.asarray(matl)

rbool = np.zeros(shape=(len(mata),1))
for row in range(len(mata)):
    if mata[row,1] < 1:
        rbool[row,0] = 1

print(len(mata),rbool.sum())


# In[53]:


pnum = np.zeros(shape=(len(ebool),1))
for pert in range(len(pbool)):
    if pbool[pert]:
        enh_num = int(pert/5) #change to 16 if doing full pert, 5 if doing pert degree
        pnum[enh_num] += 1


# In[54]:


nonz = [elem for elem in pnum if elem > 0]
len(nonz)


# In[55]:


elist = [len(elem[2]) for elem in enhol]
enum = np.zeros(shape=(len(rbool),1))
prom_num = 0
for enh in range(len(ebool)):
    if enh == np.sum(elist[0:prom_num+1]) or elist[prom_num] == 0:
        prom_num += 1
    if ebool[enh]:
        enum[prom_num] += 1


# In[56]:


nonz = [elem for elem in enum if elem > 0]
len(nonz)


# In[59]:


idxbool = np.zeros(shape=(len(pbool),3))
perbool = np.zeros(shape=(len(pbool),1))


# In[62]:


elist = [len(elem[2]) for elem in enhol]
r_num = 0
for per in range(len(pbool)):
    p_num = per
    e_num = int(per/5) #change to 16 if doing full pert
    if e_num == np.sum(elist[0:r_num+1]) or elist[r_num] == 0:
        r_num += 1
    
    idxbool[per,0] = p_num
    idxbool[per,1] = e_num
    idxbool[per,2] = r_num
    
    if pbool[p_num] and ebool[e_num] and rbool[r_num]:
        perbool[per] = 1


# In[62]:


with open('pd_perbool.pkl','wb') as pfile:
    pkl.dump(perbool,pfile)
with open('pd_idxbool.pkl','wb') as ifile:
    pkl.dump(idxbool,ifile)


# In[20]:


#make fake perbool for test for main-chains with exactly one link weak, expect >100,000 such chains
elist = [len(elem[2]) for elem in enhol]
r_num = 0
for per in range(len(pbool)):
    p_num = per
    e_num = int(per/5) #change to 16 if doing full pert, 5 if doing pert degree
    if e_num == np.sum(elist[0:r_num+1]) or elist[r_num] == 0:
        r_num += 1
    
    idxbool[per,0] = p_num
    idxbool[per,1] = e_num
    idxbool[per,2] = r_num
    
    if (pbool[p_num] + ebool[e_num] + rbool[r_num]) == 2:
        perbool[per] = 1


# In[20]:


with open('pd_twothirdsbool.pkl','wb') as pfile:
    pkl.dump(perbool,pfile)


# In[ ]:


#make expanded perbool for test for full networks with exactly one link weak
elist = [len(elem[2]) for elem in enhol]
r_num = 0
split = 5 #change to 16 if doing full pert or 5 if doing degree pert
for per in range(len(pbool)):
    p_num = per
    e_num = int(per/split)
    if e_num == np.sum(elist[0:r_num+1]) or elist[r_num] == 0:
        r_num += 1
    tp_num = r_num*split+per%5
    tr_num = r_num*split+per%5
    er_num = int(per/split)
    
    
    idxbool[per,0] = p_num
    idxbool[per,1] = e_num
    idxbool[per,2] = r_num
    
    missing_e_link = (pbool[p_num] + ebool[e_num] + erbool[r_num]) == 2
    full_side_chains = (tpbool[tp_num] and trbool[tr_num] and rbool[r_num])
    if missing_e_link and full_side_chains:
        perbool[per] = 1


# In[ ]:


print(len(perbool),sum(perbool))


# In[62]:


with open('pd_expandedbool.pkl','wb') as pfile:
    pkl.dump(perbool,pfile)


# In[57]:


idxbool = np.zeros(shape=(len(pbool),6))
perbool = np.zeros(shape=(len(pbool),1))


# In[64]:


idxbool.shape


# In[65]:


#make expanded perbool for test for full networks
elist = [len(elem[2]) for elem in enhol]
r_num = 0
split = 5 #change to 16 if doing full pert or 5 if doing degree pert
for per in range(len(pbool)):
    p_num = per
    e_num = int(per/split)
    if e_num == np.sum(elist[0:r_num+1]) or elist[r_num] == 0:
        r_num += 1
    tp_num = r_num*split+per%split
    tr_num = r_num*split+per%split
    er_num = int(per/split)
    
    idxbool[per,0] = p_num
    idxbool[per,1] = tp_num
    idxbool[per,2] = tr_num
    idxbool[per,3] = e_num
    idxbool[per,4] = er_num
    idxbool[per,5] = r_num
    
#     #delete below when done
    if r_num == 434:
        print(p_num%5,e_num-9931,pbool[p_num],tpbool[tp_num],trbool[tr_num],ebool[e_num],erbool[er_num],rbool[r_num])
#     #delete above when done
    
    full_e_link = (pbool[p_num] + ebool[e_num] + erbool[er_num]) == 3
    full_side_chains = (tpbool[tp_num] and trbool[tr_num] and rbool[r_num])
    if full_e_link and full_side_chains:
        perbool[per] = 1


# In[61]:


with open('pd_completebool.pkl','wb') as pfile:
    pkl.dump(perbool,pfile)
with open('pd_completebool_idx.pkl','wb') as pfile:
    pkl.dump(idxbool,pfile)


# ## LOOK AT SPECIFIC GRAPHS

# In[38]:


# pertenh_stats = pkl.load(open('total_perturbationenhancer_stats.pkl','rb'))     #for perts fully split     
# pertprom_stats = pkl.load(open('total_perturbationpromoter_stats.pkl','rb'))    #for perts fully split
# pertrna_stats = pkl.load(open('total_perturbationrna_stats.pkl','rb'))          #for perts fully split
pertenh_stats = pkl.load(open('total_pd_perturbationenhancer_stats.pkl','rb'))     #for perts split by degree
pertprom_stats = pkl.load(open('total_pd_perturbationpromoter_stats.pkl','rb'))    #for perts split by degree
pertrna_stats = pkl.load(open('total_pd_perturbationrna_stats.pkl','rb'))          #for perts split by degree
enhprom_stats = pkl.load(open('total_enhancerpromoter_stats.pkl','rb'))
enhrna_stats = pkl.load(open('total_enhancerrna_stats.pkl','rb'))
promrna_stats = pkl.load(open('total_promoterrna_stats.pkl','rb'))

peen = pertenh_stats
pepr = pertprom_stats
pern = pertrna_stats
enpr = enhprom_stats
enrn = enhrna_stats
prrn = promrna_stats

te = peen
tp = pepr
tr = pern
ep = enpr
er = enrn
pr = prrn


# In[22]:


pdbool = pkl.load(open('completebool.pkl','rb'))
pdbool_idx = pkl.load(open('completebool_idx.pkl','rb'))


# In[16]:


idx_counter = 0
full_graph_stats = []
for which_graph in range(len(pdbool)):
    if pdbool[which_graph]:
        ids = pdbool_idx[idx_counter]
        


# In[62]:


np.sum(pdbool)


# In[86]:


genesum = 0
for x in range(len(pdbool_idx)):
    if pdbool_idx[x][5] == 434:
        if pdbool[x]:
            genesum+=1
print(genesum)


# ## SET UP Pkp1 TABLE

# In[42]:


te = peen
tp = pepr
tr = pern
ep = enpr
er = enrn
pr = prrn


# In[47]:


#interesting genes = [Ryr2,Hcn1,Hcn4,Des,Tpm1,Actc1,Lmna,Pkp2,Dsp,Dsc2,Tmem43,Scn5a,Apob,Fbn1,Tgfbr1,Tgfbr2,Smad3,Mylk,Braf,Notch1,Ptpn11,Sox1,Tbx1,Jag1,Kras,Map2k1,Kcnh2,pdlim3]

#define gene and gather needed indices
gene_name = 'Pkp1'
tss_pos = list(ce_tss['gid']).index(gene_name)
gene_promol = [elem for elem in promol if elem[0] == tss_pos][0]
gene_enhol = [elem for elem in enhol if elem[0] == tss_pos][0]
promol_ind = promol.index(gene_promol)
print(promol_ind)
rna_ind = gene_promol[1]
prom_ind = gene_promol[2]
enh_inds = gene_enhol[2]
elist = [len(elem[2]) for elem in enhol]

#subset pqfc files
te_sub,tp_sub,tr_sub = te[promol_ind],np.array(tp[promol_ind]),np.array(tr[promol_ind])
te_sub = np.array([edge for enh in te_sub for edge in enh])     #flatten te_sub
ep_sub,er_sub,pr_sub = np.array(ep[promol_ind]),np.array(er[promol_ind]),np.array(pr[promol_ind])

#assemble into p, q, and fc matrices_________________________________________________________________________________________________________________
val = 0        #0=p, 1=q, 2=fc
t_split = 5     #number of perturbations or TF combos
e_split = len(enh_inds)

te_slice,tp_slice,tr_slice = te_sub[:,val],tp_sub[:,val],tr_sub[:,val]
ep_slice,er_slice,pr_slice = ep_sub[:,val],er_sub[:,val],pr_sub[:,val]
#extend all non-t by t_split
ep_slice = np.repeat(ep_slice,t_split)
er_slice = np.repeat(er_slice,t_split)
pr_slice = np.repeat(pr_slice,t_split)
#extend all non-e by e_split
tp_slice = np.repeat(tp_slice,e_split)
tr_slice = np.repeat(tr_slice,e_split)
pr_slice = np.repeat(pr_slice,e_split)
#concat all slices 
slices = np.array([te_slice,tp_slice,tr_slice,ep_slice,er_slice,pr_slice])
p_tree = np.transpose(slices)

#assemble into p, q, and fc matrices_________________________________________________________________________________________________________________
val = 1        #0=p, 1=q, 2=fc
t_split = 5     #number of perturbations or TF combos
e_split = len(enh_inds)

te_slice,tp_slice,tr_slice = te_sub[:,val],tp_sub[:,val],tr_sub[:,val]
ep_slice,er_slice,pr_slice = ep_sub[:,val],er_sub[:,val],pr_sub[:,val]
#extend all non-t by t_split
ep_slice = np.repeat(ep_slice,t_split)
er_slice = np.repeat(er_slice,t_split)
pr_slice = np.repeat(pr_slice,t_split)
#extend all non-e by e_split
tp_slice = np.repeat(tp_slice,e_split)
tr_slice = np.repeat(tr_slice,e_split)
pr_slice = np.repeat(pr_slice,e_split)
#concat all slices 
slices = np.array([te_slice,tp_slice,tr_slice,ep_slice,er_slice,pr_slice])
q_tree = np.transpose(slices)

#assemble into p, q, and fc matrices_________________________________________________________________________________________________________________
val = 2        #0=p, 1=q, 2=fc
t_split = 5     #number of perturbations or TF combos
e_split = len(enh_inds)

te_slice,tp_slice,tr_slice = te_sub[:,val],tp_sub[:,val],tr_sub[:,val]
ep_slice,er_slice,pr_slice = ep_sub[:,val],er_sub[:,val],pr_sub[:,val]
#extend all non-t by t_split
ep_slice = np.repeat(ep_slice,t_split)
er_slice = np.repeat(er_slice,t_split)
pr_slice = np.repeat(pr_slice,t_split)
#extend all non-e by e_split
tp_slice = np.repeat(tp_slice,e_split)
tr_slice = np.repeat(tr_slice,e_split)
pr_slice = np.repeat(pr_slice,e_split)
#concat all slices 
slices = np.array([te_slice,tp_slice,tr_slice,ep_slice,er_slice,pr_slice])
fc_tree = np.transpose(slices)
fc_tree = np.log2(fc_tree+2**-5)

#plot_______________________________________________________________________________________________________________________________________________
edge_names = ['te','tp','tr','ep','er','pr']
fig,ax = plt.subplots(figsize=(10,20),ncols=2)
#sb.heatmap(data=p_tree,xticklabels=edge_names,yticklabels=16,ax=ax[0])
sb.heatmap(data=q_tree,xticklabels=edge_names,yticklabels=5,ax=ax[0],vmax=1)
sb.heatmap(data=fc_tree,xticklabels=edge_names,yticklabels=5,ax=ax[1],vmax=5,center=0)


# ## PREPARE CONSTATS AND CONIDEN

# In[64]:


perbool = pkl.load(open('pd_perbool.pkl','rb'))
idxbool = pkl.load(open('pd_idxbool.pkl','rb'))
elist = [len(elem[2]) for elem in enhol]


# In[65]:


#linearize peen,enpr,and prrn, for indexing
peen = pertenh_stats
pepr = pertprom_stats
pern = pertrna_stats
enpr = enhprom_stats
enrn = enhrna_stats
prrn = promrna_stats

print(len(peen),len(pepr),len(pern),len(enpr),len(enrn),len(prrn))

#pertenh linearizer 1
peenl = []
for tss in peen:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    peenl+=tssl
peen = peenl
#pertenh linearizer 2
peenl = []
for tss in peen:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    peenl+=tssl
peen = peenl

#enhprom linearizer 1
enprl = []
for tss in enpr:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    enprl+=tssl
enpr = enprl

#promrna linearizer 1
prrnl = []
for tss in prrn:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    prrnl+=tssl
prrn = prrnl

print(len(peen),len(enpr),len(prrn))


# In[66]:


#create constats array where the first axis spans all hits in perbool, 
#   the second axis spans each of the three connections tested (p->e,e->p,p->r), and the third axis spans [pv,qv,fc]
#create coniden list-list where the first axis spans all hits in perbool, 
#   and the second axis is the corresponding [perturbation idx, enh ATAC idx, prom ATAC idx, RNA idx]
constats = np.ndarray((int(np.sum(perbool)),3,3))
coniden = []
con_idx = 0
how_many_perts = len(pertenh_stats[0])
for elem in range(len(perbool)):
    if perbool[elem] > 0:
        #fill in constats values
        p_idx,e_idx,r_idx = int(idxbool[elem,0]),int(idxbool[elem,1]),int(idxbool[elem,2])
        constats[con_idx,0,0],constats[con_idx,0,1],constats[con_idx,0,2] = peen[p_idx][0],peen[p_idx][1],peen[p_idx][2]
        constats[con_idx,1,0],constats[con_idx,1,1],constats[con_idx,1,2] = enpr[e_idx][0],enpr[e_idx][1],enpr[e_idx][2]
        constats[con_idx,2,0],constats[con_idx,2,1],constats[con_idx,2,2] = prrn[r_idx][0],prrn[r_idx][1],prrn[r_idx][2]
        con_idx += 1
        
        #fill in coniden values
        pert_libidx = int(idxbool[elem,0] % how_many_perts)
        enh_atac = int(e_idx - np.sum(elist[0:r_idx]))
        enh_libidx = enhol[r_idx][2][enh_atac]
        prom_libidx = promol[r_idx][2]
        rna_libidx = promol[r_idx][1]
        iden = [pert_libidx,enh_libidx,prom_libidx,rna_libidx]
        coniden.append(iden)


# In[67]:


with open('pd_constats.pkl','wb') as sfile:
    pkl.dump(constats,sfile)
with open('pd_coniden.pkl','wb') as ifile:
    pkl.dump(coniden,ifile)


# In[7]:


constats = pkl.load(open('constats.pkl','rb'))
coniden = pkl.load(open('coniden.pkl','rb'))


# In[9]:


constats.shape


# In[7]:


genes = []
for hit in coniden:
    genes.append(ce_rna.var_names[hit[3]])
gs = set(genes)


# In[10]:


gene = 'Pkp1'
for row in range(len(coniden)):
    if coniden[row][3] == list(ce_rna.var_names).index(gene):
        print(coniden[row])
        print(constats[row])


# In[13]:


ce_rna.var_names[562]


# In[75]:


#cardiac genes = [Ryr2,Hcn1,Hcn4,Des,Tpm1,Actc1,Lmna,Pkp2,Dsp,Dsc2,Tmem43,Scn5a,Apob,Fbn1,Tgfbr1,Tgfbr2,Smad3,Mylk,Braf,Notch1,Ptpn11,Sox1,Tbx1,Jag1,Kras,Map2k1,Kcnh2,pdlim3]
#+[Myl7]
gene = ''
gene in gs


# In[17]:


#search coniden for genes
genes=[]
for row in range(len(constats)):
    genes.append(ce_rna.var_names[coniden[row][3]])
print('Pkp1' in genes)


# In[30]:


pkp_rows = []
for row in range(len(coniden)):
    if coniden[row][3] == 562:     #562 is the index of Pkp1 in ce_rna
        pkp_rows.append(row)
print(pkp_rows)


# In[22]:


print(constats[2795],coniden[2795])


# In[32]:


for row in pkp_rows:
    print(constats[row],coniden[row],'\n')


# In[11]:


#search coniden for strongest hits
#strong for pd:   Actc1 Cacng1      
#                 C2 (viral response!, could also split mNG from true (-) cells -expect mNG to have more rna)
#                 Egf (cardiac-associated, very strong from perturbation)

count = 0
strong_genes = []
for row in range(len(constats)):
    if np.product(constats[row][:,2]) > 10 and constats[row][0,2] > 2 and constats[row][1,2] > 2 and constats[row][2,2] > 2:
#         print(coniden[row])
#         print(constats[row])
        count+=1
        strong_genes.append(ce_rna.var_names[coniden[row][3]])
print(count)
print(strong_genes)
print(len(set(strong_genes)))


# In[24]:


#prom id
ce_atac.var_names[127487]


# In[23]:


#enh id
ce_atac.var_names[127475]


# In[77]:


l = list(gs)
genestr = ''
for gene in l:
    genestr += gene + '\n'

with open('pd_hits.txt','w') as file:
    file.write(genestr)


# In[77]:


#TEMPORARY LOOK AT ELLIOTT'S LISTS


# In[5]:


fl = pkl.load(open('/project/GCRB/Hon_lab/shared/elliot/completeGraphsBigFcStats.pkl','rb'))


# In[13]:


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', 10)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')


# In[19]:


gn = set(list(fl['gene_name']))
gn


# ## OLDER CODE

# In[19]:


#prepare enhprom_stats
d1 = []
for elem in enhprom_stats:
    eleml = []
    for pert in elem:
        new_pert = pert.copy()
        new_pert[0] = np.log10(1/(min(new_pert[0],1-new_pert[0])+1e-20))
        eleml.append(new_pert)
    d1+=eleml


# In[29]:


#prepare enhrna_stats
d1 = []
for elem in enhrna_stats:
    eleml = []
    for pert in elem:
        new_pert = pert.copy()
        new_pert[0] = np.log10(1/(min(new_pert[0],1-new_pert[0])+1e-20))
        eleml.append(new_pert)
    d1+=eleml


# In[25]:


#prepare promrna_stats
d1 = [elem[0] for elem in promrna_stats]
d1 = np.asarray(d1)
p1 = [np.log10(1/(min(elem,1-elem)+1e-20)) for elem in d1[:,0]]
d1[:,0] = p1

#log the fc values for promrna_stats
d1[:,2] = np.log2(d1[:,2]+1e-20)


# In[120]:


#prepare pertenh_stats

pertenl = []
for tss in pertenh_stats:
    tssl = []
    for comparison in tss:
        tssl.append(comparison)
    pertenl+=tssl
perten = pertenl

d1 = []
for elem in perten:
    eleml = []
    for pert in elem:
        new_pert = pert.copy()
        new_pert[0] = np.log10(1/(min(new_pert[0],1-new_pert[0])+1e-20))
        eleml.append(new_pert)
    d1+=eleml


# In[13]:


#prepare pertprom_stats
d1 = []
for elem in pertprom_stats:
    eleml = []
    for pert in elem:
        new_pert = pert.copy()
        new_pert[0] = np.log10(1/(min(new_pert[0],1-new_pert[0])+1e-20))
        eleml.append(new_pert)
    d1+=eleml


# In[6]:


#prepare pertrna_stats
d1 = []
for elem in pertrna_stats:
    eleml = []
    for pert in elem:
        new_pert = pert.copy()
        new_pert[0] = np.log10(1/(min(new_pert[0],1-new_pert[0])+1e-20))
        eleml.append(new_pert)
    d1+=eleml


# In[121]:


len(d1)


# In[30]:


sigl = [row for row in d1 if row[0] > np.log10(1/(0.05))]
print(len(sigl))
bonsigl = [row for row in d1 if row[0] > np.log10(1/(0.05/len(d1)))]
print(len(bonsigl))


# In[32]:


len(apl)


# In[31]:


apl = [row for row in d1 if row[1] < 5]


# In[34]:


count = 0
for row1 in range(len(apl)):
    for row2 in range(len(bonsigl)):
        if np.array_equal(apl[row1],bonsigl[row2]):
            count+=1
            break
count


# In[33]:


thresh = 4.0381
sigl = [row for row in d1 if row[0] > thresh]
fakel = [row[1] for row in d1 if row[0] > thresh]
apl = [row for row in d1 if row[1] < 5]

p = len(sigl)
print(p)
ap = 0
for fakeps in fakel:
    ap+=fakeps*10
fdr = ap/(p+ap)
print(fdr)


# In[86]:


#p-val thresholds
#     pert->enh
#3.6047    pert->prom
#3.7191    pert->rna
#3.91005   enh->prom
#4.0381    enh->rna
#3.821     prom->rna


# In[86]:





# In[50]:


#FOR ATAC->RNA                   FOR PERT->ATAC                       FOR PERT->RNA
    #pfcon,prompvals                 prom_pert_stats                      rna_pert_stats
    #efcon,enhpvals                  enh_pert_stats

#prepare array from promoter perspective where each row is a prom and columns are:
#prFCrna,prPVrna,peFCpr,pePVpr,peFCrna,pePVrna
#1_____________________________________________________________________________________________________________________________
prFCrna = pfcon
#2_____________________________________________________________________________________________________________________________
prPVrna = [np.log10(1/(elem+1e-15)) for elem in prompvals]
#3_____________________________________________________________________________________________________________________________
peFCpr = []
for elem in prom_pert_stats:
    peFCpr.append([elem[pert][1] for pert in range(len(elem))])
#4_____________________________________________________________________________________________________________________________
pePVpr = []
for elem in prom_pert_stats:
    pePVpr.append([np.log10(1/(elem[pert][0]+1e-15)) for pert in range(len(elem))])
#5_____________________________________________________________________________________________________________________________
peFCrna = []
for elem in rna_pert_stats:
    peFCrna.append([elem[pert][1] for pert in range(len(elem))])
#6_____________________________________________________________________________________________________________________________
pePVrna = []
for elem in rna_pert_stats:
    pePVrna.append([np.log10(1/(elem[pert][0]+1e-15)) for pert in range(len(elem))])
#______________________________________________________________________________________________________________________________
a1,a2 = np.array(prFCrna),np.reshape(np.array(prPVrna),(-1,1))
a3,a4 = np.array(peFCpr),np.array(pePVpr)
a5,a6 = np.array(peFCrna),np.array(pePVrna)
data_all = np.concatenate((a1,a2,a3,a4,a5,a6),axis=1)
data_prtorna = np.concatenate((a1,a2),axis=1)
data_petoprom = np.concatenate((a3,a4),axis=1)
data_petorna = np.concatenate((a5,a6),axis=1)


# In[162]:


#poorly written ad hoc interlacer
x=data_petoprom
y=data_petorna

newpr = np.column_stack((x[:,0],x[:,16],x[:,1],x[:,17],x[:,2],x[:,18],x[:,3],x[:,19],x[:,4],x[:,20],x[:,5],x[:,21],x[:,6],x[:,22],x[:,7],x[:,23],
                       x[:,8],x[:,24],x[:,9],x[:,25],x[:,10],x[:,26],x[:,11],x[:,27],x[:,12],x[:,28],x[:,13],x[:,29],x[:,14],x[:,30],x[:,15],x[:,31],))
newrna = np.column_stack((y[:,0],y[:,16],y[:,1],y[:,17],y[:,2],y[:,18],y[:,3],y[:,19],y[:,4],y[:,20],y[:,5],y[:,21],y[:,6],y[:,22],y[:,7],y[:,23],
                       y[:,8],y[:,24],y[:,9],y[:,25],y[:,10],y[:,26],y[:,11],y[:,27],y[:,12],y[:,28],y[:,13],y[:,29],y[:,14],y[:,30],y[:,15],y[:,31],))


# In[24]:


#filter out TSSs that don't pass bonferroni for prPVrna
keep_l = [prPVrna.index(elem) for elem in prPVrna if elem > np.log10(8747/0.05)]
sig_data = data_all[keep_l,:]

ordered_l = np.flip(np.sort(sig_data[:,0],axis=0))
o_index = [list(sig_data[:,0]).index(elem) for elem in ordered_l]
o_data = sig_data[o_index,:]


# In[212]:


#iterative filtering

#filter out TSSs that don't pass bonferroni for prPVrna
keep_prtorna = [prPVrna.index(elem) for elem in prPVrna if elem > np.log10(8747/0.05)]
keep_


# In[216]:


#sort among highest foldchange pert-prom and pert-rna pairs, given bonferroni is met
#it = np.nditer(#data_array_here#,flags=['multi_index'])
#for x in it:
#    x = elem and it.multi_index = indices in #data_array_here#

#maybe do iterative filtering from original data_all to find any pert-prom pairs where p-val meets bonferroni for
# pert->prom     pert->rna     prom->rna
#and foldchange is highest at the same time

#if none meet all 3 cutoffs, use broader pert groupings (0F,1F,2F,3F,4F)?

#same basic process can be repeated for enh stats, to hopefully find an example where
## pert->prom    pert->enh    pert->rna     prom->rna    enh->rna 
#all meet bonferronia cutoffs AND have strong foldchange


# In[70]:


#xlabels = ['NC','','1F','','','','','','','','2F','','','','','','','','','','','','3F','','','','','','','','4F','']
#fig, ax = plt.subplots(figsize=(10,15))
sb.clustermap(d1,col_cluster=False,vmin=0,vmax=5.25,cmap='viridis',figsize=(15,50))


# ### Pert -> ENH Stats

# In[11]:


#FOR ATAC->RNA                   FOR PERT->ATAC                       FOR PERT->RNA
    #efcon,enhpvals                  enh_pert_stats                       rna_pert_stats

#make all enhs into a single list
efcon_l,enhp_l,enhps_l = [],[],[]
for tss_ind in range(len(enhpvals)):
    for peak_ind in range(len(enhpvals[tss_ind])):
        efcon_l.append(efcon[tss_ind][peak_ind])
        enhp_l.append(enhpvals[tss_ind][peak_ind])
        enhps_l.append(enh_pert_stats[tss_ind][peak_ind])
for index in range(len(enhp_l)):
    if enhp_l[index] > 0.5:
        enhp_l[index] = 1 - enhp_l[index]

    
#prepare array from enhancer perspective where each row is an enh and columns are:
#enFCrna,enPVrna,peFCen,pePVen,peFCrna,pePVrna
#1_____________________________________________________________________________________________________________________________
enFCrna = [np.log2(elem+1e-2) for elem in efcon_l]
#2_____________________________________________________________________________________________________________________________
enPVrna = [np.log10(1/(elem+1e-15)) for elem in enhp_l]
#3_____________________________________________________________________________________________________________________________
peFCen = []
for elem in enhps_l:
    peFCen.append([np.log2(elem[pert][1]+1e-2) for pert in range(len(elem))])
#4_____________________________________________________________________________________________________________________________
pePVen = []
for elem in enhps_l:
    pePVen.append([np.log10(1/(elem[pert][0]+1e-15)) for pert in range(len(elem))])
#5_____________________________________________________________________________________________________________________________
peFCrna = []
for elem in rna_pert_stats:
    peFCrna.append([np.log2(elem[pert][1]+1e-2) for pert in range(len(elem))])
#6_____________________________________________________________________________________________________________________________
pePVrna = []
for elem in rna_pert_stats:
    pePVrna.append([np.log10(1/(elem[pert][0]+1e-15)) for pert in range(len(elem))])
#______________________________________________________________________________________________________________________________
a1,a2 = np.reshape(np.array(enFCrna),(-1,1)),np.reshape(np.array(enPVrna),(-1,1))
a3,a4 = np.array(peFCen),np.array(pePVen)
a5,a6 = np.array(peFCrna),np.array(pePVrna)
data_all = np.concatenate((a1,a2,a3,a4),axis=1)
data_prtorna = np.concatenate((a1,a2),axis=1)
data_petoprom = np.concatenate((a3,a4),axis=1)
data_petorna = np.concatenate((a5,a6),axis=1)


# In[162]:


#poorly written ad hoc interlacer
x=data_petoprom
y=data_petorna

newpr = np.column_stack((x[:,0],x[:,16],x[:,1],x[:,17],x[:,2],x[:,18],x[:,3],x[:,19],x[:,4],x[:,20],x[:,5],x[:,21],x[:,6],x[:,22],x[:,7],x[:,23],
                       x[:,8],x[:,24],x[:,9],x[:,25],x[:,10],x[:,26],x[:,11],x[:,27],x[:,12],x[:,28],x[:,13],x[:,29],x[:,14],x[:,30],x[:,15],x[:,31],))
newrna = np.column_stack((y[:,0],y[:,16],y[:,1],y[:,17],y[:,2],y[:,18],y[:,3],y[:,19],y[:,4],y[:,20],y[:,5],y[:,21],y[:,6],y[:,22],y[:,7],y[:,23],
                       y[:,8],y[:,24],y[:,9],y[:,25],y[:,10],y[:,26],y[:,11],y[:,27],y[:,12],y[:,28],y[:,13],y[:,29],y[:,14],y[:,30],y[:,15],y[:,31],))


# In[24]:


#filter out TSSs that don't pass bonferroni for prPVrna
keep_l = [prPVrna.index(elem) for elem in prPVrna if elem > np.log10(8747/0.05)]
sig_data = data_all[keep_l,:]

ordered_l = np.flip(np.sort(sig_data[:,0],axis=0))
o_index = [list(sig_data[:,0]).index(elem) for elem in ordered_l]
o_data = sig_data[o_index,:]


# In[212]:


#iterative filtering

#filter out TSSs that don't pass bonferroni for prPVrna
keep_prtorna = [prPVrna.index(elem) for elem in prPVrna if elem > np.log10(8747/0.05)]
keep_


# In[216]:


#sort among highest foldchange pert-prom and pert-rna pairs, given bonferroni is met
#it = np.nditer(#data_array_here#,flags=['multi_index'])
#for x in it:
#    x = elem and it.multi_index = indices in #data_array_here#

#maybe do iterative filtering from original data_all to find any pert-prom pairs where p-val meets bonferroni for
# pert->prom     pert->rna     prom->rna
#and foldchange is highest at the same time

#if none meet all 3 cutoffs, use broader pert groupings (0F,1F,2F,3F,4F)?

#same basic process can be repeated for enh stats, to hopefully find an example where
## pert->prom    pert->enh    pert->rna     prom->rna    enh->rna 
#all meet bonferronia cutoffs AND have strong foldchange


# In[28]:


xlabels = ['NC','','1F','','','','','','','','2F','','','','','','','','','','','','3F','','','','','','','','4F','']
#fig, ax = plt.subplots(figsize=(10,15))
pcut = np.log10(1/(0.05/len(efcon_l)))

newarray = data_all[0:50000,:]
newarray.shape


# In[29]:


sb.clustermap(newarray,col_cluster=False,vmin=1,vmax=pcut,cmap='viridis')


# In[ ]:


#OLD CODE


# ### 1.3 P-Vals With Random RNA

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nr_thresh= (1e6/np.mean(ce_rna.obs[\'n_counts\'])) - 1 #roughly 101\na_thresh = 0 #since ATAC is binarized, no CPM norm can be done\n\nrand_enhp = []\nstart_num = 0\niteration_num = 1\nfor run_num in range(start_num,start_num+iteration_num):\n    #randomly mix rna_index values to create "rand_enhol" \n    rand_enhol = pkl.load(open(\'ce_enh_overlaps.pkl\',\'rb\')) \n    rna_ind_list = [elem[1] for elem in rand_enhol]\n    for elem in rand_enhol:\n        elem[1] = np.random.choice(rna_ind_list)\n        rna_ind_list.remove(elem[1])\n\n    #set up hypergeom testing for each item in enh_overlaps and generate p val list list  (should be 8747 long)\n    enhp = []\n    for these_enhs in rand_enhol:\n        rna_ind = these_enhs[1]\n        rp = [ce_rna.obs_names[cellid] for cellid in range(len(ce_rna.obs_names)) if ce_rna.X[cellid,rna_ind] > r_thresh]\n\n        #iterate through all enh for this tss\n        atac_inds = these_enhs[2]\n        pvals_this_tss = []\n        for atac_ind in atac_inds:\n            ap = [ce_atac.obs_names[cellid] for cellid in range(len(ce_atac.obs_names)) if ce_atac.X[cellid,atac_ind] > a_thresh]\n            aprp = [cellid for cellid in rp if cellid in ap]\n\n            k = len(aprp)                                                   #observed successes\n            M = len(ce_rna.obs_names)                                       #total population\n            n = len(rp)                                                     #total successes\n            N = len(ap)                                                     #total observed\n\n            pval = 1-ss.hypergeom.cdf(k,M,n,N)\n            pvals_this_tss.append(pval)\n        enhp.append(pvals_this_tss)\n    rand_enhp.append(enhp)')


# In[ ]:


#save p val list, each val list corresponds to a peak list in ce_enh_overlaps.pkl, for a given iteration
with open('nce_rand_enh_pvals_1.pkl', 'wb') as crep_file:
    pkl.dump(rand_enhp,crep_file)


# ### 1.4 MISC. Enhancer Stats

# In[9]:


#unique peaks average
elist = []
for which_tss in enhol:
    for which_peak in which_tss[2]:
        elist.append(which_peak)
print('TOTAL PEAKS =',len(elist))
siglist = set(elist)
print('UNIQUE PEAKS =',len(siglist))


# In[10]:


#peaks passing bonferroni
enhpvals = pkl.load(open('ce_enh_pvals.pkl','rb'))
plist = []
for which_tss in enhpvals:
    for which_peak in which_tss:
        plist.append(which_peak)
print('TOTAL PEAKS =',len(plist))
siglist = [val for val in plist if val <= 0.05/len(plist)]
print('BONFERRONI-PASSING PEAKS =',len(siglist))


# In[11]:


#inverse log p values
ilist = [1/(value+0.0000000001) for value in plist]
llist = [np.log10(value) for value in ilist]
sb.violinplot(llist)


# In[12]:


#average peaks detected per TSS
enh_nums = [len(which_tss[2]) for which_tss in enhol]

print('MEAN =',np.mean(enh_nums),'   MEDIAN =',np.median(enh_nums))
sb.violinplot(enh_nums)


# # 2.1 Global Promoter Scanning

# In[18]:


#create list of index triples [tss_ind,rna_ind,atac_ind] for all available tss
atac = ce_atac.copy()
tss = ce_tss.copy()
prom_overlaps = []

pos_col,gid_col = 1,2
this_peak,this_tss = 0,0 #initialize atac and tss index to beginning

while this_tss < len(tss):
    this_pos = tss.iloc[this_tss,pos_col]
    if atac.var['start'][this_peak] > this_pos:
        this_tss += 1
    elif atac.var['end'][this_peak] < this_pos:
        this_peak += 1
    else:
        tss_index = this_tss
        atac_index = this_peak
        gid = tss.iloc[this_tss,gid_col]
        rna_index = list(ce_rna.var_names).index(tss.iloc[this_tss,gid_col])
        
        this_overlap = [tss_index,rna_index,atac_index]
        prom_overlaps.append(this_overlap)
        
        this_tss += 1
        this_peak += 1

with open('ce_prom_overlaps.pkl', 'wb') as cpo_file:
    pkl.dump(prom_overlaps,cpo_file)


# In[19]:


#load in prom_overlaps
promol = pkl.load(open('ce_prom_overlaps.pkl','rb'))


# ### 2.2 Prom P-Val Generation

# In[ ]:


get_ipython().run_cell_magic('time', '', "#set up hypergeom testing for each item in prom_overlaps and generate p val list   (should be 8747 long)\npromp = []\nfor this_prom in promol:\n    rna_ind,atac_ind = this_prom[1],this_prom[2]\n    \n    rp = [ce_rna.obs_names[cellid] for cellid in range(len(ce_rna.obs_names)) if ce_rna.X[cellid,rna_ind] > 0]\n    ap = [ce_atac.obs_names[cellid] for cellid in range(len(ce_atac.obs_names)) if ce_atac.X[cellid,atac_ind] > 0]\n    aprp = [cellid for cellid in rp if cellid in ap]\n    \n    k = len(aprp)                                                   #observed successes\n    M = len(ce_rna.obs_names)                                       #total population\n    n = len(rp)                                                     #total successes\n    N = len(ap)                                                     #total observed\n\n    pval = 1-ss.hypergeom.cdf(k,M,n,N)\n    promp.append(pval)\n    \n#save p val list, each val corresponds to the tss in ce_prom_overlaps.pkl\nwith open('ce_prom_pvals.pkl', 'wb') as cpp_file:\n    pkl.dump(promp,cpp_file)")


# ### 2.3 MISC. Prom Stats

# In[ ]:


prompvals = pkl.load(open('ce_prom_pvals.pkl','rb'))
print(len(prompvals))
bon_sig = [value for value in prompvals if value <= 0.05/len(prompvals)]
print(len(bon_sig))


# In[ ]:


#prom FC generation


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nr_thresh= (1e6/np.mean(ce_rna.obs['n_counts'])) - 1 #roughly 101\na_thresh = 0 #since ATAC is binarized, no CPM norm can be done\n\n#set up hypergeom testing for each item in enh_overlaps and generate p val list list  (should be 8747 long)\nprom_fcon,prom_fcot = [],[]\nfor this_prom in promol:\n    rna_ind = this_prom[1]\n    rp = [ce_rna.obs_names[cellid] for cellid in range(len(ce_rna.obs_names)) if ce_rna.X[cellid,rna_ind] > r_thresh]\n    \n    #iterate through all enh for this tss\n    atac_ind = this_prom[2]\n    fcon_this_tss,fcot_this_tss = [],[]\n    \n    ap = [ce_atac.obs_names[cellid] for cellid in range(len(ce_atac.obs_names)) if ce_atac.X[cellid,atac_ind] > a_thresh]\n    aprp = [cellid for cellid in rp if cellid in ap]\n    \n    k = len(aprp)                                                   #observed successes\n    M = len(ce_rna.obs_names)                                       #total population\n    n = len(rp)                                                     #total successes\n    N = len(ap)                                                     #total observed\n\n    an = [cellid for cellid in ce_atac.obs_names if cellid not in ap]\n    anrp = [cellid for cellid in an if cellid in rp]\n    \n    if N == 0 or len(an) == 0:                                      #in event where all cells have atac peak or lack atac peak\n        fcon_this_tss.append(1)\n        fcot_this_tss.append(1)\n    \n    else:\n        prob_rga = k / N\n        prob_rgn = len(anrp) / len(an)\n        prob_r = n / M\n            \n        fcon_this_tss.append(prob_rga/prob_rgn)\n        fcot_this_tss.append(prob_rga/prob_r)\n        \n    prom_fcon.append(fcon_this_tss)\n    prom_fcot.append(fcot_this_tss)")


# In[ ]:


#save p val list, each val list corresponds to a peak list in ce_enh_overlaps.pkl
with open('nce_prom_fcon.pkl', 'wb') as cefcon_file:
    pkl.dump(prom_fcon,cefcon_file)
    
with open('nce_prom_fcot.pkl', 'wb') as cefcot_file:
    pkl.dump(prom_fcot,cefcot_file)


# In[14]:


#peaks passing bonferroni
prom_fc = pkl.load(open('ce_prom_fcon.pkl','rb'))
flist = []
for which_tss in prom_fc:
    for which_peak in which_tss:
        flist.append(which_peak)
print('TOTAL PEAKS =',len(flist))
siglist = [val for val in flist if val <= 0.05/len(flist)]
print('BONFERRONI-PASSING PEAKS =',len(siglist))


# In[18]:


#inverse log p values
ilist = [1/(value+0.0000000001) for value in flist]
llist = [np.log10(value) for value in ilist]
fig, ax = plt.subplots(figsize=(24,10))
sb.violinplot(flist)

