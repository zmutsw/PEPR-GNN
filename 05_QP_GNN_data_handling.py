
# coding: utf-8

import numpy as np
import scanpy as sc
import pandas as pd
import pickle as pkl
import seaborn as sb
import scipy.sparse as csr
from itertools import combinations
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from scipy.stats import gaussian_kde
import scipy.stats as ss
import math
from timeit import default_timer as dtime
import multiprocessing
from multiprocessing import Pool
from scipy.stats import percentileofscore as pos
import sys
import random
import networkx as nx
import copy
import csv

import torch
import torch_geometric as pyg
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, ResGatedGraphConv, SAGEConv, pool, to_hetero, to_hetero_with_bases
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Make TV and test different FC cut-offs

# In[2]:


orig = torch.load('pyg_hetlist_co.pt')


# In[2]:


hetl = torch.load('pyg_hetlist_predf3_trin25_co.pt')


# In[15]:


ally = [elem.y.item() for elem in hetl]
sb.histplot(ally)


# In[2]:


tv10 = torch.load('pyg_hetlist_tv10.pt')


# In[ ]:


#shows population of each deg-categ. depending on where cut-off is drawn          (linear cutoff of 0.1 is the basis of tv10)
hl = orig
count,cutoff = [0,0,0],2
for g in hl:
    tfr = g['perturbation','foldchange','rna'].edge_attr
    for elem in tfr:
        if 1/elem > cutoff:    #ratiometric
            count[0] += 1
        elif elem/1 > cutoff:    #ratiometric
            count[2] += 1
        else:
            count[1] += 1
print(count)
# total is 560,608
# [85,703, 429,726, 45,179] ratio 2
# [116,602, 375,286, 68,720] ratio 1.5
# [162,778, 288,185, 109,645] ratio 1.25
# [223,089, 159,397, 178,122] linear 1.1


# In[ ]:


temp_key,cutoff = [],1.25
for ind,g in enumerate(orig):
    tfr = g['perturbation','foldchange','rna'].edge_attr.clone().detach()
    tfr = torch.where(tfr>cutoff,2.0,tfr)
    tfr = torch.where(1/tfr>cutoff,0.0,tfr)
    tfr = torch.where((tfr-1)**2<1,1.0,tfr)
    temp_key.append(tfr)

# with open('ht_predtfr_key100.pkl', 'wb') as file:
#     pkl.dump(temp_key,file)


# In[ ]:


for idx,het in enumerate(tv10):
    het['y'] = temp_key[idx]
torch.save(tv10,'pyg_hetlist_tv25.pt')


# In[ ]:


temp_key,cutoff = [],1.5
for ind,g in enumerate(orig):
    tfr = g['perturbation','foldchange','rna'].edge_attr.clone().detach()
    tfr = torch.where(tfr>cutoff,2.0,tfr)
    tfr = torch.where(1/tfr>cutoff,0.0,tfr)
    tfr = torch.where((tfr-1)**2<1,1.0,tfr)
    temp_key.append(tfr)
for idx,het in enumerate(tv10):
    het['y'] = temp_key[idx]
torch.save(tv10,'pyg_hetlist_tv50.pt')


# In[ ]:


temp_key,cutoff = [],2
for ind,g in enumerate(orig):
    tfr = g['perturbation','foldchange','rna'].edge_attr.clone().detach()
    tfr = torch.where(tfr>cutoff,2.0,tfr)
    tfr = torch.where(1/tfr>cutoff,0.0,tfr)
    tfr = torch.where((tfr-1)**2<1,1.0,tfr)
    temp_key.append(tfr)
for idx,het in enumerate(tv10):
    het['y'] = temp_key[idx]
torch.save(tv10,'pyg_hetlist_tv100.pt')


# In[5]:


#need rna node dimensions to be adjusted
tv = torch.load('pyg_hetlist_tv10.pt')
for g in tv:
    g['rna'].x = g['rna'].x.reshape(1,1)
torch.save(tv,'pyg_hetlist_tv10.pt')
tv = torch.load('pyg_hetlist_tv25.pt')
for g in tv:
    g['rna'].x = g['rna'].x.reshape(1,1)
torch.save(tv,'pyg_hetlist_tv25.pt')
tv = torch.load('pyg_hetlist_tv50.pt')
for g in tv:
    g['rna'].x = g['rna'].x.reshape(1,1)
torch.save(tv,'pyg_hetlist_tv50.pt')
tv = torch.load('pyg_hetlist_tv100.pt')
for g in tv:
    g['rna'].x = g['rna'].x.reshape(1,1)
torch.save(tv,'pyg_hetlist_tv100.pt')


# In[2]:


#create trimmed dataset, removing graphs close to the cut-off between XXX
tv25 = torch.load('pyg_hetlist_tv25.pt')
tv50 = torch.load('pyg_hetlist_tv50.pt')


# In[37]:


p = []
for gid in range(len(tv25)):
    equiv = tv25[gid].y==tv50[gid].y
    le = equiv.tolist()
    p.append(np.sum(le)/16)


# In[40]:


np.mean(p),np.median(p)


# ## 6mer shuffling for polyA/attention bias

# In[2]:


#ghmt_motif explicator
tv25 = torch.load('pyg_hetlist_tv25_6merrchs.pt')


# In[3]:


hsrc_names = pkl.load(open('hsrc_names.pkl','rb'))


# In[36]:


#nine proposed motifs from olson paper + jaspar
# motif = 'AGATAA' #gata4 IN
# motif = 'GATAAG' #gata4
# motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
# motif = 'AAATAG' #Mef2c
# motif = 'AGGTGT'  #Tbx5 IN
# motif = 'GGTGTT'  #Tbx5 

#keep only ones already persent in rchs, includes top of each tf


# In[22]:


len(hsrc_names)


# In[6]:


ghmt_motifs = ['AGATAA','GATAAG','CAGATG','CAGCTG','AAAATA','AAAAAT','AAATAG','AGGTGT','GGTGTT']
ghmt_motif_ids = []
# hsrc_names.index(('AGATAA','TTATCT'))
for idx,rc in enumerate(hsrc_names):
    for motif in ghmt_motifs:
        if motif in rc:
            print(idx,rc)
            ghmt_motif_ids.append(idx)
ghmt_motif_ids.remove(806) #manually remove duplicate that somehow passed rc filtering


# In[12]:


motl = []
for rc in hsrc_names:
    for mot in rc:
        motl.append(mot)
#         if mot not in motl:
#             motl.append(mot)
print(len(motl),len(set(motl)))


# In[36]:


tv25 = torch.load('pyg_hetlist_tv25_6merrchs.pt')
hsrc_names = pkl.load(open('hsrc_names.pkl','rb'))
gata_id = hsrc_names.index(('AGATAA','TTATCT')) + 1


# In[74]:


#reindex 6mer matrix for gata4 motif is put at front
gmf_index = torch.tensor([0,gata_id]+list(range(1,gata_id))+list(range(gata_id+1,len(hsrc_names)+1)),dtype=torch.int)
for graph in tv25:
    graph['enhancer'].x = torch.index_select(graph['enhancer'].x,1,gmf_index)
torch.save(tv25,'pyg_hetlist_tv25_gmf.pt')


# In[ ]:


#reindex 6mer matrix so that motif positions are randomized
ran_index = torch.tensor([0]+np.random.randint(1,len(hsrc_names)+1, size=len(hsrc_names)),dtype=torch.int)
for graph in tv25:
    graph['enhancer'].x = torch.index_select(graph['enhancer'].x,1,ran_index)
torch.save(tv25,'pyg_hetlist_tv25_ran.pt')

hsrc_ran = [hsrc_names[which_id-1] for which_id in ran_index[1:]]
with open('hsrc_names_ran.pkl','wb') as file:
    pkl.dump(hsrc_ran,file)


# In[39]:


#reindex 6mer matrix so that motifs with >3 A in sequence or >4 A in total are excluded
tv25 = torch.load('pyg_hetlist_tv25_6merrchs.pt')
hsrc_names = pkl.load(open('hsrc_names.pkl','rb'))

#filter out polyA
nopa_index = [0]+[idx+1 for idx,elem in enumerate(hsrc_names) if ('AAAA' not in elem[0]) and ('AAAA' not in elem[-1])     #A sequence conditions
                                                         and (elem[0].count('A') < 5) and (elem[-1].count('A') < 5)]  #A total conditions
rng = np.random.default_rng()
rannopa_index = rng.permutation(nopa_index)
rannopa_index = torch.tensor(rannopa_index,dtype=torch.int)
for graph in tv25:
    graph['enhancer'].x = torch.index_select(graph['enhancer'].x,1,rannopa_index)
torch.save(tv25,'pyg_hetlist_tv25_rannopa.pt')

hsrc_rannopa = [hsrc_names[which_id-1] for which_id in rannopa_index[1:]]
with open('hsrc_names_rannopa.pkl','wb') as file:
    pkl.dump(hsrc_rannopa,file)


# In[73]:


#DWD


# In[83]:


s_dist = []
for motif_pair in hsrc_names:
    s = 0
    for bp in motif_pair[0]:
        if bp=='A' or bp=='T':
            s+=1
    s_dist.append(s)
subs = [elem for elem in s_dist if elem>=4]
print(len(subs)/len(s_dist))


# In[85]:


sb.histplot(s_dist,stat='probability')


# In[73]:


#END DWD


# ## 6mer modifications for tv25

# In[50]:


tv25 = torch.load('pyg_hetlist_tv25.pt')


# In[15]:


adata_6mer = sc.read('6mer_adata_co.h5ad') #adata of all atac 6mer vectors
def rev_comp(motif):
    rev_motif = ''
    for base in motif[::-1]:
        if base=='A':
            rev_motif+='T'
        if base=='C':
            rev_motif+='G'
        if base=='G':
            rev_motif+='C'
        if base=='T':
            rev_motif+='A'
    return rev_motif


# In[16]:


seqnames = list(adata_6mer.var_names)


# In[48]:


rc_seqs = []
for sid,seq in enumerate(seqnames):
    seqtup = sorted(list(set((seqnames.index(seq),seqnames.index(rev_comp(seq))))))
    if seqtup not in rc_seqs:
        rc_seqs.append(seqtup)


# In[51]:


#combine 6mer arrays into reverse-compliment-agostic arrays
new_tv = []
for gid,g in enumerate(tv25):
    numof_e = g['enhancer'].x.size()[0]
    newe = torch.empty((numof_e,len(rc_seqs)+1)) #initialize newe tensor
    newp = torch.empty((1,len(rc_seqs))) #initialize newp tensor
    newe[:,0] = g['enhancer'].x[:,0] #transfer distance from olde
    for sid in range(len(rc_seqs)): #populate 6mer values
        if len(rc_seqs[sid]) == 2:
            newe[:,1+sid] = g['enhancer'].x[:,1+rc_seqs[sid][0]]+g['enhancer'].x[:,1+rc_seqs[sid][1]]
            newp[:,sid] = g['promoter'].x[:,rc_seqs[sid][0]]+g['promoter'].x[:,rc_seqs[sid][1]]
        else:
            newe[:,1+sid] = g['enhancer'].x[:,1+rc_seqs[sid][0]]
            newp[:,sid] = g['promoter'].x[:,rc_seqs[sid][0]]
    g['enhancer'].x = newe
    g['promoter'].x = newp
    new_tv.append(g)

torch.save(new_tv,'pyg_hetlist_tv25_6merrc2.pt')


# In[52]:


#remove the lower standard-deviation-psm (see QH_6mer) 50% of reverse-compliment-agnostic arrays
tv25 = torch.load('pyg_hetlist_tv25_6merrc2.pt')
mat = pkl.load(open('promotif_smatrix_rc.pkl','rb'))
stl = []
for seq_ind,rctup in enumerate(rc_seqs):
    seq_vals = []
    for t in range(len(mat)):
        val = mat[t][seq_ind].item()
        seq_vals.append(pos(mat[t],val))
    st_thisseq = np.std(seq_vals)
    stl.append(st_thisseq)


# In[ ]:


high_std = [elem>14.15 for elem in stl] #14.15 is the midpoint of stl


# In[53]:


len(stl),len([elem>14.15 for elem in stl])


# In[ ]:


#restitch with above block, after adjusting midpoint cutoff value
hse = torch.Tensor([True]+high_std) #add starting True to account for distance element of enhancer.x
hse = hse.reshape((1,-1))>0 #reshape and cast to bool
hsp = torch.Tensor(high_std)
hsp = hsp.reshape((1,-1))>0 #reshape and cast to bool

for graph in tv25:
    graph['promoter'].x = graph['promoter'].x[hsp].reshape(1,-1)
    enum = len(graph['enhancer'].x)
    hse_scaled = hse.repeat(enum,1)
    graph['enhancer'].x = graph['enhancer'].x[hse_scaled].reshape(enum,-1)    

torch.save(tv25,'pyg_hetlist_tv25_6merrchs2.pt')


# In[5]:


#remove the higher standard-deviation-psm (see QH_6mer) 50% of reverse-compliment-agnostic arrays
tv25 = torch.load('pyg_hetlist_tv25_6merrc.pt')
mat = pkl.load(open('promotif_smatrix_rc.pkl','rb'))
stl = []
for seq_ind,rctup in enumerate(rc_seqs):
    seq_vals = []
    for t in range(len(mat)):
        val = mat[t][seq_ind].item()
        seq_vals.append(pos(mat[t],val))
    st_thisseq = np.std(seq_vals)
    stl.append(st_thisseq)
high_std = [elem<14.15 for elem in stl] #14.15 is the midpoint of stl

hse = torch.Tensor([True]+high_std) #add starting True to account for distance element of enhancer.x
hse = hse.reshape((1,-1))>0 #reshape and cast to bool
hsp = torch.Tensor(high_std)
hsp = hsp.reshape((1,-1))>0 #reshape and cast to bool

for graph in tv25:
    graph['promoter'].x = graph['promoter'].x[hsp].reshape(1,-1)
    enum = len(graph['enhancer'].x)
    hse_scaled = hse.repeat(enum,1)
    graph['enhancer'].x = graph['enhancer'].x[hse_scaled].reshape(enum,-1)    

torch.save(tv25,'pyg_hetlist_tv25_6merrcls.pt')


# In[3]:


#add tv50 ground truth to tv25 dataset for comparing accuracy outside of tv25-50 boundary
tv25 = torch.load('pyg_hetlist_tv25_6merrchs.pt')
tv50 = torch.load('pyg_hetlist_tv50.pt')


# In[5]:


for gid,graph in enumerate(tv25):
    graph.z = tv50[gid].y
torch.save(tv25,'pyg_hetlist_tv2550.pt')


# ## 6mer modifications for tv10

# In[6]:


tv = torch.load('pyg_hetlist_tv10.pt')


# In[7]:


adata_6mer = sc.read('6mer_adata_co.h5ad') #adata of all atac 6mer vectors
def rev_comp(motif):
    rev_motif = ''
    for base in motif[::-1]:
        if base=='A':
            rev_motif+='T'
        if base=='C':
            rev_motif+='G'
        if base=='G':
            rev_motif+='C'
        if base=='T':
            rev_motif+='A'
    return rev_motif

seqnames = list(adata_6mer.var_names)

rc_seqs = []
for sid,seq in enumerate(seqnames):
    seqtup = list(set((seqnames.index(seq),seqnames.index(rev_comp(seq)))))
    if seqtup not in rc_seqs:
        rc_seqs.append(seqtup)


# In[8]:


#combine 6mer arrays into reverse-compliment-agostic arrays
new_tv = []
for gid,g in enumerate(tv):
    numof_e = g['enhancer'].x.size()[0]
    newe = torch.empty((numof_e,len(rc_seqs)+1)) #initialize newe tensor
    newp = torch.empty((1,len(rc_seqs))) #initialize newp tensor
    newe[:,0] = g['enhancer'].x[:,0] #transfer distance from olde
    for sid in range(len(rc_seqs)): #populate 6mer values
        if len(rc_seqs[sid]) == 2:
            newe[:,1+sid] = g['enhancer'].x[:,1+rc_seqs[sid][0]]+g['enhancer'].x[:,1+rc_seqs[sid][1]]
            newp[:,sid] = g['promoter'].x[:,rc_seqs[sid][0]]+g['promoter'].x[:,rc_seqs[sid][1]]
        else:
            newe[:,1+sid] = g['enhancer'].x[:,1+rc_seqs[sid][0]]
            newp[:,sid] = g['promoter'].x[:,rc_seqs[sid][0]]
    g['enhancer'].x = newe
    g['promoter'].x = newp
    new_tv.append(g)

torch.save(new_tv,'pyg_hetlist_tv10_6merrc.pt')


# In[9]:


#remove the lower standard-deviation-psm (see QH_6mer) 50% of reverse-compliment-agnostic arrays
tv = torch.load('pyg_hetlist_tv10_6merrc.pt')
mat = pkl.load(open('promotif_smatrix_rc.pkl','rb'))
stl = []
for seq_ind,rctup in enumerate(rc_seqs):
    seq_vals = []
    for t in range(len(mat)):
        val = mat[t][seq_ind].item()
        seq_vals.append(pos(mat[t],val))
    st_thisseq = np.std(seq_vals)
    stl.append(st_thisseq)
high_std = [elem>14.15 for elem in stl] #14.15 is the midpoint of stl

hse = torch.Tensor([True]+high_std) #add starting True to account for distance element of enhancer.x
hse = hse.reshape((1,-1))>0 #reshape and cast to bool
hsp = torch.Tensor(high_std)
hsp = hsp.reshape((1,-1))>0 #reshape and cast to bool

for graph in tv:
    graph['promoter'].x = graph['promoter'].x[hsp].reshape(1,-1)
    enum = len(graph['enhancer'].x)
    hse_scaled = hse.repeat(enum,1)
    graph['enhancer'].x = graph['enhancer'].x[hse_scaled].reshape(enum,-1)    

torch.save(tv,'pyg_hetlist_tv10_6merrchs.pt')


# In[10]:


#add tv25 ground truth to tv10 dataset for comparing accuracy outside of tv10-25 boundary
tv10 = torch.load('pyg_hetlist_tv10_6merrchs.pt')
tv25 = torch.load('pyg_hetlist_tv25.pt')

for gid,graph in enumerate(tv10):
    graph.z = tv25[gid].y
torch.save(tv10,'pyg_hetlist_tv1025')


# ## Delete features for tfr_trin

# In[2]:


#temp,del when done (if you want)
tv25 = torch.load('pyg_hetlist_tv25.pt')
hetlist = tv25
for het in hetlist:
    del het['rna']
    del het['enhancer','foldchange','rna']
    del het['enhancer','pvalue','rna']
    del het['rna','rev_foldchange','enhancer']
    del het['rna','rev_pvalue','enhancer']
    del het['promoter','foldchange','rna']
    del het['promoter','pvalue','rna']
    del het['rna','rev_foldchange','promoter']
    del het['rna','rev_pvalue','promoter']
torch.save(hetlist,'pyg_hetlist_tv25_nor.pt')


# In[6]:


# hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
tv10 = torch.load('pyg_hetlist_tv10.pt')
tv25 = torch.load('pyg_hetlist_tv25.pt')
tv50 = torch.load('pyg_hetlist_tv50.pt')
tv100 = torch.load('pyg_hetlist_tv100.pt')


# In[44]:


key = [het['rna'].x for het in hetlist]
with open('ht_predtfr_key.pkl', 'wb') as file:
    pkl.dump(key,file)


# In[46]:


for idx,het in enumerate(hetlist):
    het['rna']['x'] = Tensor([1])
    del het['perturbation','foldchange','rna']
    del het['perturbation','pvalue','rna']
    del het['rna','rev_foldchange','perturbation']
    del het['rna','rev_pvalue','perturbation']
    het['y'] = key[idx]
torch.save(hetlist,'pyg_hetlist_tv10.pt')


# In[ ]:


hetlist = tv10
for het in hetlist:
    del het['rna']
    del het['enhancer','foldchange','rna']
    del het['enhancer','pvalue','rna']
    del het['rna','rev_foldchange','enhancer']
    del het['rna','rev_pvalue','enhancer']
    del het['promoter','foldchange','rna']
    del het['promoter','pvalue','rna']
    del het['rna','rev_foldchange','promoter']
    del het['rna','rev_pvalue','promoter']
torch.save(hetlist,'pyg_hetlist_tv10_nor.pt')

hetlist = tv25
for het in hetlist:
    del het['rna']
    del het['enhancer','foldchange','rna']
    del het['enhancer','pvalue','rna']
    del het['rna','rev_foldchange','enhancer']
    del het['rna','rev_pvalue','enhancer']
    del het['promoter','foldchange','rna']
    del het['promoter','pvalue','rna']
    del het['rna','rev_foldchange','promoter']
    del het['rna','rev_pvalue','promoter']
torch.save(hetlist,'pyg_hetlist_tv25_nor.pt')

hetlist = tv50
for het in hetlist:
    del het['rna']
    del het['enhancer','foldchange','rna']
    del het['enhancer','pvalue','rna']
    del het['rna','rev_foldchange','enhancer']
    del het['rna','rev_pvalue','enhancer']
    del het['promoter','foldchange','rna']
    del het['promoter','pvalue','rna']
    del het['rna','rev_foldchange','promoter']
    del het['rna','rev_pvalue','promoter']
torch.save(hetlist,'pyg_hetlist_tv50_nor.pt')

hetlist = tv100
for het in hetlist:
    del het['rna']
    del het['enhancer','foldchange','rna']
    del het['enhancer','pvalue','rna']
    del het['rna','rev_foldchange','enhancer']
    del het['rna','rev_pvalue','enhancer']
    del het['promoter','foldchange','rna']
    del het['promoter','pvalue','rna']
    del het['rna','rev_foldchange','promoter']
    del het['rna','rev_pvalue','promoter']
torch.save(hetlist,'pyg_hetlist_tv100_nor.pt')


# In[60]:


for het in hetlist:
    del het['rna']['x']
    del het['perturbation','foldchange','rna']
    del het['perturbation','pvalue','rna']
    del het['rna','rev_foldchange','perturbation']
    del het['rna','rev_pvalue','perturbation']
    del het['rna']
    del het['enhancer','foldchange','rna']
    del het['enhancer','pvalue','rna']
    del het['rna','rev_foldchange','enhancer']
    del het['rna','rev_pvalue','enhancer']
    del het['promoter','foldchange','rna']
    del het['promoter','pvalue','rna']
    del het['rna','rev_foldchange','promoter']
    del het['rna','rev_pvalue','promoter']
    del het['enhancer']['x']
    del het['promoter']['x']
torch.save(hetlist,'pyg_hetlist_tv_nords.pt')


# ## Shuffle features for tv25

# In[2]:


# hetlist = torch.load('pyg_hetlist_tv25.pt')
hfile_prefix,hfile_suffix = 'pyg_hetlist_tv25_6merrchs','.pt'
#BE SURE TO USE FUNCTIONS DEFINED BELOW


# In[6]:


#Node Shuffling

#perturbation_______________________________________________________________________________________________________________________________________________________________
hetlist = torch.load(hfile_prefix+hfile_suffix)
node_type,feature = 'perturbation','combo'
pool = node_compiler(node_type,feature,hetlist)
new_het = node_shuffler(node_type,feature,pool,hetlist)
torch.save(new_het,hfile_prefix+'_comboshuf'+hfile_suffix)
#enhancer___________________________________________________________________________________________________________________________________________________________________
hetlist = torch.load(hfile_prefix+hfile_suffix)
node_type,feature = 'enhancer','distance'
pool = node_compiler(node_type,feature,hetlist)
new_het = node_shuffler(node_type,feature,pool,hetlist)
torch.save(new_het,hfile_prefix+'_dishuf'+hfile_suffix)

hetlist = torch.load(hfile_prefix+hfile_suffix)
node_type,feature = 'enhancer','sequence'
pool = node_compiler(node_type,feature,hetlist)
new_het = node_shuffler(node_type,feature,pool,hetlist)
torch.save(new_het,hfile_prefix+'_eseqshuf'+hfile_suffix)
#promoter___________________________________________________________________________________________________________________________________________________________________
hetlist = torch.load(hfile_prefix+hfile_suffix)
node_type,feature = 'promoter','sequence'
pool = node_compiler(node_type,feature,hetlist)
new_het = node_shuffler(node_type,feature,pool,hetlist)
torch.save(new_het,hfile_prefix+'_pseqshuf'+hfile_suffix)
#e+p sequence_______________________________________________________________________________________________________________________________________________________________
hetlist = torch.load(hfile_prefix+hfile_suffix)
node_type,feature = 'enhancer','sequence'
pool = node_compiler(node_type,feature,hetlist)
new_het = node_shuffler(node_type,feature,pool,hetlist)
node_type,feature = 'promoter','sequence'
pool = node_compiler(node_type,feature,new_het)
new_het = node_shuffler(node_type,feature,pool,new_het)
torch.save(new_het,hfile_prefix+'_seqshuf'+hfile_suffix)
#all node features__________________________________________________________________________________________________________________________________________________________
hetlist = torch.load(hfile_prefix+hfile_suffix)
node_type,feature = 'perturbation','combo'
pool = node_compiler(node_type,feature,hetlist)
new_het = node_shuffler(node_type,feature,pool,hetlist)
node_type,feature = 'enhancer','distance'
pool = node_compiler(node_type,feature,new_het)
new_het = node_shuffler(node_type,feature,pool,new_het)
node_type,feature = 'enhancer','sequence'
pool = node_compiler(node_type,feature,new_het)
new_het = node_shuffler(node_type,feature,pool,new_het)
node_type,feature = 'promoter','sequence'
pool = node_compiler(node_type,feature,new_het)
new_het = node_shuffler(node_type,feature,pool,new_het)
torch.save(new_het,hfile_prefix+'_nodeshuf'+hfile_suffix)


# In[3]:


#DWD
#t->p edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('perturbation','foldchange','promoter'),('promoter','rev_foldchange','perturbation')),
            (('perturbation','pvalue','promoter'),('promoter','rev_pvalue','perturbation'))]
hetlist = torch.load('pyg_hetlist_tv25.pt')
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = 'tp'
torch.save(hetlist,'pyg_hetlist_tv25_'+mod_id+'shuf.pt')


# In[4]:


#DWD
#e->x edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('enhancer','foldchange','promoter'),('promoter','rev_foldchange','enhancer')),
            (('enhancer','pvalue','promoter'),('promoter','rev_pvalue','enhancer')),
            (('enhancer','foldchange','rna'),('rna','rev_foldchange','enhancer')),
            (('enhancer','pvalue','rna'),('rna','rev_pvalue','enhancer'))]
hetlist = torch.load('pyg_hetlist_tv25.pt')
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = 'ex'
torch.save(hetlist,'pyg_hetlist_tv25_'+mod_id+'shuf.pt')


# In[ ]:


#Edge Shuffling

#t->x edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('perturbation','foldchange','enhancer'),('enhancer','rev_foldchange','perturbation')),
            (('perturbation','pvalue','enhancer'),('enhancer','rev_pvalue','perturbation')),
            (('perturbation','foldchange','promoter'),('promoter','rev_foldchange','perturbation')),
            (('perturbation','pvalue','promoter'),('promoter','rev_pvalue','perturbation'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_txshuf'
torch.save(new_het,hfile_prefix+mod_id+hfile_suffix)
#t->e edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('perturbation','foldchange','enhancer'),('enhancer','rev_foldchange','perturbation')),
            (('perturbation','pvalue','enhancer'),('enhancer','rev_pvalue','perturbation'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_teshuf'
torch.save(new_het,hfile_prefix+mod_id+hfile_suffix)


# In[ ]:


#t->p edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('perturbation','foldchange','promoter'),('promoter','rev_foldchange','perturbation')),
            (('perturbation','pvalue','promoter'),('promoter','rev_pvalue','perturbation'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_tpshuf'
torch.save(hetlist,hfile_prefix+mod_id+hfile_suffix)
#e->x edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('enhancer','foldchange','promoter'),('promoter','rev_foldchange','enhancer')),
            (('enhancer','pvalue','promoter'),('promoter','rev_pvalue','enhancer')),
            (('enhancer','foldchange','rna'),('rna','rev_foldchange','enhancer')),
            (('enhancer','pvalue','rna'),('rna','rev_pvalue','enhancer'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_exshuf'
torch.save(hetlist,hfile_prefix+mod_id+hfile_suffix)
#e->p edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('enhancer','foldchange','promoter'),('promoter','rev_foldchange','enhancer')),
            (('enhancer','pvalue','promoter'),('promoter','rev_pvalue','enhancer'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_epshuf'
torch.save(hetlist,hfile_prefix+mod_id+hfile_suffix)
#e->r edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('enhancer','foldchange','rna'),('rna','rev_foldchange','enhancer')),
            (('enhancer','pvalue','rna'),('rna','rev_pvalue','enhancer'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_ershuf'
torch.save(hetlist,hfile_prefix+mod_id+hfile_suffix)
#p->r edges________________________________________________________________________________________________________________________________________________________________
tup_list = [(('promoter','foldchange','rna'),('rna','rev_foldchange','promoter')),
            (('promoter','pvalue','rna'),('rna','rev_pvalue','promoter'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_prshuf'
torch.save(hetlist,hfile_prefix+mod_id+hfile_suffix)
#indir edges_______________________________________________________________________________________________________________________________________________________________
tup_list = [(('perturbation','foldchange','promoter'),('promoter','rev_foldchange','perturbation')),
            (('perturbation','pvalue','promoter'),('promoter','rev_pvalue','perturbation')),
            (('enhancer','foldchange','rna'),('rna','rev_foldchange','enhancer')),
            (('enhancer','pvalue','rna'),('rna','rev_pvalue','enhancer'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_indirshuf'
torch.save(hetlist,hfile_prefix+mod_id+hfile_suffix)


# In[ ]:


#dir edges_________________________________________________________________________________________________________________________________________________________________
tup_list = [(('perturbation','foldchange','enhancer'),('enhancer','rev_foldchange','perturbation')),
            (('perturbation','pvalue','enhancer'),('enhancer','rev_pvalue','perturbation')),
            (('enhancer','foldchange','promoter'),('promoter','rev_foldchange','enhancer')),
            (('enhancer','pvalue','promoter'),('promoter','rev_pvalue','enhancer')),
            (('promoter','foldchange','rna'),('rna','rev_foldchange','promoter')),
            (('promoter','pvalue','rna'),('rna','rev_pvalue','promoter'))]
hetlist = torch.load(hfile_prefix+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    print(edge_tup)
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_dirshuf'
torch.save(new_het,hfile_prefix+mod_id+hfile_suffix)


# In[ ]:


#x->r edges________________________________________________________________________________________________________________________________________________________________

#all fc edges______________________________________________________________________________________________________________________________________________________________

#all pv edges______________________________________________________________________________________________________________________________________________________________

#all edges_________________________________________________________________________________________________________________________________________________________________
#load in te shuffled hetlist first, then apply rest of operations here
tup_list = [(('perturbation','foldchange','promoter'),('promoter','rev_foldchange','perturbation')),
            (('perturbation','pvalue','promoter'),('promoter','rev_pvalue','perturbation')),
            (('enhancer','foldchange','promoter'),('promoter','rev_foldchange','enhancer')),
            (('enhancer','pvalue','promoter'),('promoter','rev_pvalue','enhancer')),
            (('enhancer','foldchange','rna'),('rna','rev_foldchange','enhancer')),
            (('enhancer','pvalue','rna'),('rna','rev_pvalue','enhancer')),
            (('promoter','foldchange','rna'),('rna','rev_foldchange','promoter')),
            (('promoter','pvalue','rna'),('rna','rev_pvalue','promoter'))]
hetlist = torch.load(hfile_prefix+'_teshuf'+hfile_suffix)
for edge_tup,rev_tup in tup_list:
    print(edge_tup)
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = '_edgeshuf'
torch.save(hetlist,hfile_prefix+mod_id+hfile_suffix)


# ## Randomize features for tfr_trin

# In[2]:


hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')


# In[3]:


def edge_compiler(edge_tup,hetlist):
    pool = []
    for het in hetlist:
        thishet_pool = het[edge_tup].edge_attr.clone()
        for elem in thishet_pool:
            pool.append(elem)
    return pool

def edge_shuffler(edge_tup,pool,hetlist):
    for het in hetlist:
        for edge in range(het[edge_tup].edge_attr.size()[0]):
            het[edge_tup].edge_attr[edge] = pool.pop(int(torch.rand(1)*len(pool)))
    return hetlist

def node_compiler(node_type,feature,hetlist):
    pool = []
    if feature == 'sequence':    #assumes that sequence info. is all but first column of x
        for het in hetlist:
            thishet_pool = het[node_type].x[:,1:].clone()
            for elem in thishet_pool:
                pool.append(elem)
    elif feature == 'distance':    #assumes that distance info. is first column of x
        for het in hetlist:
            thishet_pool = het[node_type].x[:,0].clone()
            for elem in thishet_pool:
                pool.append(elem)
    elif feature == 'combo':
        for het in hetlist:
            thishet_pool = het[node_type].x.clone()
            for elem in thishet_pool:
                pool.append(elem)
    elif feature == 'tr':
        for het in hetlist:
            thishet_pool = het[node_type].x.clone()
            for elem in thishet_pool:
                pool.append(elem)
    return pool

def node_shuffler(node_type,feature,pool,hetlist):
    for het in hetlist:
        if feature == 'sequence':
            for node_seq in range(het[node_type].x[:,1:].size()[0]):
                het[node_type].x[node_seq,1:] = pool.pop(int(torch.rand(1)*len(pool)))
        if feature == 'distance':
            for node_dist in range(het[node_type].x[:,0].size()[0]):
                het[node_type].x[node_dist,0] = pool.pop(int(torch.rand(1)*len(pool)))
        if feature == 'combo':
            for node_combo in range(het[node_type].x.size()[0]):
                het[node_type].x[node_combo] = pool.pop(int(torch.rand(1)*len(pool)))
        if feature == 'tr':
            het[node_type].x = pool.pop(int(torch.rand(1)*len(pool))).reshape(1,16)
    return hetlist


# In[78]:


#shuffle enh,prom->rna, FCs and PVs individually
tup_list = [(('enhancer','foldchange','rna'),('rna','rev_foldchange','enhancer')),
            (('enhancer','pvalue','rna'),('rna','rev_pvalue','enhancer')),
            (('promoter','foldchange','rna'),('rna','rev_foldchange','promoter')),
            (('promoter','pvalue','rna'),('rna','rev_pvalue','promoter'))]
for edge_tup,rev_tup in tup_list:
    hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    new_hetlist = edge_shuffler(edge_tup,pool,hetlist)
    newer_hetlist = edge_shuffler(rev_tup,rev_pool,new_hetlist)
    mod_id = edge_tup[0][0]+edge_tup[1][0]+edge_tup[2][0]
    torch.save(newer_hetlist,'pyg_hetlist_predtfr_'+mod_id+'shuf_trin_co.pt')


# In[79]:


#shuffle enh,prom->rna, FCs and PVs all together
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
tup_list = [(('enhancer','foldchange','rna'),('rna','rev_foldchange','enhancer')),
            (('enhancer','pvalue','rna'),('rna','rev_pvalue','enhancer')),
            (('promoter','foldchange','rna'),('rna','rev_foldchange','promoter')),
            (('promoter','pvalue','rna'),('rna','rev_pvalue','promoter'))]
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
    mod_id = edge_tup[0][0]+edge_tup[1][0]+edge_tup[2][0]
torch.save(hetlist,'pyg_hetlist_predtfr_allrnashuf_trin_co.pt')


# In[ ]:


#shuffle pert->enh FCs and PVs
#note: this doesn't finish running in jupyter, maybe due to memory?   this code was copied to tes.py and ran as a slurm job
tup_list = [(('perturbation','foldchange','enhancer'),('enhancer','rev_foldchange','perturbation')),
            (('perturbation','pvalue','enhancer'),('enhancer','rev_pvalue','perturbation'))]
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
mod_id = 'te'
torch.save(hetlist,'pyg_hetlist_predtfr_'+mod_id+'shuf_trin_co.pt')


# In[4]:


#shuffle pert->prom FCs and PVs
tup_list = [(('perturbation','foldchange','promoter'),('promoter','rev_foldchange','perturbation')),
            (('perturbation','pvalue','promoter'),('promoter','rev_pvalue','perturbation'))]
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
    mod_id = 'tp'
torch.save(hetlist,'pyg_hetlist_predtfr_'+mod_id+'shuf_trin_co.pt')
    
#shuffle enh->prom FCs and PVs
tup_list = [(('enhancer','foldchange','promoter'),('promoter','rev_foldchange','enhancer')),
            (('enhancer','pvalue','promoter'),('promoter','rev_pvalue','enhancer'))]
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
    mod_id = 'ep'
torch.save(hetlist,'pyg_hetlist_predtfr_'+mod_id+'shuf_trin_co.pt')


# In[ ]:


#shuffle enh features
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
node_type,feature = 'enhancer','distance'
pool = node_compiler(node_type,feature,hetlist)
no_enh_di = node_shuffler(node_type,feature,pool,hetlist)
torch.save(no_enh_di,'pyg_hetlist_predtfr_edishuf_trin_co.pt')

hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
node_type,feature = 'enhancer','sequence'
pool = node_compiler(node_type,feature,hetlist)
no_enh_seq = node_shuffler(node_type,feature,pool,hetlist)
torch.save(no_enh_seq,'pyg_hetlist_predtfr_eseqshuf_trin_co.pt')

#shuffle prom features
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
node_type,feature = 'promoter','sequence'
pool = node_compiler(node_type,feature,hetlist)
no_prom_seq = node_shuffler(node_type,feature,pool,hetlist)
torch.save(no_prom_seq,'pyg_hetlist_predtfr_pseqshuf_trin_co.pt')


# In[ ]:


#shuffle all enh,prom features
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
node_type,feature = 'promoter','sequence'
pool = node_compiler(node_type,feature,hetlist)
hetlist = node_shuffler(node_type,feature,pool,hetlist)

node_type,feature = 'enhancer','sequence'
pool = node_compiler(node_type,feature,hetlist)
hetlist = node_shuffler(node_type,feature,pool,hetlist)

node_type,feature = 'enhancer','distance'
pool = node_compiler(node_type,feature,hetlist)
hetlist = node_shuffler(node_type,feature,pool,hetlist)

torch.save(hetlist,'pyg_hetlist_predtfr_diseqshuf_trin_co.pt')


# In[29]:


#shuffle perturbation features
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
node_type,feature = 'perturbation','combo'
pool = node_compiler(node_type,feature,hetlist[0:1])     #only fed in first het because all hets share the same pert combo feature
no_pert_combo = node_shuffler(node_type,feature,pool,hetlist[0:1])
torch.save(hetlist,'pyg_hetlist_predtfr_tcomboshuf_trin_co.pt')


# In[6]:


#shuffle rna features :(
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
node_type,feature = 'rna','tr'
pool = node_compiler(node_type,feature,hetlist)
no_tr = node_shuffler(node_type,feature,pool,hetlist)
torch.save(hetlist,'pyg_hetlist_predtfr_rnodeshuf_trin_co.pt')


# In[5]:


#shuffle all node features
hetlist = torch.load('pyg_hetlist_predtfr_trin_co.pt')
#t
node_type,feature = 'perturbation','combo'
pool = node_compiler(node_type,feature,hetlist[0:1])     #only fed in first het because all hets share the same pert combo feature
no_pert_combo = node_shuffler(node_type,feature,pool,hetlist[0:1]) #modifies het_list in place
#e
node_type,feature = 'enhancer','distance'
pool = node_compiler(node_type,feature,hetlist)
no_enh_di = node_shuffler(node_type,feature,pool,hetlist)
node_type,feature = 'enhancer','sequence'
pool = node_compiler(node_type,feature,hetlist)
no_enh_seq = node_shuffler(node_type,feature,pool,hetlist)
#p
node_type,feature = 'promoter','sequence'
pool = node_compiler(node_type,feature,hetlist)
no_prom_seq = node_shuffler(node_type,feature,pool,hetlist)

torch.save(hetlist,'pyg_hetlist_predtfr_allnodeshuf_trin_co.pt')


# In[6]:


#shuffle all edge features
hetlist = torch.load('pyg_hetlist_predtfr_teshuf_trin_co.pt') #load in te already shuffled to save time/space
#tp
tup_list = [(('perturbation','foldchange','promoter'),('promoter','rev_foldchange','perturbation')),
            (('perturbation','pvalue','promoter'),('promoter','rev_pvalue','perturbation'))]
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
#ep
tup_list = [(('enhancer','foldchange','promoter'),('promoter','rev_foldchange','enhancer')),
            (('enhancer','pvalue','promoter'),('promoter','rev_pvalue','enhancer'))]
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
#er
tup_list = [(('enhancer','foldchange','rna'),('rna','rev_foldchange','enhancer')),
            (('enhancer','pvalue','rna'),('rna','rev_pvalue','enhancer'))]
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)
#pr
tup_list = [(('promoter','foldchange','rna'),('rna','rev_foldchange','promoter')),
            (('promoter','pvalue','rna'),('rna','rev_pvalue','promoter'))]
for edge_tup,rev_tup in tup_list:
    pool,rev_pool = edge_compiler(edge_tup,hetlist),edge_compiler(rev_tup,hetlist)
    hetlist = edge_shuffler(edge_tup,pool,hetlist)
    hetlist = edge_shuffler(rev_tup,rev_pool,hetlist)

torch.save(hetlist,'pyg_hetlist_predtfr_alledgeshuf_trin_co.pt')


# In[7]:


#shuffle ALL features
hetlist = torch.load('pyg_hetlist_predtfr_alledgeshuf_trin_co.pt') #load in edges already shuffled to save time/space
#t
node_type,feature = 'perturbation','combo'
pool = node_compiler(node_type,feature,hetlist[0:1])     #only fed in first het because all hets share the same pert combo feature
no_pert_combo = node_shuffler(node_type,feature,pool,hetlist[0:1]) #modifies het_list in place
#e
node_type,feature = 'enhancer','distance'
pool = node_compiler(node_type,feature,hetlist)
no_enh_di = node_shuffler(node_type,feature,pool,hetlist) #modifies het_list in place
node_type,feature = 'enhancer','sequence'
pool = node_compiler(node_type,feature,hetlist)
no_enh_seq = node_shuffler(node_type,feature,pool,hetlist) #modifies het_list in place
#p
node_type,feature = 'promoter','sequence'
pool = node_compiler(node_type,feature,hetlist)
no_prom_seq = node_shuffler(node_type,feature,pool,hetlist) #modifies het_list in place


torch.save(hetlist,'pyg_hetlist_predtfr_omnishuf_trin_co.pt')


# ## REMAKE OF FIG 3 DATASETS, GHMT/NC->R BINARY & TRINARY

# In[ ]:


orig = torch.load('pyg_hetlist_co.pt')
cutoff = 1.25 #25% difference
for graph in orig:
    graph['rna'].x = Tensor([[1]])
    ghmt_edge,nc_edge = graph['perturbation','foldchange','rna'].edge_attr[15],graph['perturbation','foldchange','rna'].edge_attr[0]
    if ghmt_edge/nc_edge >= cutoff:
        graph['y'] = Tensor([2.0])
    elif nc_edge/ghmt_edge >= cutoff:
        graph['y'] = Tensor([0.0])
    else:
        graph['y'] = Tensor([1.0])
    del graph['perturbation','foldchange','rna']
    del graph['rna','rev_foldchange','perturbation']
    del graph['perturbation','pvalue','rna']
    del graph['rna','rev_pvalue','perturbation']
torch.save(orig,'pyg_hetlist_predf3_trin25_co.pt')


# In[ ]:


orig = torch.load('pyg_hetlist_co.pt')
cutoff = 1.5 #50% difference
for graph in orig:
    graph['rna'].x = Tensor([[1]])
    ghmt_edge,nc_edge = graph['perturbation','foldchange','rna'].edge_attr[15],graph['perturbation','foldchange','rna'].edge_attr[0]
    if ghmt_edge/nc_edge >= cutoff:
        graph['y'] = Tensor([2.0])
    elif nc_edge/ghmt_edge >= cutoff:
        graph['y'] = Tensor([0.0])
    else:
        graph['y'] = Tensor([1.0])
    del graph['perturbation','foldchange','rna']
    del graph['rna','rev_foldchange','perturbation']
    del graph['perturbation','pvalue','rna']
    del graph['rna','rev_pvalue','perturbation']
torch.save(orig,'pyg_hetlist_predf3_trin50_co.pt')


# In[ ]:


orig = torch.load('pyg_hetlist_co.pt')
cutoff = 1 #0% difference
for graph in orig:
    graph['rna'].x = Tensor([[1]])
    ghmt_edge,nc_edge = graph['perturbation','foldchange','rna'].edge_attr[15],graph['perturbation','foldchange','rna'].edge_attr[0]
    if ghmt_edge/nc_edge >= cutoff:
        graph['y'] = Tensor([1.0])
    else:
        graph['y'] = Tensor([0.0])
    del graph['perturbation','foldchange','rna']
    del graph['rna','rev_foldchange','perturbation']
    del graph['perturbation','pvalue','rna']
    del graph['rna','rev_pvalue','perturbation']
torch.save(orig,'pyg_hetlist_predf2_bin_co.pt')


# In[2]:


f325 = torch.load('pyg_hetlist_predf3_trin25_co.pt')
f350 = torch.load('pyg_hetlist_predf3_trin50_co.pt')
f2 = torch.load('pyg_hetlist_predf2_bin_co.pt')


# In[16]:


#remove rna info.
orig = torch.load('pyg_hetlist_predf3_trin25_co.pt')
for graph in orig:
    del graph['enhancer','foldchange','rna']
    del graph['rna','rev_foldchange','enhancer']
    del graph['enhancer','pvalue','rna']
    del graph['rna','rev_pvalue','enhancer']
    del graph['promoter','foldchange','rna']
    del graph['rna','rev_foldchange','promoter']
    del graph['promoter','pvalue','rna']
    del graph['rna','rev_pvalue','promoter']
torch.save(orig,'pyg_hetlist_predf3_trin25norna_co.pt')


# In[17]:


orig = torch.load('pyg_hetlist_predf3_trin50_co.pt')
for graph in orig:
    del graph['enhancer','foldchange','rna']
    del graph['rna','rev_foldchange','enhancer']
    del graph['enhancer','pvalue','rna']
    del graph['rna','rev_pvalue','enhancer']
    del graph['promoter','foldchange','rna']
    del graph['rna','rev_foldchange','promoter']
    del graph['promoter','pvalue','rna']
    del graph['rna','rev_pvalue','promoter']
torch.save(orig,'pyg_hetlist_predf3_trin50norna_co.pt')


# In[ ]:


orig = torch.load('pyg_hetlist_predf3_bin_co.pt')
for graph in orig:
    del graph['enhancer','foldchange','rna']
    del graph['rna','rev_foldchange','enhancer']
    del graph['enhancer','pvalue','rna']
    del graph['rna','rev_pvalue','enhancer']
    del graph['promoter','foldchange','rna']
    del graph['rna','rev_foldchange','promoter']
    del graph['promoter','pvalue','rna']
    del graph['rna','rev_pvalue','promoter']
torch.save(orig,'pyg_hetlist_predf2_binnorna_co.pt')


# In[ ]:


#mask checking :(

#htbtrainmask is separated into blocks of 100 to minimize shared enhancers appearing in both the training and test sets
gblocks = list(range(int(len(fulllist)/100)+1))
testblocks = random.sample(gblocks,int(len(gblocks)/5))
bmask = [False]*len(fulllist)
for block in testblocks:
    for idx in range(100*block,100*block+100):
        bmask[idx] = True
with open('newbtrainmask.pkl','wb') as mfile:
    pkl.dump(bmask,mfile)


# ## LOAD IN FILES

# lib = 'bl'
# lib = 'ln'
lib = 'co'

#ORIGINAL FILTERED LIBS
rna_lib = sc.read('corna_filtered.h5ad')
atac_lib = sc.read('coatac_filtered.h5ad')
pert_lib = pkl.load(open('copert_filtered.pkl','rb'))
ccc = pkl.load(open('coccc.pkl','rb'))
#nc = 1426, ghmt = 626

# [tss_ind,rna_ind,[atac_inds]] for enhol
# [tss_ind,rna_ind,atac_ind] for promol
# load in new prom_overlaps and promenh_overlaps with 1000bp offset
promol = pkl.load(open('co1000ce_prom_overlaps.pkl','rb'))
promenhol = pkl.load(open('co1000ce_promenh_overlaps.pkl','rb'))
enhol = pkl.load(open('co1000ce_enh_overlaps.pkl','rb'))

# LOAD IN CE LIBS
ce_rna = pkl.load(open('conce_rna.pkl','rb'))                   #ce_rna but CPM normalized
zce_rna = pkl.load(open('cozce_rna.pkl','rb'))                  #nce_rna but z-scored
ce_atac = pkl.load(open('conce_atac.pkl','rb'))                 #ce_atac but binarized and then CPM normalized
ce_tss = pkl.load(open('coce_tss.pkl','rb'))

# LOAD IN FULL STATS FILES,    each element is in form [pv,qv,fc]
unflat_te = pkl.load(open('total_co_perturbationenhancer_stats.pkl','rb'))
# te = [[stats for pert in enh for stats in pert] for enh in unflat_te] #flatten by one dimension
te = unflat_te
tp = pkl.load(open('total_co_perturbationpromoter_stats.pkl','rb'))
tr = pkl.load(open('total_co_perturbationrna_stats.pkl','rb'))
ep = pkl.load(open('total_co_enhancerpromoter_stats.pkl','rb'))
er = pkl.load(open('total_co_enhancerrna_stats.pkl','rb'))
pr = pkl.load(open('total_co_promoterrna_stats.pkl','rb'))

#Misc. auxiliary structures
perts,tf_list = ['NC'],['G','H','M','T']
for i in range(1,len(tf_list)+1):
    perts+=list(combinations(tf_list,i))
perts = [''.join(elem) for elem in perts]
pert_nums = list(range(len(perts)))

all_tss = list(ce_tss.loc[:,'gid'])
gl = []
for entry in promol:
    tss_ind = entry[0]
    gene = all_tss[tss_ind]
    gl.append(gene)
len(gl)

# #optional GO structures
# with open('cardiac_gl.txt') as c_file:
#     c_lines = c_file.readlines()
# for c_line in range(len(c_lines)):
#     fixed_name = c_lines[c_line].split('\n')[0]
#     if '/iso:' in fixed_name:
#         fixed_name = fixed_name.split('/iso:')[0]+fixed_name.split('/iso:')[1]
#     c_lines[c_line] = fixed_name
    
# with open('fibroblast_gl.txt') as f_file:
#     f_lines = f_file.readlines()
# for f_line in range(len(f_lines)):
#     fixed_name = f_lines[f_line].split('\n')[0]
#     if '/iso:' in fixed_name:
#         fixed_name = fixed_name.split('/iso:')[0]+fixed_name.split('/iso:')[1]
#     f_lines[f_line] = fixed_name
    
# cgl = c_lines
# fibrogl = f_lines

#full graph list
#remember!!!    new_fc=5*np.log2(raw_fc+2**-5)    new_pv=np.log10(raw_pv+10**-15)
fgl = pkl.load(open('debug?_full_graph_list_'+lib+'.pkl','rb'))
# databatch = pkl.load(open('pyg_databatch_'+lib+'.pkl','rb'))

datalist = pkl.load(open('pyg_datalist_'+lib+'.pkl','rb'))
# hetlist = pkl.load(open('pyg_hetlist_'+lib+'.pkl','rb'))
# datalist = pkl.load(open('pyg_datalist_'+lib+'_parperteprdiseq.pkl','rb'))


# In[3]:


#load in various features for heterodata construction
adata_6mer = sc.read('6mer_adata_'+lib+'.h5ad') #adata of all atac 6mer vectors
diseq_l = pkl.load(open('diseq_list_'+lib+'.pkl','rb')) #counts tensor of all nodes distances to promoter (-1 for non-Enh nodes) plus 6mer vectors (set to null vectors for non-ATAC nodes)
# asogene_l = pkl.load(open('asogene_list_'+lib+'.pkl','rb')) # tesnor of number of number of genes associated with each ATAC node in each graph
# gotensor = pkl.load(open('pyg_gotensor_'+lib+'.pkl','rb')) #binary tensor of all GO terms associated with all graphs
# setgoid = pkl.load(open('pyg_setgoid_'+lib+'.pkl','rb')) #list of all trimmed GO IDs with matching index to gotensor (see go_getter() function)
tf_te = pkl.load(open('component_tfs_tensor.pkl','rb')) #binary tensor encoding component TFs for all perturbation combos


# In[48]:


# x=[10**0,10**0.5,10**1,10**1.5,10**2,10**2.5,10**3,10**3.5]
x=[10**(exp/10) for exp in range(0,34)]
y = [np.sum(adata_6mer.uns['pca']['variance_ratio'][0:int(x_point)]) for x_point in x]
logx = range(0,34)
plt.scatter(logx,y)


# ## DATA PREP and UTILS

#takes int (element in setgoid) and converts it to corresponding go ID string
def go_getter(go_integer):
    num_digits = len(str(go_integer))
    goid_str = 'GO:'+'0'*(7-num_digits)+str(go_integer)
    return goid_str

#takes graph (pyg het object), edge or node type name, and position (0=origin node, 1=edge type, 2=destination node)
#and returns list of edge dictionaries
def get_edges(graph,key,pos):
    dict_keys = [edge_type for edge_type in graph.metadata()[1] if edge_type[pos] == key]
    return [graph[key] for key in dict_keys]

#optional: add full gotensor as graph level features, along with select go_bool rows
for graph_id,graph in enumerate(datalist):
#     graph.card = cardiac_bool[graph_id]
#     graph.fibr = fibroblast_bool[graph_id]
    graph.gote = gotensor[graph_id]

#classify genes by DEG status (up,down,null) between NC and GHMT
np_cells = [cell for cell in ccc[1][0] if cell not in ccc[1][15]] + [cell for cell in ccc[1][15] if cell not in ccc[1][0]]
np_rna = ce_rna[np_cells]
np_rna.obs['treatment'] = ['NC'*(cell in ccc[1][0])+'GHMT'*(cell in ccc[1][15]) for cell in np_rna.obs_names]

sc.tl.rank_genes_groups(np_rna, 'treatment', method='wilcoxon')
# sc.pl.rank_genes_groups(np_rna,fontsize=15)
cutoff = len([np_rna.uns['rank_genes_groups']['logfoldchanges'][row][0] for row in range(len(np_rna.uns['rank_genes_groups']['logfoldchanges'])) if not (np_rna.uns['rank_genes_groups']['logfoldchanges'][row][0] < 2)])
updeg,downdeg = [],[]
for gene_pair in np_rna.uns['rank_genes_groups']['names'][0:cutoff]:
    updeg.append(gene_pair[0]),downdeg.append(gene_pair[1])
nodeg = [gene for gene in gl if gene not in updeg and gene not in downdeg]

len(updeg),len(downdeg),len(nodeg)

#convert to true heterodata
hetlist = []
for graph_id,data in enumerate(datalist):
    het = pyg.data.HeteroData()
    
    het['perturbation'].x = tf_te
    het['enhancer'].x = diseq_l[graph_id][16:-2,:]
    het['promoter'].x = diseq_l[graph_id][-2:-1,1:]
#     het['rna'].x = gotensor[graph_id] #this one kinda sucks, may want to PCA it before use?
    this_gene = ce_rna.var_names[promol[int(data.y[0])][1]]
    if this_gene in updeg:
        deg_status = 0
    elif this_gene in nodeg:
        deg_status = 1
    else:
        deg_status = 2
    het['rna'].x = Tensor([[deg_status]])
    
    #for indexing edges
    e_num, t_num = len(diseq_l[graph_id][16:-2,:]), len(tf_te)
    px_offset = t_num*(e_num+2)
    amat = data.edge_index.clone().detach()
    #t->e
    te_edges = torch.cat([amat[:,(e_num+2)*pert:(e_num+2)*pert+e_num] for pert in range(t_num)],dim=1)
    te_edges[1,:]-=t_num #reindex e node designators
    te_fcs = torch.cat([data.fc[(e_num+2)*pert:(e_num+2)*pert+e_num] for pert in range(t_num)],dim=0).to(torch.float32)
    te_fcs[te_fcs>100] = 100 #prob don't need
    te_pvs = torch.cat([data.pv[(e_num+2)*pert:(e_num+2)*pert+e_num] for pert in range(t_num)],dim=0).to(torch.float32)
    het['perturbation','foldchange','enhancer'].edge_index, het['perturbation','pvalue','enhancer'].edge_index = te_edges,te_edges
    het['perturbation','foldchange','enhancer'].edge_attr, het['perturbation','pvalue','enhancer'].edge_attr = te_fcs, te_pvs
    #t->p
    tp_edges = torch.cat([amat[:,e_num+(e_num+2)*pert:e_num+1+(e_num+2)*pert] for pert in range(t_num)],dim=1)
    tp_edges[1,:]-=(t_num+e_num) #reindex p node designators
    tp_fcs = torch.cat([data.fc[e_num+(e_num+2)*pert:e_num+1+(e_num+2)*pert] for pert in range(t_num)],dim=0).to(torch.float32)
    tp_fcs[tp_fcs>100] = 100 #prob don't need
    tp_pvs = torch.cat([data.pv[e_num+(e_num+2)*pert:e_num+1+(e_num+2)*pert] for pert in range(t_num)],dim=0).to(torch.float32)
    het['perturbation','foldchange','promoter'].edge_index, het['perturbation','pvalue','promoter'].edge_index = tp_edges, tp_edges
    het['perturbation','foldchange','promoter'].edge_attr, het['perturbation','pvalue','promoter'].edge_attr = tp_fcs, tp_pvs
    #t->r
    tr_edges = torch.cat([amat[:,e_num+1+(e_num+2)*t_num:e_num+(e_num+2)*t_num+2] for t_num in range(len(tf_te))],dim=1)
    tr_edges[1,:]-=(t_num+e_num+1) #reindex r node designators
    tr_fcs = torch.cat([data.fc[e_num+1+(e_num+2)*t_num:e_num+(e_num+2)*t_num+2] for t_num in range(len(tf_te))],dim=0).to(torch.float32)
    tr_fcs[tr_fcs>100] = 100 #prob don't need
    tr_pvs = torch.cat([data.pv[e_num+1+(e_num+2)*t_num:e_num+(e_num+2)*t_num+2] for t_num in range(len(tf_te))],dim=0).to(torch.float32)
    het['perturbation','foldchange','rna'].edge_index, het['perturbation','pvalue','rna'].edge_index = tr_edges, tr_edges
    het['perturbation','foldchange','rna'].edge_attr, het['perturbation','pvalue','rna'].edge_attr = tr_fcs, tr_pvs
    #e->p
    ep_edges = torch.cat([amat[:,px_offset+enh*2:px_offset+enh*2+1] for enh in range(e_num)],dim=1)
    ep_edges[0,:]-=t_num #reindex e node designators
    ep_edges[1,:]-=(t_num+e_num) #reindex p node designators
    ep_fcs = torch.cat([data.fc[px_offset+enh*2:px_offset+enh*2+1] for enh in range(e_num)],dim=0).to(torch.float32)
    ep_fcs[ep_fcs>100] = 100 #prob don't need
    ep_pvs = torch.cat([data.pv[px_offset+enh*2:px_offset+enh*2+1] for enh in range(e_num)],dim=0).to(torch.float32)
    het['enhancer','foldchange','promoter'].edge_index, het['enhancer','pvalue','promoter'].edge_index = ep_edges, ep_edges
    het['enhancer','foldchange','promoter'].edge_attr, het['enhancer','pvalue','promoter'].edge_attr = ep_fcs, ep_pvs
    #e->r
    er_edges = torch.cat([amat[:,px_offset+enh*2+1:px_offset+enh*2+2] for enh in range(e_num)],dim=1)
    er_edges[0,:]-=t_num #reindex e node designators
    er_edges[1,:]-=(t_num+e_num+1) #reindex r node designators
    er_fcs = torch.cat([data.fc[px_offset+enh*2+1:px_offset+enh*2+2] for enh in range(e_num)],dim=0).to(torch.float32)
    if Tensor.any(er_fcs>100):
        print('ALERT!',er_fcs)
        er_fcs[er_fcs>100] = 100 #prob don't need
        print('post correction?:',er_fcs)
    er_fcs[er_fcs>100] = 100
    er_pvs = torch.cat([data.pv[px_offset+enh*2+1:px_offset+enh*2+2] for enh in range(e_num)],dim=0).to(torch.float32)
    het['enhancer','foldchange','rna'].edge_index, het['enhancer','pvalue','rna'].edge_index = er_edges, er_edges
    het['enhancer','foldchange','rna'].edge_attr, het['enhancer','pvalue','rna'].edge_attr = er_fcs, er_pvs
    #p->r
    pr_edges = amat[:,-1:]
    pr_edges[0,:]-=(t_num+e_num) #reindex e node designators
    pr_edges[1,:]-=(t_num+e_num+1) #reindex p node designators
    pr_fcs = data.fc[-1:].to(torch.float32)
    pr_fcs[pr_fcs>100] = 100 #prob don't need
    pr_pvs = data.pv[-1:].to(torch.float32)
    het['promoter','foldchange','rna'].edge_index, het['promoter','pvalue','rna'].edge_index = pr_edges, pr_edges
    het['promoter','foldchange','rna'].edge_attr, het['promoter','pvalue','rna'].edge_attr = pr_fcs, pr_pvs
    
    hetlist.append(het)


# ## Remove 6mer/indirect edges from predtfr hetlist

hetlist = torch.load('pyg_hetlist_predtfr_trin_'+lib+'.pt')

lib = 'co'
hetlist = torch.load('pyg_hetlist500pca_predtfr_'+lib+'.pt')


# In[42]:


for het in hetlist:
    het['enhancer']['x'] = het['enhancer']['x'][:,0:1]
torch.save(hetlist,'pyg_hetlist500pca_predtfr_noeseq_'+lib+'.pt')


# In[48]:


for het in hetlist:
    het['enhancer']['x'] = het['enhancer']['x'][:,0:1]
    
    del het[('perturbation','foldchange','promoter')]
    del het[('perturbation','pvalue','promoter')]
    del het[('perturbation','foldchange','rna')]
    del het[('perturbation','pvalue','rna')]
    del het[('enhancer','foldchange','rna')]
    del het[('enhancer','pvalue','rna')]

    del het[('promoter','rev_foldchange','perturbation')]
    del het[('promoter','rev_pvalue','perturbation')]
    del het[('rna','rev_foldchange','perturbation')]
    del het[('rna','rev_pvalue','perturbation')]
    del het[('rna','rev_foldchange','enhancer')]
    del het[('rna','rev_pvalue','enhancer')]

torch.save(hetlist,'pyg_hetlist500pca_predtfr_noeseq_noindir_'+lib+'.pt')


# In[50]:


for het in hetlist:
    del het[('perturbation','foldchange','promoter')]
    del het[('perturbation','pvalue','promoter')]
    del het[('perturbation','foldchange','rna')]
    del het[('perturbation','pvalue','rna')]
    del het[('enhancer','foldchange','rna')]
    del het[('enhancer','pvalue','rna')]

    del het[('promoter','rev_foldchange','perturbation')]
    del het[('promoter','rev_pvalue','perturbation')]
    del het[('rna','rev_foldchange','perturbation')]
    del het[('rna','rev_pvalue','perturbation')]
    del het[('rna','rev_foldchange','enhancer')]
    del het[('rna','rev_pvalue','enhancer')]
torch.save(hetlist,'pyg_hetlist500pca_predtfr_noindir_'+lib+'.pt')


# In[56]:


hetlist = torch.load('pyg_hetlist_predtfr_'+lib+'.pt')


# In[54]:


for het in hetlist:
    het['enhancer']['x'] = het['enhancer']['x'][:,0:1]
torch.save(hetlist,'pyg_hetlist_predtfr_noeseq_'+lib+'.pt')


# In[55]:


for het in hetlist:
    het['enhancer']['x'] = het['enhancer']['x'][:,0:1]
    
    del het[('perturbation','foldchange','promoter')]
    del het[('perturbation','pvalue','promoter')]
    del het[('perturbation','foldchange','rna')]
    del het[('perturbation','pvalue','rna')]
    del het[('enhancer','foldchange','rna')]
    del het[('enhancer','pvalue','rna')]

    del het[('promoter','rev_foldchange','perturbation')]
    del het[('promoter','rev_pvalue','perturbation')]
    del het[('rna','rev_foldchange','perturbation')]
    del het[('rna','rev_pvalue','perturbation')]
    del het[('rna','rev_foldchange','enhancer')]
    del het[('rna','rev_pvalue','enhancer')]

torch.save(hetlist,'pyg_hetlist_predtfr_noeseq_noindir_'+lib+'.pt')


# In[57]:


for het in hetlist:
    del het[('perturbation','foldchange','promoter')]
    del het[('perturbation','pvalue','promoter')]
    del het[('perturbation','foldchange','rna')]
    del het[('perturbation','pvalue','rna')]
    del het[('enhancer','foldchange','rna')]
    del het[('enhancer','pvalue','rna')]

    del het[('promoter','rev_foldchange','perturbation')]
    del het[('promoter','rev_pvalue','perturbation')]
    del het[('rna','rev_foldchange','perturbation')]
    del het[('rna','rev_pvalue','perturbation')]
    del het[('rna','rev_foldchange','enhancer')]
    del het[('rna','rev_pvalue','enhancer')]
torch.save(hetlist,'pyg_hetlist_predtfr_noindir_'+lib+'.pt')


# ## Randomize each feature type across het lists

# In[42]:


lib = 'co'
#load in h1 hetlist, contains true heterodata, undirected, diseq, up/down/nodeg, GHMT binary matrix, and cleaned and defined FC edge weights
# hetmod = 'hetlist'
#load in h2 hetlist, contains true heterodata, undirected, diseq500pca, up/down/nodeg, GHMT binary matrix, and cleaned and defined FC edge weights
# hetmod = 'hetlist500pca'
#load in h2 hetlist, contains true heterodata, undirected, diseq500pca, top 10% up/downdeg, GHMT binary matrix, and cleaned and defined FC edge weights
hetmod = 'hetlist500pca_sw'

hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[35]:


#randomize pert nodes
#every graph has one of each combo by definition, so we can merely shuffle which node has which combo.
for het in hetlist:
    feat = het['perturbation'].x
    het['perturbation'].x=feat[torch.randperm(feat.size()[0])]
torch.save(hetlist,'pyg_'+hetmod+'_pshuffle_'+lib+'.pt')


# In[10]:


#randomize enh dists
#compile pool of all enh distances
enhdist_pool = []
for het in hetlist:
    thishet_enhdist = het['enhancer'].x[:,0].clone()
    for elem in thishet_enhdist:
        enhdist_pool.append(float(elem))
#assign enhdist feat randomly to each node from total pool
for het in hetlist:
    for which_enh in range(het['enhancer'].x.size()[0]):
        het['enhancer'].x[which_enh,0] = enhdist_pool.pop(int(torch.rand(1)*len(enhdist_pool)))
torch.save(hetlist,'pyg_'+hetmod+'_edshuffle_'+lib+'.pt')


# In[39]:


#randomize enh 6mers
#compile pool of all enh 6mers
enh6mer_pool = []
for het in hetlist:
    thishet_enh6mer = het['enhancer'].x[:,1:].clone()
    for elem in thishet_enh6mer:
        enh6mer_pool.append(elem)
#assign enh6mer feat randomly to each node from total pool
for het in hetlist:
    for which_enh in range(het['enhancer'].x.size()[0]):
        ri = int(torch.rand(1)*len(enh6mer_pool))
        het['enhancer'].x[which_enh,1:] = enh6mer_pool.pop(ri)
torch.save(hetlist,'pyg_'+hetmod+'_e6shuffle_'+lib+'.pt')


# In[41]:


#randomize prom 6mers
#compile pool of all prom 6mers
prom6mer_pool = []
for het in hetlist:
    thishet_prom6mer = het['promoter'].x[:,:].clone()
    for elem in thishet_prom6mer:
        prom6mer_pool.append(elem)
#assign prom6mer feat randomly to each node from total pool
for het in hetlist:
    for which_enh in range(het['promoter'].x.size()[0]):
        het['promoter'].x[which_enh,:] = prom6mer_pool.pop(int(torch.rand(1)*len(prom6mer_pool)))
torch.save(hetlist,'pyg_'+hetmod+'_p6shuffle_'+lib+'.pt')


# In[77]:


#randomize rna degstats   NOTA BENE: using this alone will NOT work for models based on prediction of rna degstats



# In[2]:


lib = 'co'
hetmod = 'hetlist500pca'
hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize pert->enh FCs
#compile pool of all pert->enh FCs
tfe_pool = []
for het in hetlist:
    thishet_tfe = het['perturbation','foldchange','enhancer'].edge_attr.clone()
    for elem in thishet_tfe:
        tfe_pool.append(elem)
#assign pfe feat randomly to each edge from total pool
for het in hetlist:
    for which_tfe in range(het['perturbation','foldchange','enhancer'].edge_attr.size()[0]):
        het['perturbation','foldchange','enhancer'].edge_attr[which_tfe] = tfe_pool.pop(int(torch.rand(1)*len(tfe_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_tfeshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize pert->prom FCs
#compile pool of all pert->prom FCs
tfp_pool = []
for het in hetlist:
    thishet_tfp = het['perturbation','foldchange','promoter'].edge_attr.clone()
    for elem in thishet_tfp:
        tfp_pool.append(elem)
#assign pfe feat randomly to each edge from total pool
for het in hetlist:
    for which_tfp in range(het['perturbation','foldchange','promoter'].edge_attr.size()[0]):
        het['perturbation','foldchange','promoter'].edge_attr[which_tfp] = tfp_pool.pop(int(torch.rand(1)*len(tfp_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_tfpshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize pert->rna FCs
#compile pool of all pert->rna FCs
tfr_pool = []
for het in hetlist:
    thishet_tfr = het['perturbation','foldchange','rna'].edge_attr.clone()
    for elem in thishet_tfr:
        tfr_pool.append(elem)
#assign tfr feat randomly to each edge from total pool
for het in hetlist:
    for which_tfr in range(het['perturbation','foldchange','rna'].edge_attr.size()[0]):
        het['perturbation','foldchange','rna'].edge_attr[which_tfr] = tfr_pool.pop(int(torch.rand(1)*len(tfr_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_tfrshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize enh->prom FCs
#compile pool of all enh->prom FCs
efp_pool = []
for het in hetlist:
    thishet_efp = het['enhancer','foldchange','promoter'].edge_attr.clone()
    for elem in thishet_efp:
        efp_pool.append(elem)
#assign efp feat randomly to each edge from total pool
for het in hetlist:
    for which_efp in range(het['enhancer','foldchange','promoter'].edge_attr.size()[0]):
        het['enhancer','foldchange','promoter'].edge_attr[which_efp] = efp_pool.pop(int(torch.rand(1)*len(efp_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_efpshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize enh->rna FCs
#compile pool of all enh->rna FCs
efr_pool = []
for het in hetlist:
    thishet_efr = het['enhancer','foldchange','rna'].edge_attr.clone()
    for elem in thishet_efr:
        efr_pool.append(elem)
#assign efr feat randomly to each edge from total pool
for het in hetlist:
    for which_efr in range(het['enhancer','foldchange','rna'].edge_attr.size()[0]):
        het['enhancer','foldchange','rna'].edge_attr[which_efr] = efr_pool.pop(int(torch.rand(1)*len(efr_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_efrshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize prom->rna FCs
#compile pool of all prom->rna FCs
pfr_pool = []
for het in hetlist:
    thishet_pfr = het['promoter','foldchange','rna'].edge_attr.clone()
    for elem in thishet_pfr:
        pfr_pool.append(elem)
#assign pfr feat randomly to each edge from total pool
for het in hetlist:
    for which_pfr in range(het['promoter','foldchange','rna'].edge_attr.size()[0]):
        het['promoter','foldchange','rna'].edge_attr[which_pfr] = pfr_pool.pop(int(torch.rand(1)*len(pfr_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_pfrshuffle_'+lib+'.pt')


# In[3]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize pert->enh PVs
#compile pool of all pert->enh PVs
tpe_pool = []
for het in hetlist:
    thishet_tpe = het['perturbation','pvalue','enhancer'].edge_attr.clone()
    for elem in thishet_tpe:
        tpe_pool.append(elem)
#assign tpe feat randomly to each edge from total pool
for het in hetlist:
    for which_tpe in range(het['perturbation','pvalue','enhancer'].edge_attr.size()[0]):
        het['perturbation','pvalue','enhancer'].edge_attr[which_tpe] = tpe_pool.pop(int(torch.rand(1)*len(tpe_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_tpeshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize pert->prom PVs
#compile pool of all pert->prom PVs
tpp_pool = []
for het in hetlist:
    thishet_tpp = het['perturbation','pvalue','promoter'].edge_attr.clone()
    for elem in thishet_tpp:
        tpp_pool.append(elem)
#assign tpp feat randomly to each edge from total pool
for het in hetlist:
    for which_tpp in range(het['perturbation','pvalue','promoter'].edge_attr.size()[0]):
        het['perturbation','pvalue','promoter'].edge_attr[which_tpp] = tpp_pool.pop(int(torch.rand(1)*len(tpp_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_tppshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize pert->rna PVs
#compile pool of all pert->rna PVs
tpr_pool = []
for het in hetlist:
    thishet_tpr = het['perturbation','pvalue','rna'].edge_attr.clone()
    for elem in thishet_tpr:
        tpr_pool.append(elem)
#assign tpr feat randomly to each edge from total pool
for het in hetlist:
    for which_tpr in range(het['perturbation','pvalue','rna'].edge_attr.size()[0]):
        het['perturbation','pvalue','rna'].edge_attr[which_tpr] = tpr_pool.pop(int(torch.rand(1)*len(tpr_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_tprshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')

#randomize enh->prom PVs
#compile pool of all enh->prom PVs
epp_pool = []
for het in hetlist:
    thishet_epp = het['enhancer','pvalue','promoter'].edge_attr.clone()
    for elem in thishet_epp:
        epp_pool.append(elem)
#assign epp feat randomly to each edge from total pool
for het in hetlist:
    for which_epp in range(het['enhancer','pvalue','promoter'].edge_attr.size()[0]):
        het['enhancer','pvalue','promoter'].edge_attr[which_epp] = epp_pool.pop(int(torch.rand(1)*len(epp_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_eppshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize enh->rna PVs
#compile pool of all enh->rna PVs
epr_pool = []
for het in hetlist:
    thishet_epr = het['enhancer','pvalue','rna'].edge_attr.clone()
    for elem in thishet_epr:
        epr_pool.append(elem)
#assign epr feat randomly to each edge from total pool
for het in hetlist:
    for which_epr in range(het['enhancer','pvalue','rna'].edge_attr.size()[0]):
        het['enhancer','pvalue','rna'].edge_attr[which_epr] = epr_pool.pop(int(torch.rand(1)*len(epr_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_eprshuffle_'+lib+'.pt')


# In[ ]:


hetlist = torch.load('pyg_'+hetmod+'_'+lib+'.pt')


# In[ ]:


#randomize prom->rna PVs
#compile pool of all prom->rna PVs
ppr_pool = []
for het in hetlist:
    thishet_ppr = het['promoter','pvalue','rna'].edge_attr.clone()
    for elem in thishet_ppr:
        ppr_pool.append(elem)
#assign ppr feat randomly to each edge from total pool
for het in hetlist:
    for which_ppr in range(het['promoter','pvalue','rna'].edge_attr.size()[0]):
        het['promoter','pvalue','rna'].edge_attr[which_ppr] = ppr_pool.pop(int(torch.rand(1)*len(ppr_pool)))
for het in hetlist:
    del(het['enhancer','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','perturbation'])
    del(het['rna','rev_foldchange','perturbation'])
    del(het['promoter','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','enhancer'])
    del(het['rna','rev_foldchange','promoter'])
    del(het['enhancer','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','perturbation'])
    del(het['rna','rev_pvalue','perturbation'])
    del(het['promoter','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','enhancer'])
    del(het['rna','rev_pvalue','promoter'])
    T.ToUndirected()(het)
torch.save(hetlist,'pyg_'+hetmod+'_pprshuffle_'+lib+'.pt')


# In[77]:


#randomize adjacency matrices?


# ## END Randomize rach feature type across het lists

# ## Actual gnn construction

# In[78]:


# make heterographs undirected (needed for most heteromodel implementations?)
hetlist = [T.ToUndirected()(het) for het in hetlist]
# hetlist = [T.AddSelfLoops()(het) for het in hetlist]


# In[79]:


#NEW heterodata prep

fulllist,trainlist,testlist = hetlist,hetlist[:int(len(datalist)/2)],hetlist[int(len(datalist)/2):]

#transform fc to belong to {R>=0}     note that current_fc=5*np.log2(raw_fc+2**-5)
for entry in fulllist:
    el = get_edges(entry,'foldchange',1)
    for et in el:
        et.edge_attr = 2**(et.edge_attr/5)-2**-5
        et.edge_attr[et.edge_attr>100] = 100
for entry in trainlist:
    el = get_edges(entry,'foldchange',1)
    for et in el:
        et.edge_attr = 2**(et.edge_attr/5)-2**-5
        et.edge_attr[et.edge_attr>100] = 100
for entry in testlist:
    el = get_edges(entry,'foldchange',1)
    for et in el:
        et.edge_attr = 2**(et.edge_attr/5)-2**-5
        et.edge_attr[et.edge_attr>100] = 100

#loaderize datasets
fullloader = pyg.loader.DataLoader(fulllist,shuffle=True)
trainloader = pyg.loader.DataLoader(trainlist,shuffle=True)
testloader = pyg.loader.DataLoader(testlist,shuffle=True)


# ## hetero GraphConv model

# In[80]:


class GC(torch.nn.Module):
    def __init__(self):
        intra_channels,out_channels = 64,3
        super().__init__()
#         self.conv1 = SAGEConv(-1, intra_channels,add_self_loops=False)
#         self.conv2 = SAGEConv(intra_channels, out_channels,add_self_loops=False)
#         self.conv1 = SAGEConv(-1, intra_channels)
#         self.conv2 = SAGEConv(intra_channels, out_channels)
        self.conv1 = GraphConv(-1, intra_channels)
        self.conv2 = GraphConv(intra_channels, out_channels)
#         self.double()

    def forward(self, data,data_x_dict, data_edge_index_dict, data_edge_attr_dict):
        x, edge_index,edge_weight = data_x_dict, data_edge_index_dict, data_edge_attr_dict
        open_conv1 = x.clone().detach()
        x = self.conv1(x,edge_index,edge_weight)
        conv1_relu = x.clone().detach()
        x = F.relu(x)
        relu_dropout = x.clone().detach()
        x = F.dropout(x,training=self.training)
        dropout_conv2 = x.clone().detach()
        x = self.conv2(x,edge_index,edge_weight)

#         return F.log_softmax(x, dim=1)
        return open_conv1,conv1_relu,relu_dropout,dropout_conv2,x,F.softmax(x, dim=1)


# In[81]:


#...but not on my data and to_hetero_with_bases call!!!
data = fulllist[0]
model = GC()
# model = to_hetero_with_bases(model, fulllist[0].metadata(),num_bases=12)
model = to_hetero(model, data.metadata(), aggr='sum')

# print([data.x_dict[entry].dtype for entry in .x_dict])
# print([data.edge_index_dict[entry].dtype for entry in data.edge_index_dict])
# print([data.edge_attr_dict[entry].dtype for entry in data.edge_attr_dict])

with torch.no_grad():  # Initialize lazy modules.
    print(data.num_nodes)
    out = model(data,data.x_dict,data.edge_index_dict, data.edge_attr_dict)


# In[ ]:


# model = m1
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
init = fulllist[0]
model(init,init.x_dict,init.edge_index_dict,init.edge_attr_dict)
model.train()
start = dtime()

# target_feat = data.card
# target_feat = data.x[:,1].long()
# target_feat = data.x[:,1]

for epoch in range(0,10):
    for data in trainloader:
#         target_feat = data.x[:,1].long()
        optimizer.zero_grad()
        open_conv1,conv1_relu,relu_dropout,dropout_conv2,rawx,out = model(data,data.x_dict,data.edge_index_dict, data.edge_attr_dict)
        if torch.isnan(torch.squeeze(open_conv1['rna'][0][0])):
            print(out['rna'].argmax(dim=1),'WE HAVE A NAN <open_conv1> HERE, ABORT!')
        if torch.isnan(torch.squeeze(conv1_relu['rna'][0][0])):
            print(out['rna'].argmax(dim=1),'WE HAVE A NAN <conv1_relu> HERE, ABORT!')
            print(data.edge_attr_dict)
        if torch.isnan(torch.squeeze(relu_dropout['rna'][0][0])):
            print(out['rna'].argmax(dim=1),'WE HAVE A NAN <relu_dropout> HERE, ABORT!')
        if torch.isnan(torch.squeeze(dropout_conv2['rna'][0][0])):
            print(out['rna'].argmax(dim=1),'WE HAVE A NAN <dropout_conv2> HERE, ABORT!')
        if torch.isnan(torch.squeeze(rawx['rna'][0][0])):
            print(out['rna'].argmax(dim=1),'WE HAVE A NAN <conv2_softmax> HERE, ABORT!')
        if torch.isnan(torch.squeeze(out['rna'][0][0])):
            print(out['rna'].argmax(dim=1),'WE HAVE A NAN <OUT> HERE, ABORT!')
            break
#         print(torch.squeeze(out['rna']),torch.squeeze(data.x_dict['rna']).long())
#         print('RAW:',torch.squeeze(rawx['rna']),torch.squeeze(data.x_dict['rna']).long())
#         print('RAWRAW:',rawx['rna'].shape)

        
        
        loss = F.nll_loss(torch.squeeze(out['rna']),torch.squeeze(data.x_dict['rna']).long())
        loss.backward()
        optimizer.step()
    if epoch%1==0:
        print(dtime()-start)
        with open('het1_iter'+str(epoch)+'.pkl', 'wb') as dl_file:
            pkl.dump(model,dl_file)
print(dtime()-start)


# In[16]:


m0 = pkl.load(open('het1_iter0.pkl', 'rb'))
m1 = pkl.load(open('het1_iter1.pkl', 'rb'))
m10 = pkl.load(open('het1_iter10.pkl', 'rb'))
m20 = pkl.load(open('het1_iter20.pkl', 'rb'))
m30 = pkl.load(open('het1_iter30.pkl', 'rb'))
m40 = pkl.load(open('het1_iter40.pkl', 'rb'))
m49 = pkl.load(open('het1_iter49.pkl', 'rb'))


# In[19]:


model = m1

subloader = pyg.loader.DataLoader(testlist[:1000],shuffle=True)
acl = []
model.eval()
for data in subloader:
    out = model(data,data.x_dict,data.edge_index_dict, data.edge_attr_dict)
    pred = out[-1]['rna'].argmax(dim=1)
    correct = torch.squeeze(data.x_dict['rna']).long()
#     print(pred,correct)
    if pred != 0:
        print(pred)
    acl.append(pred==correct)
print(np.sum([int(entry) for entry in acl])/len(acl))


# In[ ]:


model = m14
acl = []
model.eval()
for data in subloader:
    out = model(data,data.x_dict,data.edge_index_dict, data.edge_attr_dict)
    pred = out['rna'].argmax(dim=1)
    correct = torch.squeeze(data.x_dict['rna']).long()
    acl.append(pred==correct)
    if torch.isnan(torch.squeeze(out['rna'][0][0])):
        print('OPE')
print(np.sum([int(entry) for entry in acl])/len(acl))


# In[60]:


subloader = pyg.loader.DataLoader(testlist[:10],shuffle=True)


# ## edge weight predictor NN architecture

# In[21]:


class GCNpool_ew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(-1, 32)
        self.pool1 = pool.SAGPooling(32,4,GCNConv)
        self.conv2 = GCNConv(32, 19)

    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index, data.fc.float()
        x = self.conv1(x,edge_index,edge_weight)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)
#         x = self.conv2(x,edge_index,edge_weight)     #old conv2 call

#         #pool returns are [x,edge_index,edge_attr,batch,perm (node indexes i think),score]
        px,pa,pw,_,pn,_ = self.pool1(x=x,edge_index=edge_index,edge_attr=edge_weight)
        x = self.conv2(px,pa,pw)
        real_pertepr = data.x[pn,1]

        return F.log_softmax(x, dim=1), real_pertepr


# In[20]:


#run GCNpool model
model = GCNpool_ew()
loader = trainloader
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model(fulllist[0])

model.train()
start = dtime()

# target_feat = data.card
# target_feat = data.x[:,1].long()
# target_feat = data.x[:,1]

for epoch in range(2):
    for data in trainloader:
#         target_feat = data.x[:,1].long()
        optimizer.zero_grad()
        out,real_pertepr = model(data)
        loss = F.nll_loss(out,real_pertepr.long())
        loss.backward()
        optimizer.step()
    if epoch%10==0:
        print(dtime()-start)
print(dtime()-start)


# In[6]:


# data = testdata
# test_target = data.x[:,1].long()
# model.eval()
# pred = model(data).argmax(dim=1)
# correct = (pred == test_target).sum()
# acc = int(correct) / len(target_feat)
# print(f'Accuracy: {acc:.4f}')
# print(acc)
# print(pred)
# with open('1k_iter_acc_fc_mod.pkl', 'wb') as dl_file:
#     pkl.dump(acc,dl_file)

acl = []
model.eval()
for data in testloader:
#     pred = model(data).argmax(dim=1)
#     correct = (pred == data.x[:,1]).sum()
    outs = model(data)
    pred = outs[0].argmax(dim=1)
    correct = (pred == outs[1]).sum()
    acc = int(correct) / len(pred)
    acl.append(acc)
print(np.mean(acl))
# with open('nnacc_100iter_batched.pkl', 'wb') as dl_file:
#     pkl.dump(acl,dl_file)
sb.histplot(acl)


# ## graph feature predictor NN architecture

# In[6]:


class GCNpool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(-1, 32)
        self.pool1 = pool.SAGPooling(32,4,GCNConv)
        self.conv2 = GCNConv(32, 19)

    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index, data.pv.float()
        x = self.conv1(x,edge_index,edge_weight)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)
#         x = self.conv2(x,edge_index,edge_weight)     #old conv2 call

#         #pool returns are [x,edge_index,edge_attr,batch,perm (node indexes i think),score]
        px,pa,pw,_,pn,_ = self.pool1(x=x,edge_index=edge_index,edge_attr=edge_weight)
        x = self.conv2(px,pa,pw)
        real_feature = data.x[pn,1]

        return F.log_softmax(x, dim=1), real_feature


# In[ ]:


#run GCNpool model
model = GCNpool()
loader = trainloader
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model(fulllist[0])

model.train()
start = dtime()

# target_feat = data.card
# target_feat = data.x[:,1].long()
# target_feat = data.x[:,1]

for epoch in range(20):
    for data in trainloader:
#         target_feat = data.x[:,1].long()
        optimizer.zero_grad()
        out,real_feature = model(data)
        loss = F.nll_loss(out,real_feature.long())
        loss.backward()
        optimizer.step()
    if epoch%10==0:
        print(dtime()-start)
print(dtime()-start)


# In[ ]:


# data = testdata
# test_target = data.x[:,1].long()
# model.eval()
# pred = model(data).argmax(dim=1)
# correct = (pred == test_target).sum()
# acc = int(correct) / len(target_feat)
# print(f'Accuracy: {acc:.4f}')
# print(acc)
# print(pred)
# with open('1k_iter_acc_fc_mod.pkl', 'wb') as dl_file:
#     pkl.dump(acc,dl_file)

acl = []
model.eval()
for data in testloader:
#     pred = model(data).argmax(dim=1)
#     correct = (pred == data.x[:,1]).sum()
    outs = model(data)
    pred = outs[0].argmax(dim=1)
    correct = (pred == outs[1]).sum()
    acc = int(correct) / len(pred)
    acl.append(acc)
print(np.mean(acl))
with open('diseq_200iter.pkl', 'wb') as dl_file:
    pkl.dump(acl,dl_file)
sb.histplot(acl)


# In[23]:


#ignore filesave above ^       10 iter gives acc of ~0.50
yeet = pkl.load(open('nnacc_diseq_10iter.pkl','rb'))
sb.histplot(yeet)


# In[ ]:


#probably need to batch each graph to be by itself (instead of letting the nn treat everything as one big but disconnected graph :/)
#then can ensure each graph becomes pooled to just 4 nodes
#done!^   accuracy seems unchanged?    can focus on implementing pooling now
#                                      done!^     can focus on learning gote now? 

#want to eventually switch to a node clustering pooling method, instead of a node drop pooling method like SAGPool?


# ## UMAP

# In[ ]:


#save model
with open('model_gcnp_pval_1000iter.pkl', 'wb') as file:
    pkl.dump(model,file)


# In[10]:


#load model
model = pkl.load(open('model_gcnp_pval_100iter.pkl','rb'))


# In[11]:


#adata construction
gem,num_nodes,promols,enhols = [],[],[],[]
model.eval()
for data in fullloader:
    out,_ = model(data)
#     print('data is ',data)
#     print('out is ',out,'\n\n\n')
    gem.append(out.flatten())
    num_nodes.append(data['num_nodes'])
    promols_enhols = [int(elem) for elem in data['y']]
    promols.append(promols_enhols[0])
    enhols.append(promols_enhols[1:])

print(len(testloader),len(fullloader))
print(len(gem),len(gem[0]))

gema = np.array([elem.detach().numpy() for elem in gem])

adata = sc.AnnData(gema)

#add graph info. to .obs
adata.obs['promols'] = promols
adata.obs['enhols'] = enhols
adata.obs['num_nodes'] = num_nodes

adata


# In[12]:


#optional, add goid to adata
goid_str = [go_getter(elem) for elem in setgoid]

for index,this_goid in enumerate(goid_str):
    adata.obs[this_goid] = gotensor[:,index]
    adata.obs[this_goid] = adata.obs[this_goid].astype(int)


# In[13]:


adata.obs


# In[14]:


sc.tl.pca(adata,n_comps=6)
# sc.tl.pca(adata)


# In[15]:


print([np.sum(adata.obsm['X_pca'][:,comp]) for comp in range(6)])


# In[16]:


sc.pp.neighbors(adata,use_rep='X_pca')

sc.tl.louvain(adata,resolution = 0.2)

sc.tl.paga(
    adata,
    groups='louvain', 
)

sc.tl.umap(
    adata,
    init_pos='X_pca',
    min_dist = 1,
    spread = 1,
)


# In[17]:


fig, ax = plt.subplots(figsize=(15,12))
sc.pl.umap(
    adata,
#     color=['louvain','ont_c','ont_f'], 
    color=['num_nodes'],
#     color=[goid_str[0]],
    size=50,
#     vmin=0, vmax= 100,
    ax=ax
)


# In[18]:


goi = 'Vim'
promol_id = list(zce_rna.var_names).index(goi)
goi_status = adata.obs['promols']==promol_id
adata.obs[goi+'_status'] = goi_status.astype(int)


# In[19]:


fig, ax = plt.subplots(figsize=(15,12))
sc.pl.umap(
    adata,
#     color=['louvain','ont_c','ont_f'], 
    color=[goi+'_status'],
    size=100,
#     vmin=0, vmax= 100,
    ax=ax
)


# In[20]:


sc.pl.umap(adata,size=30,
          color='louvain'
#           vmin=-5,
#           vmax=30,
#           ax=ax,
#           use_raw=True
          )


# In[52]:


go_sums = [np.sum(adata.obs[elem]) for elem in goid_str]

gooi = []
for this_go in range(len(goid_str)):
    if go_sums[this_go] > 1000:
        print(goid_str[this_go])
        gooi.append(goid_str[this_go])


# In[62]:


fig, ax = plt.subplots(figsize=(15,12))
sc.pl.umap(
    adata,
#     color=['louvain','ont_c','ont_f'], 
    color=[gooi[9]], 
    cmap = 'bwr',
#     vmin=0, vmax= 100,
    size=30,
    ax=ax
)


# ## EVERYTHING ELSE

# In[45]:


#graph feature predictor NN architecture
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(-1, 32)
        self.conv2 = GCNConv(32, 19)

    def forward(self, data):
        x, edge_index,edge_weight = data.x, data.edge_index, data.fc.float()
        x = self.conv1(x,edge_index)
#         x = self.conv1(x,edge_index,edge_weight)
        x = F.relu(x)
        x = F.dropout(x,training=self.training)
        x = self.conv2(x,edge_index)
#         x = self.conv2(x,edge_index,edge_weight)

        return F.log_softmax(x, dim=1)


# In[ ]:


#run GCN model
model = GCN()
data = traindata
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model(data)

model.train()
start = dtime()

# target_feat = data.card
target_feat = data.x[:,1].long()
# target_feat = data.x[:,1]

for epoch in range(1000):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out,target_feat)
    loss.backward()
    optimizer.step()
    if epoch%10==0:
        print(dtime()-start)

data = testdata
test_target = data.x[:,1].long()
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred == test_target).sum()
acc = int(correct) / len(target_feat)
print(f'Accuracy: {acc:.4f}')
print(acc)
# with open('1k_iter_acc_fc_mod.pkl', 'wb') as dl_file:
#     pkl.dump(acc,dl_file)


# In[37]:


print(pred)


# In[47]:


pred


# In[69]:


#NOTA BENE!!!!!
#COULD TRY DOING 6-mer TILING OF ALL ENH SEQUENCES AND USING THAT FREQUENCY MATRIX AS FEATURE MATRIX FOR ENHANCERS FOR 
#LEARNING (SHOULD CAPTURE BINDING MOTIFS), INSTEAD OF DOING OTHER LEARNING
#IF THAT IS TOO LARGE A MATRIX, COULD JUST LIMIT TO GHMT MOTIFS (BUT WILL LOSE ANY INDIRECT INFO THEREBY)

# ## Initial pyg data set-up, may not need

# In[74]:


#load in golist
with open("/project/GCRB/Hon_lab/s437603/00.misc/go_terms/mgi.gaf") as file:
    tsv_file = list(csv.reader(file, delimiter="\t"))
    #remove header
    tsv_file = [elem for elem in tsv_file if elem[0][0] != '!']
#record list of all relevant genes' GO IDs and int()ify to create a binary feature matrix with len(gl) rows by len(set(GO IDs)) columns
ggo = [[elem[2],elem[4]] for elem in tsv_file]
cleanggo = [elem for elem in ggo if elem[0] in gl]
intgo = [[elem[0],int(elem[1].split(':')[1])] for elem in cleanggo]
setgoid = list(set([elem[1] for elem in intgo]))

#initialize boolean tensor to represent all gene GO terms, row index is gl and column index is setgoid
gotensor = torch.zeros([len(gl),len(setgoid)],dtype=torch.bool)
for annotation in intgo:
    gene_index = gl.index(annotation[0])
    go_index = setgoid.index(annotation[1])
    gotensor[gene_index,go_index] = True

#iterate through fgl to see which graphs need to be removed from gotensor to match datalist
gote_bool = [type(multigraph)!=int for multigraph in fgl]
fgotensor = gotensor[gote_bool]
    
#saving gotensor and setgoid
with open('pyg_gotensor_'+lib+'.pkl', 'wb') as file:
    pkl.dump(fgotensor,file)
with open('pyg_setgoid_'+lib+'.pkl', 'wb') as file:
    pkl.dump(setgoid,file)
#LEFT OFF HERE, SEE HOW TO INCORPORATE gotensor WITH DATABATCH OBJECTS, MAY NEED TO SAVE AS databatch.go (or something) AND REGENERATE PKL FILES


# In[2]:


#full vectorized graph list
vecl = pkl.load(open('debug?_vec_list_TEPR_'+lib+'.pkl','rb')).copy()

#everything below is for making adata objects out of vecl
namearr = []
xarr = []
for elem_ind in range(len(vecl)):
    if type(vecl[elem_ind]) != int:
        xarr.append(vecl[elem_ind])
        namearr.append(gl[elem_ind])
# xarr = [elem for elem in vecl if type(elem) != int]
xarr = np.array(xarr)
print(np.shape(xarr))
adata = sc.AnnData(xarr)

t_num = 16
full_te_names = []
for t_ind in range(t_num):
    te_names = ['te_t'+str(t_ind)+'_pmu','te_t'+str(t_ind)+'_fmu','te_t'+str(t_ind)+'_pmed','te_t'+str(t_ind)+'_fmed','te_t'+str(t_ind)+'_pstd','te_t'+str(t_ind)+'_fstd','te_t'+str(t_ind)+'_pmin','te_t'+str(t_ind)+'_fmin','te_t'+str(t_ind)+'_pmax','te_t'+str(t_ind)+'_fmax',
                'te_t'+str(t_ind)+'_bon_mu','te_t'+str(t_ind)+'_bon_med','te_t'+str(t_ind)+'_bon_std','te_t'+str(t_ind)+'_bon_min','te_t'+str(t_ind)+'_bon_max','te_t'+str(t_ind)+'_bon_por','te_t'+str(t_ind)+'_bon_ambi']
    full_te_names+=te_names
    
ter_names = ['ter_mup','ter_muf','ter_medp','ter_medf','ter_stdp','ter_stdf','ter_minp','ter_minf','ter_maxp','ter_maxf','ter_bon_mu','ter_bon_med','ter_bon_std','ter_bon_min','ter_bon_max','ter_bon_por','ter_bon_ambi']
tp_names = ['tp_p0','tp_f0','tp_p1','tp_f1','tp_p2','tp_f2','tp_p3','tp_f3','tp_p4','tp_f4','tp_p5','tp_f5','tp_p6','tp_f6','tp_p7','tp_f7','tp_p8','tp_f8',
            'tp_p9','tp_f9','tp_p10','tp_f10','tp_p11','tp_f11','tp_p12','tp_f12','tp_p13','tp_f13','tp_p14','tp_f14','tp_p15','tp_f15','tp_por','tp_ambi']
tr_names = ['tr_p0','tr_f0','tr_p1','tr_f1','tr_p2','tr_f2','tr_p3','tr_f3','tr_p4','tr_f4','tr_p5','tr_f5','tr_p6','tr_f6','tr_p7','tr_f7','tr_p8','tr_f8',
            'tr_p9','tr_f9','tr_p10','tr_f10','tr_p11','tr_f11','tr_p12','tr_f12','tr_p13','tr_f13','tr_p14','tr_f14','tr_p15','tr_f15','tr_por','tr_ambi']
ep_names = ['ep_pmu','ep_fmu','ep_pmed','ep_fmed','ep_pstd','ep_fstd','ep_pmin','ep_fmin','ep_pmax','ep_fmax','ep_len','ep_bon_mu','ep_bon_med','ep_bon_std','ep_bon_min','ep_bon_max','ep_bon_por','ep_bon_ambi']
er_names = ['er_pmu','er_fmu','er_pmed','er_fmed','er_pstd','er_fstd','er_pmin','er_fmin','er_pmax','er_fmax','er_len','er_bon_mu','er_bon_med','er_bon_std','er_bon_min','er_bon_max','er_bon_por','er_bon_ambi']
pr_names = ['pr_p','pr_f']
adata.var_names = full_te_names+ter_names+tp_names+tr_names+ep_names+er_names+pr_names

#for annotating .obs
adata.obs['gid'] = namearr

agl = list(adata.obs['gid'])
cardiac_bool = [int(gene in cgl) for gene in agl]
fibroblast_bool = [int(gene in fibrogl) for gene in agl]

adata.obs['ont_c'] = cardiac_bool
adata.obs['ont_f'] = fibroblast_bool


# In[33]:


datalist=[]
for graph_id,multidigraph in enumerate(fgl):
    #skips over the few empty graphs
    if type(multidigraph) == int:
        continue
    #must convert graphs to nx.DiGraphs before converting to PyG Data() objects
    this_digraph = nx.DiGraph(multidigraph)
    
    #node-level (x) and graph-level (y) data, according to pyg convention
    #x saves node type (PAR value -{0,1,2}- shows whether node comes from Pert.lib, ATAC lib, or RNA lib)
    #and identity (PERTEP value -{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}- shows whether a node is preturbation (0-15), an enh. (16), a prom. (17), or an RNA (18))
    x=[]
    for node_id,node_name in enumerate(list(this_digraph.nodes)):
        if node_id < len(perts): #categorizes all perturbation nodes
            PARval = 0
            PERTEPval = node_id
        elif 'ATAC' in node_name: #categorizes all enh nodes followed by the promoter node
            PARval = 1
            if 'Promoter' not in node_name:
                PERTEPval = 16
            else:
                PERTEPval = 17
        else: #categorizes the RNA node
            PARval = 2
            PERTEPval = 18

        x.append([PARval,PERTEPval])
    x = Tensor(x)

    #y saves promol index associated with each graph, followed by list of enh IDs corresponding to index in enhol list
    y = Tensor([graph_id]+[int(elem.split('\n')[0][1:]) for elem in list(this_digraph.nodes)[16:-2]])
    
    #create pyg Data object for this graph
    this_Data = pyg.utils.from_networkx(G=this_digraph)
    this_Data.x,this_Data.y = x,y
    
    datalist.append(this_Data)

#convert list of pyg Data objects into pyg dataloader (not certain if this is needed for model building or can just leave in form of "datalist"?)
loader = pyg.loader.DataLoader(datalist,batch_size=len(datalist))
databatch = [elem for elem in loader][0]

#saving datalist and databatch
#saving datalist
with open('pyg_datalist_'+lib+'.pkl', 'wb') as dl_file:
    pkl.dump(datalist,dl_file)
# with open('pyg_databatch_'+lib+'.pkl', 'wb') as dl_file:
#     pkl.dump(databatch,dl_file)


# In[33]:


#create heterodatas from networkx
datalist=[]
for graph_id,multidigraph in enumerate(fgl):
    #skips over the few empty graphs
    if type(multidigraph) == int:
        continue
    #must convert graphs to nx.DiGraphs before converting to PyG Data() objects
    this_digraph = nx.DiGraph(multidigraph)

    #node-level {x_pert,x_enh,x_prom,x_rna} and graph-level (y) data, according to pyg convention
    
    
    #x saves node type (PAR value -{0,1,2}- shows whether node comes from Pert.lib, ATAC lib, or RNA lib)
    #and identity (PERTEP value -{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}- shows whether a node is perturbation (0-15), an enh. (16), a prom. (17), or an RNA (18))
    x=[]
    for node_id,node_name in enumerate(list(this_digraph.nodes)):
        if node_id < len(perts): #categorizes all perturbation nodes
            PARval = 0
            PERTEPval = node_id
        elif 'ATAC' in node_name: #categorizes all enh nodes followed by the promoter node
            PARval = 1
            if 'Promoter' not in node_name:
                PERTEPval = 16
            else:
                PERTEPval = 17
        else: #categorizes the RNA node
            PARval = 2
            PERTEPval = 18

        x.append([PARval,PERTEPval])
    x = Tensor(x)

    #y saves promol index associated with each graph, followed by list of enh IDs corresponding to index in enhol list
    y = Tensor([graph_id]+[int(elem.split('\n')[0][1:]) for elem in list(this_digraph.nodes)[16:-2]])
    
    #create pyg Data object for this graph
    this_Data = pyg.utils.from_networkx(G=this_digraph)
    this_Data.x,this_Data.y = x,y
    
    datalist.append(this_Data)

#convert list of pyg Data objects into pyg dataloader (not certain if this is needed for model building or can just leave in form of "datalist"?)
loader = pyg.loader.DataLoader(datalist,batch_size=len(datalist))
databatch = [elem for elem in loader][0]

#saving datalist and databatch
#saving datalist
with open('pyg_datalist_'+lib+'.pkl', 'wb') as dl_file:
    pkl.dump(datalist,dl_file)
# with open('pyg_databatch_'+lib+'.pkl', 'wb') as dl_file:
#     pkl.dump(databatch,dl_file)


# In[33]:


#add promoter distance feature to all ENH nodes

# scan by PAR or pertep value, asign distance from prom to each E node, and -1 to other nodes     
#may want to change this later, see how others deal with heterodata?
start = dtime()
diseq_l = []
for graph_id,graph in enumerate(datalist):
    #initialize tensor (num_nodes x [prom.distance,6mer adata])
    diseq = torch.zeros((datalist[graph_id].num_nodes,len(adata_6mer.var)+1))
    promol_id,enhol_ids = int(graph.y[0]),graph.y[1:].tolist()
    enhol_ids = [int(elem) for elem in enhol_ids]
    
    #assign E->P distances column, leaving distance as 0 for non-enhancer nodes
    prom_pos = int((ce_atac.var.iloc[promol[promol_id][2],3]+ce_atac.var.iloc[promol[promol_id][2],4])/2)                                                    #finds midpoint of promoter peak
    enh_pos_list = [int((ce_atac.var.iloc[enhol[promol_id][2][enhol_id],3]+ce_atac.var.iloc[enhol[promol_id][2][enhol_id],4])/2) for enhol_id in enhol_ids]  #finds midpoint of enhancer peaks
    enh_dist_list = [abs(enh_pos-prom_pos) for enh_pos in enh_pos_list]
    diseq[16:-2,0] = Tensor(enh_dist_list)
    
    #assign 6mer tensor to atac nodes, leaving tensor as all 0s for non-atac nodes
    ids_6mer = [enhol[promol_id][2][enhol_id] for enhol_id in enhol_ids]+[promol[promol_id][2]]
    for offset,id_6mer in enumerate(ids_6mer):
        diseq[16+offset,1:] = Tensor(adata_6mer[id_6mer,:].X)
#     graph.x = torch.cat((graph.x,diseq),dim=1) #adds diseq to graphs as it goes
    diseq_l.append(diseq)
print(dtime() - start)

#save diseq
with open('diseq_list_'+lib+'.pkl', 'wb') as dl_file:
    pkl.dump(diseq_l,dl_file)
    


# In[ ]:


#add number of associated genes feature for all ATAC nodes
# scan by PAR or pertep value, asign associated-genes number to each E or P node, and -1 to other nodes
start = dtime() #del when done
adhoc_promol = promol

asogene_l = []
for graph_id,graph in enumerate(datalist):
    #initialize tensor (num_nodes x 1)
    asogene = torch.zeros((datalist[graph_id].num_nodes,1))
    
    #track down atac_ids and positions
    promol_id,enhol_ids = int(graph.y[0]),graph.y[1:].tolist()
    enhol_ids = [int(elem) for elem in enhol_ids]
    atac_ids = [enhol[promol_id][2][enhol_id] for enhol_id in enhol_ids]+[promol[promol_id][2]]
    
    #lazy way to constrain promol search
    if graph_id > 200:
        adhoc_promol = promol[graph_id-200:]
    
    #record number of promol entries within 100kb range of each atac_id
    for this_id,this_entry in enumerate(atac_ids):
        open_window,close_window = int(ce_atac[:,this_entry].var['start']) - 1e5, int(ce_atac[:,this_entry].var['end']) + 1e5
            
        for prom in adhoc_promol:
            tss_info = ce_atac[:,prom[2]]
            if int(tss_info.var['end']) < open_window:
                continue
            elif close_window < int(tss_info.var['start']):
                break
            else:
#             elif (open_window < tss_info.var['start'] and tss_info.var['start'] < close_window) or (open_window < tss_info.var['end'] and tss_info.var['end'] < close_window):
                asogene[len(ccc[0])+this_id-1] += 1
    asogene_l.append(asogene)
    
    if graph_id%100==0:
        print(graph_id, dtime() - start)
    
print(dtime() - start)

#save asogene_l
with open('asogene_list_'+lib+'.pkl', 'wb') as al_file:
    pkl.dump(asogene_l,al_file)


# In[6]:


promol[0:5]


# In[12]:


#add number of associated genes feature for all ATAC nodes
# scan by PAR or pertep value, asign associated-genes number to each E or P node, and -1 to other nodes
start = dtime() #del when done
adhoc_promol = promol

asogene_l = []
for graph_id,graph in enumerate(datalist):
    #initialize tensor (num_nodes x 1)
    asogene = torch.zeros((datalist[graph_id].num_nodes,1))
    
    #track down atac_ids and positions
    promol_id,enhol_ids = int(graph.y[0]),graph.y[1:].tolist()
    enhol_ids = [int(elem) for elem in enhol_ids]
    atac_ids = [enhol[promol_id][2][enhol_id] for enhol_id in enhol_ids]+[promol[promol_id][2]]
    
    #lazy way to constrain promenhol search
    if graph_id > 200:
        adhoc_promol = promol[graph_id-200:]
    
    for atac_id in atac_ids:
    
    for tss in promenhol:
        promenhs = tss[2]
        if promenhs[0] > a_ind:
            break
        if a_ind in enhs:
            c4_this_a+=1
    if tss_id%100 == 0:
        print(tss_id,dtime() - start)
    test_list.append(c4_this_a)
print(dtime() - start)

#create binary matrix from all perturbation combos
tf_combos = ccc[0].copy()
tf_combos[0] = ''
tf_te = torch.zeros((len(tf_combos),len(tf_combos[len(tf_combos)-1].split(','))))     #initializes tensor of (# of pert combos) by (# of component perts)             #            G   H   M   T
for combo_index,combo in enumerate(tf_combos):                                                                                                                        # combo 0    -   -   -   -
    #populate tf_te, assume GHMT ordering                                                                                                                             # combo 1    -   -   -   -
    component_perts = combo.split(',')                                                                                                                                # combo n    -   -   -   -
    if 'Gata4' in component_perts:
        tf_te[combo_index,0] = 1
    if 'Hand2' in component_perts:
        tf_te[combo_index,1] = 1
    if 'Mef2c' in component_perts:
        tf_te[combo_index,2] = 1
    if 'Tbx5' in component_perts:
        tf_te[combo_index,3] = 1
        
#save tf_te
with open('component_tfs_tensor.pkl', 'wb') as dl_file:
    pkl.dump(tf_te,dl_file)


# In[25]:


# (for homodata)
# add promoter distance feature to all ENH nodes
# scan by PAR or pertep value, asign distance from prom to each E node, and -1 to other nodes     
#may want to change this later, see how others deal with heterodata?

diseq_l = pkl.load(open('diseq_list_'+lib+'.pkl','rb'))
for graph_id,graph in enumerate(datalist):
    #initialize tensor (num_nodes x [prom.distance,6mer adata])
    diseq = diseq_l[graph_id]
    graph.x = torch.cat((graph.x,diseq),dim=1)


# In[ ]:


#OLD homodata prep

trainlist,testlist = datalist[:int(len(datalist)/2)],datalist[int(len(datalist)/2):]
#heterize datasets
fulllist = [pyg.data.HeteroData(data_entry) for data_entry in datalist]
trainlist = [pyg.data.HeteroData(data_entry) for data_entry in trainlist]
testlist = [pyg.data.HeteroData(data_entry) for data_entry in testlist]

#transform fc to belong to {R>=0}     note that current_fc=5*np.log2(raw_fc+2**-5)
for entry in fulllist:
    entry.fc = 2**(entry.fc/5)-2**-5
for entry in trainlist:
    entry.fc = 2**(entry.fc/5)-2**-5
for entry in testlist:
    entry.fc = 2**(entry.fc/5)-2**-5

#loaderize datasets
fullloader = pyg.loader.DataLoader(fulllist,shuffle=True)
trainloader = pyg.loader.DataLoader(trainlist,shuffle=True)
testloader = pyg.loader.DataLoader(testlist,shuffle=True)
