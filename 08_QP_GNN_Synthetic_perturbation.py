
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
import torch as tc
import torch_geometric as pyg
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, ResGatedGraphConv, SAGEConv, pool, to_hetero, to_hetero_with_bases
import torch_geometric.transforms as T
from scipy.stats import percentileofscore as pos

# tv = torch.load('pyg_hetlist_tv10.pt')
tv = torch.load('pyg_hetlist_tv25_6merrchs.pt')
# tv = torch.load('pyg_hetlist_tv50.pt')


# In[3]:


#heterodata prep and loading
fulllist = tv
mask = pkl.load(open('newbtrainmask.pkl','rb'))
testlist = [fulllist[ind] for ind in range(len(fulllist)) if mask[ind]]
trainlist = [fulllist[ind] for ind in range(len(fulllist)) if not mask[ind]]


# In[3]:


#loaderize datasets
gpb = 10 #graphs per batch
fullloader = pyg.loader.DataLoader(fulllist,shuffle=False)
trainloader = pyg.loader.DataLoader(trainlist,shuffle=True,batch_size=gpb)
testloader = pyg.loader.DataLoader(testlist,shuffle=False)


# In[12]:


x=1404
mask[x-1:x+5]


# In[2]:


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


# In[3]:


seqnames = list(adata_6mer.var_names)


# In[4]:


rc_seqs = []
for sid,seq in enumerate(seqnames):
    seqtup = list(set((seqnames.index(seq),seqnames.index(rev_comp(seq)))))
    if seqtup not in rc_seqs:
        rc_seqs.append(seqtup)


# In[25]:


#del when done, for checking number of sig. hits per edge type for F2
#need te,ep,pr edges
threshte = -np.log10(0.05/15761408)
threshep = -np.log10(0.05/985088)
threshpr = -np.log10(0.05/35038)

sigte,sigep,sigpr = 0,0,0
for gid,graph in enumerate(tv):
    sigte += sum(graph['perturbation','pvalue','enhancer'].edge_attr>threshte)
    sigep += sum(graph['enhancer','pvalue','promoter'].edge_attr>threshep)
    sigpr += sum(graph['promoter','pvalue','rna'].edge_attr>threshpr)
    if gid%1000==0:
        print(gid)
sigte,sigep,sigpr


# In[16]:


sum(tv[1404]['perturbation','pvalue','enhancer'].edge_attr>10)


# # Data Manipulation

# In[3]:


tv = torch.load('pyg_hetlist_tv25_6merrchs.pt')
mask = pkl.load(open('newbtrainmask.pkl','rb'))


# In[4]:


fulllist = tv
testlist = [fulllist[ind] for ind in range(len(fulllist)) if mask[ind]]
trainlist = [fulllist[ind] for ind in range(len(fulllist)) if not mask[ind]]
#loaderize datasets
gpb = 20 #graphs per batch
fullloader = pyg.loader.DataLoader(fulllist,shuffle=False)
trainloader = pyg.loader.DataLoader(trainlist,shuffle=True,batch_size=gpb)
testloader = pyg.loader.DataLoader(testlist,shuffle=False)


# In[5]:


#which perturbations to test??
#    6mer changes: adding and removing Gata4 motif, or empirically important/unimportant motifs?   scale to what?
#    enh changes: translocate with T-E edges but ignore E-X? give neutral E-X, hopefully more prone to false negatives not false positives
#    promoter changes:
#should edge info. be manipulated or "carried over" with other perturbations?

#look at distribution of per graph accuracies 
#scan effect of removal of each enh. node, score based on ???   (probably only care about correct predictions being changed when enh is removed)
#    does prediction change mirror T->E values, other edges, or have more to do with E sequence?
#    how can you decide which changed predictions are more reliable? 
#    can any of the perts have outside confirmation? e.g. GHMT treatment or development changes associated with a particular motif or enh alteration?
#    
#


# In[5]:


#load in/generate all the motif indices
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
rc_names = [tuple([seqnames[ind] for ind in elem]) for elem in rc_seqs]
hsrc_names = [rc_names[idx] for idx in range(len(rc_names)) if hsp[0][idx]]


# In[36]:


with open('hsrc_names.pkl', 'wb') as file:
    pkl.dump(hsrc_names,file)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#add tv25 ground truth to tv10 dataset for comparing accuracy outside of tv10-25 boundary
tv10 = torch.load('pyg_hetlist_tv10_6merrchs.pt')
tv25 = torch.load('pyg_hetlist_tv25.pt')

for gid,graph in enumerate(tv10):
    graph.z = tv25[gid].y
torch.save(tv10,'pyg_hetlist_tv1025')


# ## Running select graphs

# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pickle
import io
class CPU_unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#use example:
#model = CPU_unpickler(open('model_iterX.pkl','rb')).load()

#takes trinary encoded 'real' and returns one hot version (up,down)
def get_oh(real):
    real_oh = []
    for elem in real:
        if elem==0:
            real_oh.append(0)
            real_oh.append(1)
        elif elem==1:
            real_oh.append(0)
            real_oh.append(0)
        elif elem==2:
            real_oh.append(1)
            real_oh.append(0)
    real_oh = tc.Tensor([real_oh])
    return real_oh

#takes trinary encoded one hot and returns 'real' version (down/neutral/up)
def get_trin(oh):
    oh = oh[0]
    oh_real = []
    for ind in range(int(len(oh)/2)):
        if oh[2*ind]<0.5 and oh[2*ind+1]>=0.5:
            oh_real.append(0)
        elif oh[2*ind]<0.5 and oh[2*ind+1]<0.5:
            oh_real.append(1)
        elif oh[2*ind]>=0.5 and oh[2*ind+1]<0.5:
            oh_real.append(2)
        # always erroneous, up AND down prediction
        elif oh[2*ind]>=0.5 and oh[2*ind+1]>=0.5:
            oh_real.append(-13) #ensures error tracing for linear model
    oh_real = tc.Tensor([oh_real])        
    return oh_real

sig = tc.nn.Sigmoid() #define sigmoid function for transforming outputs

#given pre-defined and ".eval()"-ed model, a real graph, and the perturbed corresponding graph, returns the correct values followed by [pred(real g),pred(pert g)]
def sp_predictor(real_graph,pert_graph):
    loader = pyg.loader.DataLoader([real_graph,pert_graph],shuffle=False,batch_size=2)
    correct = get_oh(real_graph['y'])
    sp_predictions = []
    for data in loader:
        out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
        numof_g = len(data['rna']['batch'])
        tp_pool = torch.empty((1,32*numof_g))
        for g in range(numof_g):
            id_start = 16*g
            id_end = id_start + 16
            e_start,e_end = data['enhancer'].ptr[g],data['enhancer'].ptr[g+1]
            ot = out['perturbation'][id_start:id_end,:].sum(dim=0)
            oe = out['enhancer'][e_start:e_end,:].sum(dim=0)
            op = out['promoter'][g,:].sum(dim=0)
            tp_pool_pergraph = ot+oe+op
            tp_pool_pergraph = tp_pool_pergraph.reshape(1,32)
            tp_pool[0,2*id_start:2*id_end]= tp_pool_pergraph
        pred = tp_pool
        pred = torch.where(pred>=0,1,0)
        sp_predictions.append(pred)
    return correct,sp_predictions

#given 6mer string and hsrc_names, returns index of 6mer
def motif_indexer(motif):
    for names_idx,names_tup in enumerate(hsrc_names):
        if motif in names_tup:
            idx = names_idx
    return idx


# In[4]:


model = CPU_unpickler(open('hg5_udep_hsbb_iter2000.pkl','rb')).load() #emb6 model
model.eval()
tv = torch.load('pyg_hetlist_tv25_6merrchs.pt')
mask = pkl.load(open('newbtrainmask.pkl','rb'))
ghl = pkl.load(open('hl_gene_names.pkl','rb'))

hsrc_names = pkl.load(open('hsrc_names.pkl','rb'))


# In[13]:


#dwd, like actually
motif = 'AGGTGT'  #Tbx5 IN
mid = motif_indexer(motif)
tv[3379]['enhancer'].x[:,1+mid]


# In[12]:


motif = 'AGGTGT'  #Tbx5 IN
mid = motif_indexer(motif)
mid


# In[ ]:


agl = []
for gnum in range(len(tv)):
    real_graph = tv[gnum]
    pg = real_graph.clone()
    e_subs=list(range(len(pg['enhancer'].x)))
    for x in range(0,int(len(e_subs)*0.5)):
        e_subs.remove(x)
#     e_subs.remove()
    subset_dict = {'enhancer':tc.tensor(e_subs)}
    pg = pg.subgraph(subset_dict)
    pert_graph = pg
    corr,preds = sp_predictor(real_graph,pert_graph)
    sim = get_trin(preds[0])[0][0:16]==get_trin(preds[0])[0][16:32]
    sim = sim.sum().item()/16
    agl.append(sim)
    
#     print(get_trin(corr),'CORR HERE')
#     print(get_trin(preds[0])[0][0:16])
#     print(get_trin(preds[0])[0][16:32])


# In[5]:


tgrh = 'Ttn'
gx=ghl.index(tgrh)
print(ghl[gx-1:gx+10])
print(gx)
# print(ce_rna[:,promol[gl.index(tgrh)][1]].X)


# In[11]:


# for gnum in range(10):
for gnum in [3379]:
    real_graph = tv[gnum]
    pg = real_graph.clone()
    pg['enhancer'].x[:,1] = pg['enhancer'].x[:,1]
    #LEFT OFF HERE, NEED TO MODIFY select e nodes, try adding or removing select 6mers?  iterate through all?     might need to sample multiple times due to stochastic prediction???
    pert_graph = pg
    corr,preds = sp_predictor(real_graph,pert_graph)

    print(get_trin(corr),'CORR HERE')
    print(get_trin(preds[0])[0][0:16])
    print(get_trin(preds[0])[0][16:32])


# In[ ]:


#FOR POINT PERTURBATION OF ENH NODES
racc,pacc = [],[]
for repeat in range(10):
    for gnum in [3379]:
        real_graph = tv[gnum]
        pg = real_graph.clone()
        e_subs=list(range(len(pg['enhancer'].x)))
        print(e_subs)
    #     for x in range(0,int(len(e_subs)*0.8)):
        for x in range(len(pg['enhancer'].x)):
            print(x)
            e_subs.remove(x)
    #     e_subs.remove(17)
        subset_dict = {'enhancer':tc.tensor(e_subs)}
        print(subset_dict)
        pg = pg.subgraph(subset_dict)

        pert_graph = pg
        corr,preds = sp_predictor(real_graph,pert_graph)

#         print(get_trin(corr),'CORR')
#         print(get_trin(preds[0])[0][0:16],'REAL PRED')
#         print(get_trin(preds[0])[0][16:32],'PERT PRED')
        rscore = (get_trin(corr)==get_trin(preds[0])[0][0:16]).sum()/16
        pscore = (get_trin(corr)==get_trin(preds[0])[0][16:32]).sum()/16
        racc.append(rscore.item())
        pacc.append(pscore.item())


# In[166]:


bw = 5
sb.kdeplot(racc,color='red',bw_adjust=bw,clip=(0,1)),sb.kdeplot(pacc,color='blue',bw_adjust=bw,clip=(0,1))


# In[164]:


bw = 5
sb.kdeplot(racc,color='red',bw_adjust=bw,clip=(0,1)),sb.kdeplot(pacc,color='blue',bw_adjust=bw,clip=(0,1))


# In[162]:


bw = 5
sb.kdeplot(racc,color='red',bw_adjust=bw,clip=(0,1)),sb.kdeplot(pacc,color='blue',bw_adjust=bw,clip=(0,1))


# In[160]:


bw = 5
sb.kdeplot(racc,color='red',bw_adjust=bw,clip=(0,1)),sb.kdeplot(pacc,color='blue',bw_adjust=bw,clip=(0,1))


# In[97]:


#saving modified graphs
with open('synthg_pkp1_ol_tnnt2.pkl','wb') as file:
    pkl.dump(pg,file)


# In[22]:


tv[1404]['enhancer'].x.size()[0],tv[1408]['enhancer'].x.size()[0]


# In[ ]:


#motif reference:
#G: m&h  cTTATCT   
#H: h    CAGATG
#M: h    aAAATAg
#T: h    AGGTGT


# In[34]:


# tv[gx][('promoter','foldchange','rna')]
tv[gx][('perturbation','foldchange','promoter')]


# In[95]:


gi = ghl.index('Mansc4')
print(gi)
print(ghl[gi-1:gi+5])


# In[12]:


#FOR POINT PERTURBATION OF MOTIFS

# motif = 'AGATAA' #gata4 IN
# motif = 'GATAAG' #gata4
# motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
# motif = 'AAATAG' #Mef2c
motif = 'AGGTGT'  #Tbx5 IN

repeats = 10
real_graph = tv[3379]
# real_graph = tv[gx]
mid = motif_indexer(motif)
enh_num = len(real_graph['enhancer'].x)
diflist = []

for which_e in range(enh_num):
    tq_real = torch.zeros(16)
    tq_pert = torch.zeros(16)
    for reps in range(repeats):
        pg = real_graph.clone()
        pg['enhancer'].x[which_e,1+mid] += 1
    #     for mid in midlist:
    #         pg['enhancer'].x[:,1+mid] = 0
        pert_graph = pg
        corr,preds = sp_predictor(real_graph,pert_graph)

    #     print(get_trin(preds[0])[0][0:16][-5:],'REAL PRED')
    #     print(get_trin(preds[0])[0][16:32][-5:],'PERT PRED')
        unmod_real = get_trin(preds[0])[0][0:16]
        unmod_pert = get_trin(preds[0])[0][16:32]
        mod_real = torch.where(unmod_real==-13,1,unmod_real)
        mod_pert = torch.where(unmod_pert==-13,1,unmod_pert)
        tq_real += mod_real
        tq_pert += mod_pert
    tq_real = tq_real/repeats
    tq_pert = tq_pert/repeats

    tq_real = [round(elem.item(),3) for elem in tq_real]
    tq_pert = [round(elem.item(),3) for elem in tq_pert]

    print('REAL:\t',[int(elem.item()) for elem in get_trin(corr)[0]])
    print('PRED:\t',tq_real)
    print('PERT:\t',tq_pert)
    
    dif = 0
    for idx in range(len(tq_real)):
        dif+=abs(tq_real[idx]-tq_pert[idx])
    print(dif)
    diflist.append(dif)
diflist = [round(elem,3) for elem in diflist]
# print(tq_real)
# print(tq_pert)


# In[26]:


#FOR POINT PERTURBATION OF MOTIFS, all enh at a time

# motif = 'AGATAA' #gata4 IN
# motif = 'GATAAG' #gata4
# motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
# motif = 'AAATAG' #Mef2c
motif = 'AGGTGT'  #Tbx5 IN

repeats = 2
real_graph = tv[3379]
# real_graph = tv[gx]
mid = motif_indexer(motif)

tq_real = torch.zeros(16)
tq_pert = torch.zeros(16)
for reps in range(repeats):
    pg = real_graph.clone()
    pg['enhancer'].x[:,1+mid] = 20
#     for mid in midlist:
#         pg['enhancer'].x[:,1+mid] = 0
    pert_graph = pg
    corr,preds = sp_predictor(real_graph,pert_graph)

#     print(get_trin(preds[0])[0][0:16][-5:],'REAL PRED')
#     print(get_trin(preds[0])[0][16:32][-5:],'PERT PRED')
    unmod_real = get_trin(preds[0])[0][0:16]
    unmod_pert = get_trin(preds[0])[0][16:32]
    mod_real = torch.where(unmod_real==-13,1,unmod_real)
    mod_pert = torch.where(unmod_pert==-13,1,unmod_pert)
    tq_real += mod_real
    tq_pert += mod_pert
tq_real = tq_real/repeats
tq_pert = tq_pert/repeats

tq_real = [round(elem.item(),3) for elem in tq_real]
tq_pert = [round(elem.item(),3) for elem in tq_pert]

print('REAL:\t',[int(elem.item()) for elem in get_trin(corr)[0]])
print('PRED:\t',tq_real)
print('PERT:\t',tq_pert)

dif = 0
for idx in range(len(tq_real)):
    dif+=abs(tq_real[idx]-tq_pert[idx])
dif = round(dif,3)
print(dif)
# print(tq_real)
# print(tq_pert)


# In[27]:


#saving modified graphs
with open('synthg_ttn_t20.pkl','wb') as file:
    pkl.dump(pg,file)


# In[7]:


#diflists
#Ttn (index 3379 for enhol, 3379 for tv)
#
#chr2:76926625-76927441     first e
#chr2:76981450-76982325     prom
#chr2:77080969-77081860     last e
#
#Mef2c Motif: 0.12, [0.06, 0.16, 0.12, 0.07, 0.1, 0.13, 0.17, 0.08, 0.07, 0.04, 0.14, 0.09, 0.1, 0.03, 0.07, 0.07, 0.07, 0.09, 0.16, 0.17]
#Tbx5 Motif: 0.18, [0.103, 0.09, 0.089, 0.056, 0.057, 0.122, 0.225, 0.309, 0.082, 0.093, 0.121, 0.136, 0.111, 0.056, 0.215, 0.1, 0.085, 0.17, 0.068, 0.07]
#
#Myom2 (index 15856 for enhol, 15845 for tv)
#
#chr8:14964021-14964823
#chr8:15058510-15059365
#chr8:15155737-15156571
#
#Tbx5 Motif: 0.337, [0.065, 0.081, 0.056, 0.157, 0.053, 0.092, 0.061, 0.105, 0.076, 0.055, 0.187, 0.077, 0.053, 0.087, 0.135, 0.074, 0.064, 0.088, 0.058, 0.045, 0.059, 0.054, 0.091, 0.121, 0.059, 0.063, 0.129, 0.093, 0.07, 0.074, 0.108, 0.115, 0.087, 0.089, 0.075, 0.13, 0.058]
#Mef2c Motif: , []
#
#Csrp3 (index 14074 for enhol, 14065 for tv)
#
#chr7:48749265-48750024
#chr7:48847860-48848801
#chr7:48941364-48942331
#
#Mef2c Motif:
#Tbx5 Motif: 0.092, [0.373, 0.27, 0.192, 0.228, 0.137, 0.21, 0.281, 0.244, 0.256, 0.309, 0.295, 0.425, 0.487, 0.319, 0.314, 0.394, 0.3, 0.268, 0.538, 0.213, 0.172, 0.259, 0.359, 0.519, 0.478, 0.199, 0.381, 0.312, 0.249, 0.27, 0.268, 0.356, 0.304, 0.166, 0.158, 0.401, 0.223]
#
#Tnnt2 (index ? for enhol, 1404 for tv)
#
#Mef2c Motif:
#Tbx5 Motif: 0.004, [0.005, 0.003, 0.004, 0.0, 0.003, 0.004, 0.007, 0.004, 0.004, 0.002, 0.003, 0.004, 0.006, 0.001, 0.008, 0.005, 0.007, 0.001, 0.006, 0.002, 0.006, 0.004, 0.006, 0.003, 0.003, 0.001, 0.003, 0.004, 0.006, 0.004, 0.004, 0.008, 0.006, 0.003, 0.01, 0.005, 0.003, 0.004, 0.004, 0.005, 0.005, 0.004, 0.01, 0.005, 0.004, 0.004
#
#Fap (index ? for enhol, 3154 for tv)
#
##Tbx5 Motif: 0.0312, [0.024, 0.023, 0.043, 0.016, 0.048, 0.018, 0.019, 0.023, 0.022, 0.01, 0.027, 0.025, 0.022, 0.045, 0.033, 0.031, 0.023, 0.025, 0.061, 0.026, 0.034, 0.063, 0.057, 0.033, 0.032]
#


# In[14]:


dif


# In[15]:


print(diflist[0:11])
print(diflist[10:21])
print(diflist[20:31])
print(diflist[30:])


# In[7]:


#for finding tss node number
enhol = pkl.load(open('co1000ce_enh_overlaps.pkl','rb'))


# In[33]:


enhol[3153],len(enhol[3153][2])


# In[2]:


myom = [0.065, 0.081, 0.056, 0.157, 0.053, 0.092, 0.061, 0.105, 0.076, 0.055, 0.187, 0.077, 0.053, 0.087, 0.135, 0.074, 0.064, 0.088, 0.058, 0.045, 0.059, 0.054, 0.091, 0.121, 0.059, 0.063, 0.129, 0.093, 0.07, 0.074, 0.108, 0.115, 0.087, 0.089, 0.075, 0.13, 0.058]
ttn = [0.103, 0.09, 0.089, 0.056, 0.057, 0.122, 0.225, 0.309, 0.082, 0.093, 0.121, 0.136, 0.111, 0.056, 0.215, 0.1, 0.085, 0.17, 0.068, 0.07]
tnnt = [0.005, 0.003, 0.004, 0.0, 0.003, 0.004, 0.007, 0.004, 0.004, 0.002, 0.003, 0.004, 0.006, 0.001, 0.008, 0.005, 0.007, 0.001, 0.006, 0.002, 0.006, 0.004, 0.006, 0.003, 0.003, 0.001, 0.003, 0.004, 0.006, 0.004, 0.004, 0.008, 0.006, 0.003, 0.01, 0.005, 0.003, 0.004, 0.004, 0.005, 0.005, 0.004, 0.01, 0.005, 0.004, 0.004]
csrp = [0.373, 0.27, 0.192, 0.228, 0.137, 0.21, 0.281, 0.244, 0.256, 0.309, 0.295, 0.425, 0.487, 0.319, 0.314, 0.394, 0.3, 0.268, 0.538, 0.213, 0.172, 0.259, 0.359, 0.519, 0.478, 0.199, 0.381, 0.312, 0.249, 0.27, 0.268, 0.356, 0.304, 0.166, 0.158, 0.401, 0.223]
fap = [0.024, 0.023, 0.043, 0.016, 0.048, 0.018, 0.019, 0.023, 0.022, 0.01, 0.027, 0.025, 0.022, 0.045, 0.033, 0.031, 0.023, 0.025, 0.061, 0.026, 0.034, 0.063, 0.057, 0.033, 0.032]


# In[31]:


#ttn tss: id 6-7
#myom tss: id 22-23
#tnnt tss: id 18-19
#csrp tss: id 18-19
#fap tss: id 7-8


# In[3]:


tto = np.flip(np.sort(ttn)).reshape(1,-1)
tno = np.flip(np.sort(tnnt)).reshape(1,-1)
cso = np.flip(np.sort(csrp)).reshape(1,-1)
fao = np.flip(np.sort(fap)).reshape(1,-1)
myo = np.flip(np.sort(myom)).reshape(1,-1)


# In[3]:


tto = np.array(ttn).reshape(1,-1)
tno = np.array(tnnt).reshape(1,-1)
cso = np.array(csrp).reshape(1,-1)
fao = np.array(fap).reshape(1,-1)
myo = np.array(myom).reshape(1,-1)


# In[4]:


cmap = 'viridis'

fig, ax = plt.subplots(figsize=(20,1))
sb.heatmap(tto,vmin=0,vmax=0.5,cmap=cmap,linewidths=0.1)
fig, ax = plt.subplots(figsize=(37,1))
sb.heatmap(cso,vmin=0,vmax=0.5,cmap=cmap,linewidths=0.1)
fig, ax = plt.subplots(figsize=(37,1))
sb.heatmap(myo,vmin=0,vmax=0.5,cmap=cmap,linewidths=0.1)
fig, ax = plt.subplots(figsize=(46,1))
sb.heatmap(tno,vmin=0,vmax=0.5,cmap=cmap,linewidths=0.1)
fig, ax = plt.subplots(figsize=(25,1))
sb.heatmap(fao,vmin=0,vmax=0.5,cmap=cmap,linewidths=0.1)


# In[94]:


fig, ax = plt.subplots(figsize=(40,14))
log = False
csoff,ttoff,faoff,tnoff = 4,16,15,4
plt.bar(x=list(range(0,len(cso[0])+csoff,1)),height=[0]*csoff+cso[0].tolist(),color='palevioletred',log=log,alpha=1)
plt.bar(x=list(range(0,len(tto[0])+ttoff,1)),height=[0]*ttoff+tto[0].tolist(),color='darkred',log=log,alpha=1)
plt.bar(x=list(range(0,len(myo[0]),1)),height=myo[0].tolist(),color='gold',log=log,alpha=1)
plt.bar(x=list(range(0,len(fao[0])+faoff,1)),height=[0]*faoff+fao[0].tolist(),color='mediumseagreen',log=log,alpha=1)
plt.bar(x=list(range(0,len(tno[0])+tnoff,1)),height=[0]*tnoff+tno[0].tolist(),color='mediumblue',log=log,alpha=1)


# In[3]:


mt = [0.07, 0.1, 0.13, 0.17, 0.08, 0.07, 0.04, 0.14, 0.09, 0.1, 0.03, 0.07, 0.07, 0.07]
plt.bar(x=list(range(0,28,2)),height=mt,color='darkred')


# In[12]:


tt = [0.056, 0.057, 0.122, 0.225, 0.309, 0.082, 0.093, 0.121, 0.136, 0.111, 0.056, 0.215, 0.1, 0.085]
plt.bar(x=list(range(0,28,2)),height=tt,color='darkred')


# In[100]:


#combined rank metric

#OLD CANDIDATES LIST:
#FOR POINT PERTURBATION OF MOTIFS, FOR TARGETTED NON-LIN GRAPHS
#Candidates: Smpx, Csrp3 (3rd iso), Myom3, Myom2 (1st&2nd iso), Cmya5, Ktn1, Ltk, Rbp3, Kcnq1,
#            Rab6b, Tnni3k, Myh6, Atpaf2, Apbb1, Pank4, Mylk3, Col17a1, Smim8, Tmem67, Moap1,
#            Dnajb5, Pld6, Ank1, Synpo2l, Pla2g5, Acot2, Pgr, Rnaseh2b, Kcnip2, Syngr1, Fignl1,
#            Sall3 (1st iso), Sall2, Glp1r, Hhatl, Kyat1, Rwdd2b, Ptpn20, Kcnj12, Ppp1r9a, 
#            Taco1, Rbm19, Obscn, *Ddo, *Cryab, Lgr6, Fry, Col8a2, Ky, Kcna4, Atp5b, Clgn,
#            Pam, Atg4a, Fsd2, Dnajc28, Ktn1, Hand1, Srpk3, Hsp90aa1, Mlip, Gck, Paqr9, Adamts8,
#            Rab6b, Cep131, Dbi, Hcn4, Col6a6, Smim5, Myh6, Cav3, Adam11, Alpk3, Trim45, 

#HIGH delta-mu Candidates: *Csrp3 (3rd iso), ~Dmtn (1st iso), *Rbp3, ~Atpaf2, ~Dnajb5 (1st iso)
#HIGH delta-mu Candidates: *Pld6, ~Ank1, ~Synop2l, ~Pla2g5, Acot2, *Pgr(*1st, ~2nd), ~Rnaseh2b (1st)

#scan motifs for effect with a given graph
#nine proposed motifs from olson paper + jaspar
# motif = 'AGATAA' #gata4 IN
# motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
motif = 'AGGTGT'  #Tbx5 IN
# motif = 'AAAAAA'
for motif_id,rc in enumerate(hsrc_names):
    if motif in rc:
        motif_idx = motif_id

giq = 'Ttn'
whichg = genel.index(giq)
print(whichg)
whichg=3379
mnum=1122 #for full dataset
    
gscann,gscang,gscanh,gscanm,gscant = [],[],[],[],[]
for whichm in range(mnum): #for full dataset
# for whichg in range(400): #for gmf
    gscann.append(nl[whichg*mnum+whichm])
    gscang.append(gl[whichg*mnum+whichm])
    gscanh.append(hl[whichg*mnum+whichm])
    gscanm.append(ml[whichg*mnum+whichm])
    gscant.append(tl[whichg*mnum+whichm])
    
mscann,mscang,mscanh,mscanm,mscant = [],[],[],[],[]
for whichgind in range(len(genel)): #for full dataset
# for whichg in range(400): #for gmf
    mscann.append(nl[whichgind*mnum+motif_idx])
    mscang.append(gl[whichgind*mnum+motif_idx])
    mscanh.append(hl[whichgind*mnum+motif_idx])
    mscanm.append(ml[whichgind*mnum+motif_idx])
    mscant.append(tl[whichgind*mnum+motif_idx])

print('N:',gscann[motif_idx],'   \tPOSG:',pos(gscann,gscann[motif_idx]),'   \tPOSM:',pos(mscann,mscann[whichg]),'\n')
print('G:',gscang[motif_idx],'   \tPOSG:',pos(gscang,gscang[motif_idx]),'   \tPOSM:',pos(mscang,mscang[whichg]),'\n')
print('H:',gscanh[motif_idx],'   \tPOSG:',pos(gscanh,gscanh[motif_idx]),'   \tPOSM:',pos(mscanh,mscanh[whichg]),'\n')
print('M:',gscanm[motif_idx],'   \tPOSG:',pos(gscanm,gscanm[motif_idx]),'   \tPOSM:',pos(mscanm,mscanm[whichg]),'\n')
print('T:',gscant[motif_idx],'   \tPOSG:',pos(gscant,gscant[motif_idx]),'   \tPOSM:',pos(mscant,mscant[whichg]),'\n')


# In[8]:


#load in kall for matrix-making
kall = pkl.load(open('dln_data_all.pkl','rb'))
#flatten kall to lists of all synth.pert.s of all graphs per perturbation category
nl = [motif for graph in kall[0] for motif in graph]
gl = [motif for graph in kall[1] for motif in graph]
hl = [motif for graph in kall[2] for motif in graph]
ml = [motif for graph in kall[3] for motif in graph]
tl = [motif for graph in kall[4] for motif in graph]


# In[25]:


#make matrix (motifs X graphs) populated by the maximum position score of (motif percentile)*(graph percentile) across all 5 pert predictions (N,G,H,M,T)
motif_span,graph_span,pert_span = len(hsrc_names),len(ghl),len(kall)
hm = torch.zeros((pert_span,motif_span,graph_span))
#scan by motif, create distribution to compare
for m in range(motif_span):
    motif_dist = torch.empty(pert_span,graph_span) 
    for p in range(pert_span):
        motif_dist[p] = Tensor([motif_array[m] for motif_array in kall[p]])
    #scan by graph, create distribution to compare
    for g in range(graph_span):
        graph_dist = torch.empty(pert_span,motif_span)
        for p in range(pert_span):
            graph_dist[p] = Tensor([kall[p][g]])
        #populate hm
        for p in range(pert_span):
            hm[p,m,g] = pos(motif_dist[p],motif_dist[p][g].item())*pos(graph_dist[p],graph_dist[p][m].item())


# In[26]:


torch.save(hm,'sps_matrix.pt')


# In[3]:


yur = torch.load('sps_matrix.pt')


# In[40]:


# motif = 'AGATAA' #gata4 IN
# motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
motif = 'AGGTGT'  #Tbx5 IN
# motif = 'AAAAAA'
for motif_id,rc in enumerate(hsrc_names):
    if motif in rc:
        motif_idx = motif_id
print(motif_idx)


# In[4]:


yurbool = torch.where(yur[1]>7644.2,True,False)
keeplist = [yurbool[:,col].any() for col in range(len(yurbool[0]))]
keeplist = Tensor(keeplist)>0
print(sum(keeplist))
suby = yur[1][:,keeplist]
start = dtime()

hidend = sb.clustermap(Tensor.numpy(suby),figsize=(35,25),z_score=0,cmap='gnuplot2')
print(dtime() - start)

hirerow = hidend.dendrogram_row.reordered_ind
hirecol = hidend.dendrogram_col.reordered_ind


# In[42]:


# motif = 'AGATAA' #gata4 IN
motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
# motif = 'AGGTGT'  #Tbx5 IN
# motif = 'AAAAAA'
for motif_id,rc in enumerate(hsrc_names):
    if motif in rc:
        motif_idx = motif_id
print(motif_idx)


# In[37]:


hirerow.index(motif_idx)/1122 #tbx5


# In[39]:


hirerow.index(motif_idx)/1122 #mef2c


# In[41]:


hirerow.index(motif_idx)/1122 #gata4


# In[43]:


hirerow.index(motif_idx)/1122 #hand2


# In[46]:


#dendro analysis


# In[8]:


#loads in ogid
ogid = pkl.load(open('dendro_ogid.pkl','rb'))
print(ogid.index(1404)/17519*4+3.1719,ogid.index(3154)/17519*4+3.1719)


# In[23]:


len(hirecol),max(hirecol),hirecol[0:5]


# In[28]:


#needlessly inefficient loop for reindexing from hirecol back to tv25 (almost bogosort tier, wow)
ogid = []
for elem in hirecol:
    tid = 0
    for idx,entry in enumerate(keeplist):
        if tid == elem:
            ogid.append(idx)
            break
        if entry:
            tid+=1
    if len(ogid)%1000==0:
        print(len(ogid))


# In[35]:


#saving ogid
with open('dendro_ogid.pkl','wb') as file:
    pkl.dump(ogid,file)


# In[46]:


#end dendro analysis


# In[23]:


giq = 'Tnnt2'
whichg = ghl.index(giq)
print(whichg)


# In[29]:


yurbool = torch.where(yur[1]>7644.2,True,False)
keeplist = [yurbool[:,col].any() for col in range(len(yurbool[0]))]
keeplist = Tensor(keeplist)>0
print(sum(keeplist))
suby = yur[1][:,keeplist]
start = dtime()
hidend = sb.clustermap(Tensor.numpy(suby),figsize=(35,25))
print(dtime() - start)

hirerow = hidend.dendrogram_row.reordered_ind
hirecol = hidend.dendrogram_col.reordered_ind


# In[8]:


yurbool = torch.where(yur[1]<=7644.2,True,False)
keeplist = [sum(yurbool[:,col]) for col in range(len(yurbool[0]))]
keeplist = Tensor(keeplist)>=1122
print(sum(keeplist))
suby = yur[1][:,keeplist]
print(suby.shape)


# In[ ]:


# fig, ax = plt.subplots(figsize=(25,15))
start = dtime()
lodend = sb.clustermap(Tensor.numpy(suby[:,0:-8019]),vmax=10000)
print(dtime() - start)

lorerow = lodend.dendrogram_row.reordered_ind
lorecol = lodend.dendrogram_col.reordered_ind


# In[ ]:


# fig, ax = plt.subplots(figsize=(25,15))
start = dtime()
lodend = sb.clustermap(Tensor.numpy(suby[:,-10019:-1019]),vmax=10000)
print(dtime() - start)

lorerow = lodend.dendrogram_row.reordered_ind
lorecol = lodend.dendrogram_col.reordered_ind


# In[9]:





# In[43]:


# fig, ax = plt.subplots(figsize=(25,15))
start = dtime()
dend = sb.clustermap(Tensor.numpy(suby))
print(dtime() - start)

rerow = dend.dendrogram_row.reordered_ind
recol = dend.dendrogram_col.reordered_ind


# In[42]:


yurbool = torch.where(yur[1]>3000,True,False)
keeplist = [sum(yurbool[:,col]) for col in range(len(yurbool[0]))]
keeplist = Tensor(keeplist)>0
suby = yur[0][:,keeplist]
# fig, ax = plt.subplots(figsize=(25,15))
start = dtime()
dend = sb.clustermap(Tensor.numpy(suby))
print(dtime() - start)

rerow = dend.dendrogram_row.reordered_ind
recol = dend.dendrogram_col.reordered_ind


# In[17]:


# fig, ax = plt.subplots(figsize=(25,15))
start = dtime()
dend = sb.clustermap(Tensor.numpy(suby))
print(dtime() - start)

rerow = dend.dendrogram_row.reordered_ind
recol = dend.dendrogram_col.reordered_ind


# In[24]:


print(recol.index(3379)/len(suby[0]))
print(recol.index(15845)/len(suby[0]))
print(recol.index(14065)/len(suby[0]))
print(recol.index(1404)/len(suby[0]))


# In[40]:


# motif = 'AGATAA' #gata4 IN
# motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
# motif = 'AGGTGT'  #Tbx5 IN
motif = 'AAAAAA'
for motif_id,rc in enumerate(hsrc_names):
    if motif in rc:
        motif_idx = motif_id


# In[33]:


rerow.index(motif_idx)/len(hsrc_names)


# In[35]:


rerow.index(motif_idx)/len(hsrc_names)


# In[37]:


rerow.index(motif_idx)/len(hsrc_names)


# In[39]:


rerow.index(motif_idx)/len(hsrc_names)


# In[41]:


rerow.index(motif_idx)/len(hsrc_names)


# In[197]:


#saving modified graphs
with open('synthg_Pkp1_nog.pkl','wb') as file:
    pkl.dump(pg,file)


# In[18]:


#FOR POINT PERTURBATION OF MOTIFS
rnl,pnl = [],[]
rgl,pgl = [],[]
rhl,phl = [],[]
rml,pml = [],[]
rtl,ptl = [],[]
ndl,gdl,hdl,mdl,tdl = [],[],[],[],[]

start = dtime()

start_graph = 40
end_graph = start_graph + 10
ndll,gdll,hdll,mdll,tdll = [],[],[],[],[]
for x in range(start_graph,end_graph):
    ndl,gdl,hdl,mdl,tdl = [],[],[],[],[]
    gnum = x
    real_graph = tv[gnum]
    # mid = motif_indexer('')
    # for gnum in range(len())
    for mid in range(len(hsrc_names)):
        for reps in range(10):
            for gnum in [gnum]:
        #         real_graph = tv[gnum]
                pg = real_graph.clone()
        #         mid = motif_indexer('')
            #     print(pg['enhancer'].x[:,1+mid])
                pg['enhancer'].x[:,1+mid] = 0
            #     print(pg['enhancer'].x[:,1+mid])
                pert_graph = pg
                corr,preds = sp_predictor(real_graph,pert_graph)

        #         print(get_trin(corr),'CORR')
        #         print(get_trin(preds[0])[0][0:16],'REAL PRED')
        #         print(get_trin(preds[0])[0][16:32],'PERT PRED')
                rpn,ppn = get_trin(preds[0])[0][0],get_trin(preds[0])[0][16]
                rnl.append(rpn.item())
                pnl.append(ppn.item())

                rpg,ppg = get_trin(preds[0])[0][1],get_trin(preds[0])[0][17]
                rgl.append(rpg.item())
                pgl.append(ppg.item())

                rph,pph = get_trin(preds[0])[0][2],get_trin(preds[0])[0][18]
                rhl.append(rph.item())
                phl.append(pph.item())

                rpm,ppm = get_trin(preds[0])[0][3],get_trin(preds[0])[0][19]
                rml.append(rpm.item())
                pml.append(ppm.item())

                rpt,ppt = get_trin(preds[0])[0][4],get_trin(preds[0])[0][20]
                rtl.append(rpt.item())
                ptl.append(ppt.item())
        ndl.append(abs(np.mean(rnl)-np.mean(pnl)))
        gdl.append(abs(np.mean(rgl)-np.mean(pgl)))
        hdl.append(abs(np.mean(rhl)-np.mean(phl)))
        mdl.append(abs(np.mean(rml)-np.mean(pml)))
        tdl.append(abs(np.mean(rtl)-np.mean(ptl)))
    #     if mid%100==0:
    #         print(mid)
    ndll.append(ndl),gdll.append(gdl),hdll.append(hdl),mdll.append(mdl),tdll.append(tdl)
    if x%10==0:
        print(x)
with open('singles_dll_'+str(start_graph)+'to'+str(end_graph)+'.pkl','wb') as file:
    pkl.dump((ndll,gdll,hdll,mdll,tdll),file)
rt = dtime() - start
print('10 iter took ',rt)


# In[35]:


#FOR POINT PERTURBATION OF MOTIFS, FOR TARGETTED NON-LIN GRAPHS
#Candidates: Smpx, Csrp3 (3rd iso), Myom3, Myom2 (1st&2nd iso), Cmya5, Ktn1, Ltk, Rbp3, Kcnq1,
#            Rab6b, Tnni3k, Myh6, Atpaf2, Apbb1, Pank4, Mylk3, Col17a1, Smim8, Tmem67, Moap1,
#            Dnajb5, Pld6, Ank1, Synpo2l, Pla2g5, Acot2, Pgr, Rnaseh2b, Kcnip2, Syngr1, Fignl1,
#            Sall3 (1st iso), Sall2, Glp1r, Hhatl, Kyat1, Rwdd2b, Ptpn20, Kcnj12, Ppp1r9a, 
#            Taco1, Rbm19, Obscn, *Ddo, *Cryab, Lgr6, Fry, Col8a2, Ky, Kcna4, Atp5b, Clgn,
#            Pam, Atg4a, Fsd2, Dnajc28, Ktn1, Hand1, Srpk3, Hsp90aa1, Mlip, Gck, Paqr9, Adamts8,
#            Rab6b, Cep131, Dbi, Hcn4, Col6a6, Smim5, Myh6, Cav3, Adam11, Alpk3, Trim45, 
#
#HIGH delta-mu Candidates: *Csrp3 (3rd iso), ~Dmtn (1st iso), *Rbp3, ~Atpaf2, ~Dnajb5 (1st iso)
#HIGH delta-mu Candidates: *Pld6, ~Ank1, ~Synop2l, ~Pla2g5, Acot2, *Pgr(*1st, ~2nd), ~Rnaseh2b (1st)
#HIGH delta-mu Candidates: 


# In[6]:


#load in adata
adata = sc.read('adata6_annotated.h5ad')
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


# In[7]:


#for single gene annotation
gl = pkl.load(open('hl_gene_names.pkl','rb')) # + ['synthetic_gene']*len(synthlist)
#can dwd
giq = 'Myom3'
fgl = []
for elem in gl:
    fgl.append((elem==giq)*len(set(fgl)))
adata.obs[giq] = fgl


# In[8]:


adata[adata.obs['gene']==giq].obs


# In[ ]:


#define proper graph subset, want non-lin?, sensitive, cardiac gene, not highly or reliably expressed?
#current code block takes about 26mins to check each gene at 10iter depth :(
fig, ax = plt.subplots(figsize=(30,20))
sc.pl.tsne(
    adata,
    color=['nlr1'],
    size=350,
    palette = ['darkslategray','darkorchid','crimson','orangered','coral'],
#     vmin=0, vmax= 1,
#     dimensions=(0,1),
    title = '',
#     legend_loc='None',
    ax=ax
)


# In[ ]:


rnl,pnl = [],[]
rgl,pgl = [],[]
rhl,phl = [],[]
rml,pml = [],[]
rtl,ptl = [],[]
ndl,gdl,hdl,mdl,tdl = [],[],[],[],[]

start = dtime()

start_graph = 10000
end_graph = start_graph + 1
ndll,gdll,hdll,mdll,tdll = [],[],[],[],[]
for x in range(start_graph,end_graph):
    print(dtime()-start,'    Graph loop entry')
    ndl,gdl,hdl,mdl,tdl = [],[],[],[],[]
    gnum = x
    real_graph = tv[gnum]
    print(dtime()-start,'    Seq loop entry')
    for mid in range(len(hsrc_names)):
        for reps in range(10):
            for gnum in [gnum]:
                pg = real_graph.clone()
                pg['enhancer'].x[:,1+mid] = 0
                pert_graph = pg
                corr,preds = sp_predictor(real_graph,pert_graph)
                trin_pred = get_trin(preds[0])[0]

                rpn,ppn = trin_pred[0],trin_pred[16]
                if rpn.item() != -13 and ppn.item() != -13:
                    rnl.append(rpn.item())
                    pnl.append(ppn.item())

                rpg,ppg = trin_pred[1],trin_pred[17]
                if rpg.item() != -13 and ppg.item() != -13:
                    rgl.append(rpg.item())
                    pgl.append(ppg.item())

                rph,pph = trin_pred[2],trin_pred[18]
                if rph.item() != -13 and pph.item() != -13:
                    rhl.append(rph.item())
                    phl.append(pph.item())

                rpm,ppm = trin_pred[3],trin_pred[19]
                if rpm.item() != -13 and ppm.item() != -13:
                    rml.append(rpm.item())
                    pml.append(ppm.item())

                rpt,ppt = trin_pred[4],trin_pred[20]
                if rpt.item() != -13 and ppt.item() != -13:
                    rtl.append(rpt.item())
                    ptl.append(ppt.item())
            print(dtime()-start,'    Rep  completion')

        ndl.append(abs(np.mean(rnl)-np.mean(pnl)))
        gdl.append(abs(np.mean(rgl)-np.mean(pgl)))
        hdl.append(abs(np.mean(rhl)-np.mean(phl)))
        mdl.append(abs(np.mean(rml)-np.mean(pml)))
        tdl.append(abs(np.mean(rtl)-np.mean(ptl)))
        print(dtime()-start,'    l appension')
    #     if mid%100==0:
    #         print(mid)
    ndll.append(ndl),gdll.append(gdl),hdll.append(hdl),mdll.append(mdl),tdll.append(tdl)
    if x%10==0:
        print(x)
# with open('singles_dll_'+giq+'_alliso'+'.pkl','wb') as file:
#     pkl.dump((ndll,gdll,hdll,mdll,tdll),file)
rt = dtime() - start
print('1 iter took ',rt)


# In[ ]:


#DWD
with open('singles_dll_'+giq+'_iso3_10iter'+'.pkl','wb') as file:
    pkl.dump((ndll,gdll,hdll,mdll,tdll),file)


# In[ ]:


# giqdll = pkl.load(open('singles_dll_'+giq+'_alliso'+'.pkl','rb'))
giqdll = pkl.load(open('singles_dll_'+giq+'_iso3_10iter'+'.pkl','rb'))
ndll,gdll,hdll,mdll,tdll = giqdll[0],giqdll[1],giqdll[2],giqdll[3],giqdll[4]


# In[ ]:


x = 0
max(ndll[x]),max(gdll[x]),max(hdll[x]),max(mdll[x]),max(tdll[x])


# In[ ]:


max(ndll[x][2:]),max(gdll[x][2:]),max(hdll[x][2:]),max(mdll[x][2:]),max(tdll[x][2:])


# In[ ]:


sb.histplot(ndll[x][2:],color='brown'),sb.histplot(gdll[x][2:],color='red'),sb.histplot(hdll[x][2:],color='gold'),sb.histplot(mdll[x][2:],color='green'),sb.histplot(tdll[x][2:],color='blue')


# In[36]:


#motif reference:
#G: m&h  cTTATCT   
#H: h    CAGATG
#M: h    aAAATAg
#T: h    AGGTGT
for idx,val in enumerate(gdl):
    if val>1:
        print(hsrc_names[idx],'\t',val)
    if 'TTATCT' in hsrc_names[idx]:
        print(val)
    if 'CAGATG' in hsrc_names[idx]:
        print(val)
    if 'AAAATA' in hsrc_names[idx]:
        print(val)
    if 'AGGTGT' in hsrc_names[idx]:
        print(val)


# In[12]:


sb.histplot(ndl,color='brown'),sb.histplot(gdl,color='red'),sb.histplot(hdl,color='yellow'),sb.histplot(mdl,color='green'),sb.histplot(tdl,color='blue')


# In[11]:


# print(np.sum([elem==2 for elem in rtl]),np.sum([elem==2 for elem in ptl]))
sb.histplot(rnl,color='green',stat='probability'),sb.histplot(pnl,color='red',stat='probability')


# In[12]:


# print(np.sum([elem==2 for elem in rgl]),np.sum([elem==2 for elem in pgl]))
sb.histplot(rgl,color='green',stat='probability'),sb.histplot(pgl,color='red',stat='probability')


# In[13]:


# print(np.sum([elem==2 for elem in rgl]),np.sum([elem==2 for elem in pgl]))
sb.histplot(rhl,color='green',stat='probability'),sb.histplot(phl,color='red',stat='probability')


# In[14]:


# print(np.sum([elem==2 for elem in rgl]),np.sum([elem==2 for elem in pgl]))
sb.histplot(rml,color='green',stat='probability'),sb.histplot(pml,color='red',stat='probability')


# In[15]:


# print(np.sum([elem==2 for elem in rtl]),np.sum([elem==2 for elem in ptl]))
sb.histplot(rtl,color='green',stat='probability'),sb.histplot(ptl,color='red',stat='probability')


# In[110]:


#motif reference:
#G: m&h  cTTATCT   
#H: h    CAGATG
#M: h    aAAATAg
#T: h    AGGTGT


# In[110]:


#acc distribution checker
acl = []
agl = []
for gid,graph in enumerate(tv):
    data = graph
    out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
    ot = out['perturbation'].sum(dim=0)
    oe = out['enhancer'].sum(dim=0)
    op = out['promoter'].sum(dim=0)
    tp_pool_pergraph = ot+oe+op
    tp_pool_pergraph = tp_pool_pergraph.reshape(1,32)
    pred = tp_pool_pergraph
    pred = torch.where(pred>=0,1,0)
    correct = graph.y
    pred1 = get_trin(pred)
    inten = correct == pred1
    acc = inten/16
    acl.append(acc)
    if gid%3500==0:
        print(gid)
    out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
    ot = out['perturbation'].sum(dim=0)
    oe = out['enhancer'].sum(dim=0)
    op = out['promoter'].sum(dim=0)
    tp_pool_pergraph = ot+oe+op
    tp_pool_pergraph = tp_pool_pergraph.reshape(1,32)
    pred = tp_pool_pergraph
    pred = torch.where(pred>=0,1,0)
    pred2 = get_trin(pred)
    agrtin = pred1==pred2
    agl.append(agrtin)
acl = [elem.sum().item() for elem in acl]
agl = [elem.sum().item()/16 for elem in agl]


# In[101]:


sb.histplot(acl)


# In[112]:


sb.histplot(acl,stat='probability')


# In[111]:


sb.histplot(agl,stat='probability')


# In[108]:


agl = [elem/16 for elem in agl]


# In[ ]:


#rm no e
sb.histplot(agl,stat='probability')


# ## FILE STITCHING

# In[2]:


#gmf 100 joiner
kfiles = ['dln_gmf_data_0to99.pkl','dln_gmf_data_100to199.pkl','dln_gmf_data_200to299.pkl','dln_gmf_data_300to399.pkl']
klists = [pkl.load(open(file,'rb')) for file in kfiles]
kfull = [[],[],[],[],[]]
for klist in klists:
    kfull[0]+=klist[0]
    kfull[1]+=klist[1]
    kfull[2]+=klist[2]
    kfull[3]+=klist[3]
    kfull[4]+=klist[4]
with open('dln_gmf_data_all.pkl','wb') as file:
    pkl.dump(kfull,file)


# In[9]:


#1k joiner
k=str(10)
kfiles = ['dln_data_'+k+str(h)+'00to'+k+str(h)+'99.pkl' for h in range(10)]
klists = [pkl.load(open(file,'rb')) for file in kfiles]
kfull = [[],[],[],[],[]]
for klist in klists:
    kfull[0]+=klist[0]
    kfull[1]+=klist[1]
    kfull[2]+=klist[2]
    kfull[3]+=klist[3]
    kfull[4]+=klist[4]
with open('dln_data'+k+'k.pkl','wb') as file:
    pkl.dump(kfull,file)


# In[58]:


#1k joiner
for which_k in range(10):
    k=str(which_k)
    kfiles = ['dln_data_'+k+str(h)+'00to'+k+str(h)+'99.pkl' for h in range(10)]
    klists = [pkl.load(open(file,'rb')) for file in kfiles]
    kfull = [[],[],[],[],[]]
    for klist in klists:
        kfull[0]+=klist[0]
        kfull[1]+=klist[1]
        kfull[2]+=klist[2]
        kfull[3]+=klist[3]
        kfull[4]+=klist[4]
    with open('dln_data'+k+'k.pkl','wb') as file:
        pkl.dump(kfull,file)


# In[8]:


#10k joiner
dk=str(0)
kfiles = ['dln_data'+str(k)+'k.pkl' for k in range(10)]
klists = [pkl.load(open(file,'rb')) for file in kfiles]
kfull = [[],[],[],[],[]]
for klist in klists:
    kfull[0]+=klist[0]
    kfull[1]+=klist[1]
    kfull[2]+=klist[2]
    kfull[3]+=klist[3]
    kfull[4]+=klist[4]
with open('dln_data'+dk+'dk.pkl','wb') as file:
    pkl.dump(kfull,file)


# ## RUNNING SCREENS

# In[101]:


kall = pkl.load(open('dln_data_all.pkl','rb'))
# kall = pkl.load(open('dln_gmf_data_all.pkl','rb'))


# In[102]:


#flatten kall to lists of all synth.pert.s of all graphs per perturbation category
nl = [motif for graph in kall[0] for motif in graph]
gl = [motif for graph in kall[1] for motif in graph]
hl = [motif for graph in kall[2] for motif in graph]
ml = [motif for graph in kall[3] for motif in graph]
tl = [motif for graph in kall[4] for motif in graph]


# In[3]:


nl = [elem for elem in nl if elem>0.2]
gl = [elem for elem in gl if elem>0.2]
hl = [elem for elem in hl if elem>0.2]
ml = [elem for elem in ml if elem>0.2]
tl = [elem for elem in tl if elem>0.2]


# In[24]:


# with open('qsp_outputs.txt','a') as file:
#     file.write('NL:'+str(len(nl))+'\tGL:'+str(len(gl))+'\tHL:'+str(len(hl))+'\tML:'+str(len(ml))+'\tTL:'+str(len(tl)))

out_string = []
for pert_cat in kall:
    out_string.append('\nSTART OF NEW PERT_CAT________________________________________________\n')
    for gnum,graph in enumerate(pert_cat):
        for mnum,motif in enumerate(graph):
            if motif>0.2:
                out_string[-1] += '\ngraph_id:'+str(gnum)+'\tmotif_id:'+str(mnum)+'\tvalue:'+str(motif)

# with open('qsp_outputs.txt','a') as file:
#     file.write(out_string[0]+out_string[1]+out_string[2]+out_string[3]+out_string[4])


# In[25]:


out_string[0]


# In[4]:


len(nl),len(gl),len(hl),len(ml),len(tl)


# In[5]:


print(gl[0::100])


# In[142]:


len(kall[0][0])


# In[4]:


genel = pkl.load(open('hl_gene_names.pkl','rb'))
hsrc_names = pkl.load(open('hsrc_names.pkl','rb'))


# In[29]:


giq = 'Ttn'
whichg = genel.index(giq)
print(whichg)
# whichg = 731
# motif = 'AGATAA' #gata4
# motif = 'CAGATG'  #hand2
# motif = 'TAAAAA' #Mef2c
# motif = 'AAAATA' #Mef2c
# motif = 'AAAAAT' #Mef2c
# motif = 'GGTGTT'  #Tbx5
# motif = 'AGGTGT'  #Tbx5
for motif_id,rc in enumerate(hsrc_names):
    if motif in rc:
        motif_idx = motif_id
mnum=1122
print(nl[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
print(gl[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
print(hl[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
print(ml[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
print(tl[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
# print(max(nl[whichg*mnum+0:whichg*mnum+1122]))
# print(max(gl[whichg*mnum+0:whichg*mnum+1122]))
# print(max(hl[whichg*mnum+0:whichg*mnum+1122]))
# print(max(ml[whichg*mnum+0:whichg*mnum+1122]))
# print(max(tl[whichg*mnum+0:whichg*mnum+1122]))


# In[57]:


#scan graphs for effect with a given motif
#nine proposed motifs from olson paper + jaspar
# motif = 'AGATAA' #gata4 IN
# motif = 'GATAAG' #gata4
# motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
# motif = 'AAATAG' #Mef2c
motif = 'AGGTGT'  #Tbx5 IN
# motif = 'GGTGTT'  #Tbx5 
mnum=1122 #for full dataset
# mnum = 100 #for gmf
for motif_id,rc in enumerate(hsrc_names):
    if motif in rc:
        motif_idx = motif_id
#         motif_idx = motif_id+1 #offset for gmf 
# motif_idx = 0 #manual for gata motif in gmf
    
gscann,gscang,gscanh,gscanm,gscant = [],[],[],[],[]
for whichg in range(35038): #for full dataset
# for whichg in range(400): #for gmf
    gscann.append(nl[whichg*mnum+motif_idx])
    gscang.append(gl[whichg*mnum+motif_idx])
    gscanh.append(hl[whichg*mnum+motif_idx])
    gscanm.append(ml[whichg*mnum+motif_idx])
    gscant.append(tl[whichg*mnum+motif_idx])
# print(nl[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
# print(gl[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
# print(hl[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
# print(ml[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])
# print(tl[whichg*mnum+motif_idx:whichg*mnum+motif_idx+10])


# In[118]:


# sb.histplot(lgn,log_scale=True,color='black')
sb.histplot(lgg,log_scale=True,color='red')
# sb.histplot(lgh,log_scale=True,color='yellow')
# sb.histplot(lgm,log_scale=True,color='green')
sb.histplot(lgt,log_scale=True,color='blue')


# In[49]:


gscang[3375],gscang[3379]


# In[58]:


giq = 'Myom2'
whichg = genel.index(giq)
print(whichg)
print(genel[whichg-1:whichg+10])


# In[60]:


hits4 = []
gscan = [gscann,gscang,gscanh,gscanm,gscant]
for scan in gscan:
    print('YUR_____________________________________________________________________________')
    for gid,elem in enumerate(scan):
        if elem>0.05:
            print(genel[gid])
            hits4.append(genel[gid])
            if genel[gid] == 'Myom2':
                print('___FOUND IT___!!!!!!!!!!!!!!!!!!!!!!')
                print(gid)
                
# hits3 = []
# gscan = [gscann,gscang,gscanh,gscanm,gscant]
# for scan in gscan:
#     print('YUR_____________________________________________________________________________')
#     for gid,elem in enumerate(scan):
#         if elem>0.05:
#             print(genel[gid])
#             hits3.append(genel[gid])
#             if genel[gid] == 'Myom2':
#                 print('___FOUND IT___!!!!!!!!!!!!!!!!!!!!!!')
                
# hits2 = []
# gscan = [gscann,gscang,gscanh,gscanm,gscant]
# for scan in gscan:
#     print('YUR_____________________________________________________________________________')
#     for gid,elem in enumerate(scan):
#         if elem>0.03:
#             print(genel[gid])
#             hits2.append(genel[gid])
#             if genel[gid] == 'Myom2':
#                 print('___FOUND IT___!!!!!!!!!!!!!!!!!!!!!!')
                
# hits = []
# gscan = [gscann,gscang,gscanh,gscanm,gscant]
# for scan in gscan:
#     print('YUR_____________________________________________________________________________')
#     for gid,elem in enumerate(scan):
#         if elem>0.05:
#             print(genel[gid])
#             hits.append(genel[gid])
#             if genel[gid] == 'Myom2':
#                 print('___FOUND IT___!!!!!!!!!!!!!!!!!!!!!!')


# In[24]:


#OLD CANDIDATES LIST:
#FOR POINT PERTURBATION OF MOTIFS, FOR TARGETTED NON-LIN GRAPHS
#Candidates: Smpx, Csrp3 (3rd iso), Myom3, Myom2 (1st&2nd iso), Cmya5, Ktn1, Ltk, Rbp3, Kcnq1,
#            Rab6b, Tnni3k, Myh6, Atpaf2, Apbb1, Pank4, Mylk3, Col17a1, Smim8, Tmem67, Moap1,
#            Dnajb5, Pld6, Ank1, Synpo2l, Pla2g5, Acot2, Pgr, Rnaseh2b, Kcnip2, Syngr1, Fignl1,
#            Sall3 (1st iso), Sall2, Glp1r, Hhatl, Kyat1, Rwdd2b, Ptpn20, Kcnj12, Ppp1r9a, 
#            Taco1, Rbm19, Obscn, *Ddo, *Cryab, Lgr6, Fry, Col8a2, Ky, Kcna4, Atp5b, Clgn,
#            Pam, Atg4a, Fsd2, Dnajc28, Ktn1, Hand1, Srpk3, Hsp90aa1, Mlip, Gck, Paqr9, Adamts8,
#            Rab6b, Cep131, Dbi, Hcn4, Col6a6, Smim5, Myh6, Cav3, Adam11, Alpk3, Trim45, 
#
#HIGH delta-mu Candidates: *Csrp3 (3rd iso), ~Dmtn (1st iso), *Rbp3, ~Atpaf2, ~Dnajb5 (1st iso)
#HIGH delta-mu Candidates: *Pld6, ~Ank1, ~Synop2l, ~Pla2g5, Acot2, *Pgr(*1st, ~2nd), ~Rnaseh2b (1st)
#HIGH delta-mu Candidates: 


#scan motifs for effect with a given graph
#nine proposed motifs from olson paper + jaspar
# motif = 'AGATAA' #gata4 IN
# motif = 'GATAAG' #gata4
# motif = 'CAGATG'  #hand2 IN
# motif = 'CAGCTG'  #hand2 IN
# motif = 'AAAATA' #Mef2c IN
# motif = 'AAAAAT' #Mef2c IN
# motif = 'AAATAG' #Mef2c
motif = 'AGGTGT'  #Tbx5 IN
# motif = 'GGTGTT'  #Tbx5 
# motif = 'AAAAAA'
for motif_id,rc in enumerate(hsrc_names):
    if motif in rc:
        motif_idx = motif_id

giq = 'Atpaf2'
whichg = genel.index(giq)
print(whichg)
# whichg=14065
mnum=1122 #for full dataset
    
gscann,gscang,gscanh,gscanm,gscant = [],[],[],[],[]
for whichm in range(mnum): #for full dataset
# for whichg in range(400): #for gmf
    gscann.append(nl[whichg*mnum+whichm])
    gscang.append(gl[whichg*mnum+whichm])
    gscanh.append(hl[whichg*mnum+whichm])
    gscanm.append(ml[whichg*mnum+whichm])
    gscant.append(tl[whichg*mnum+whichm])
    
mscann,mscang,mscanh,mscanm,mscant = [],[],[],[],[]
for whichgind in range(len(genel)): #for full dataset
# for whichg in range(400): #for gmf
    mscann.append(nl[whichgind*mnum+motif_idx])
    mscang.append(gl[whichgind*mnum+motif_idx])
    mscanh.append(hl[whichgind*mnum+motif_idx])
    mscanm.append(ml[whichgind*mnum+motif_idx])
    mscant.append(tl[whichgind*mnum+motif_idx])

print('N:',gscann[motif_idx],'   \tPOSG:',pos(gscann,gscann[motif_idx]),'   \tPOSM:',pos(mscann,mscann[whichg]),'\n')
print('G:',gscang[motif_idx],'   \tPOSG:',pos(gscang,gscang[motif_idx]),'   \tPOSM:',pos(mscang,mscang[whichg]),'\n')
print('H:',gscanh[motif_idx],'   \tPOSG:',pos(gscanh,gscanh[motif_idx]),'   \tPOSM:',pos(mscanh,mscanh[whichg]),'\n')
print('M:',gscanm[motif_idx],'   \tPOSG:',pos(gscanm,gscanm[motif_idx]),'   \tPOSM:',pos(mscanm,mscanm[whichg]),'\n')
print('T:',gscant[motif_idx],'   \tPOSG:',pos(gscant,gscant[motif_idx]),'   \tPOSM:',pos(mscant,mscant[whichg]),'\n')


# In[92]:


pos(range(10),8)


# In[81]:


fig, ax = plt.subplots(figsize=(30,10))
sb.histplot(gscang)


# In[79]:





# In[84]:


nzn = [elem for elem in gscann if elem>0]
nzg = [elem for elem in gscang if elem>0]
nzh = [elem for elem in gscanh if elem>0]
nzm = [elem for elem in gscanm if elem>0]
nzt = [elem for elem in gscant if elem>0]
# plt.violinplot(nzg)
plt.violinplot(nzh)
# plt.violinplot(nzm)
plt.violinplot(nzt)


# In[51]:


len(nzn),len(nzg),len(nzh),len(nzm),len(nzt)


# In[45]:


plt.violinplot(gscann),plt.violinplot(gscang),plt.violinplot(gscanh),plt.violinplot(gscanm),plt.violinplot(gscant)

