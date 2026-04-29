
# coding: utf-8

# In[1]:


#optimize for neural net running, including modular architecture and graph inclusion

import numpy as np
import scanpy as sc
import pandas as pd
import pickle as pkl
import os
import seaborn as sb
import scipy.sparse as csr
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolo
import sklearn.linear_model as skl
from scipy.stats import gaussian_kde
import scipy.stats as ss
import math
from timeit import default_timer as dtime
import multiprocessing
from multiprocessing import Pool
from scipy.stats import percentileofscore
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
from torch_geometric.nn import GCNConv, GraphConv, ResGatedGraphConv, SAGEConv, pool, to_hetero, to_hetero_with_bases, TransformerConv
import torch_geometric.transforms as T

# set a working directory for saving plots
os.chdir('/project/GCRB/Hon_lab/s437603/data/ghmt_multiome/analysis')


# ## LOAD IN FILES

# In[2]:


#version check
print(sc.__version__)
print(pyg.__version__)


# In[3]:


import doubletdetection as dd
print(dd.__version__)


# In[2]:


# tv10 = torch.load('pyg_hetlist_tv10.pt')
# tv25 = torch.load('pyg_hetlist_tv25.pt')
# tv25 = torch.load('pyg_hetlist_tv25_6merrc.pt')
tv25 = torch.load('pyg_hetlist_tv25_6merrchs.pt')
# tv25 = torch.load('pyg_hetlist_tv25_ran.pt')
# tv25 = torch.load('pyg_hetlist_tv25_gmf.pt')
# tv25 = torch.load('pyg_hetlist_tv25_rannopa.pt')
# tv50 = torch.load('pyg_hetlist_tv50.pt')
# tv100 = torch.load('pyg_hetlist_tv100.pt')
# hetl = torch.load('pyg_hetlist_predf3_trin25_co.pt')


# ## DATA PREP and UTILS

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


# In[4]:


#NEW heterodata prep and loading
fulllist = tv25
mask = pkl.load(open('newbtrainmask.pkl','rb'))
testlist = [fulllist[ind] for ind in range(len(fulllist)) if mask[ind]]
trainlist = [fulllist[ind] for ind in range(len(fulllist)) if not mask[ind]]
#loaderize datasets
fullloader = pyg.loader.DataLoader(fulllist,shuffle=False)
trainloader = pyg.loader.DataLoader(trainlist[::50],shuffle=False)
testloader = pyg.loader.DataLoader(testlist,shuffle=False)


# ## hetero GraphConv model

# In[ ]:


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


# In[ ]:


#for running existing model
# model = CPU_unpickler(open('hg4_48n_iter930.pkl','rb')).load()
# model = CPU_unpickler(open('hg6_64ndo25ll_iter5000.pkl','rb')).load()   #emb7
# model = CPU_unpickler(open('hg5_udep_hsb_iter760.pkl','rb')).load()     #emb6
# model = CPU_unpickler(open('hg5_ran_iter1920.pkl','rb')).load()     #emb6_ran
# model = CPU_unpickler(open('hg5_gmf_iter1920.pkl','rb')).load()     #emb6_gmf
model = CPU_unpickler(open('hg5_rannopa_iter1960.pkl','rb')).load()     #emb6_rannopa


# In[ ]:


model.eval()
#for start metrics, not essential
losslist,timelist = [],[]
#end metrics
raw_emb = []
for epoch in range(0,1):
    start = dtime()
    count=0
    for data in fullloader:
        out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
        pool = torch.cat((out['perturbation'].sum(dim=0),out['enhancer'].sum(dim=0),out['promoter'].sum(dim=0)))
        raw_emb.append(pool.reshape((1,-1)))
        count+=1
        if count%5000==0:
            print(count)
            print(dtime()-start)
#         print(out['perturbation'].reshape((1,-1)))
#         tp_pool = out['perturbation'].sum(dim=0)+out['promoter'].sum(dim=0)
#         pred = sig(tp_pool.reshape(1,32))
#         correct = get_oh(data['y'])
#         loss = multilabel(pred,correct)
#         epoch_loss+=loss
#         break

with open('raw_emb6_rannopa.pkl','wb') as f:
    pkl.dump(raw_emb,f)


# In[8]:


model.eval()
#for start metrics, not essential
losslist,timelist = [],[]
#end metrics
raw_emb = []
for epoch in range(0,1):
    start = dtime()
    count=0
    for data in fullloader:
        out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
        pool = torch.cat((out['perturbation'].sum(dim=0),out['promoter'].sum(dim=0)))
        raw_emb.append(pool.reshape((1,-1)))
        count+=1
        if count%5000==0:
            print(count)
            print(dtime()-start)
#         print(out['perturbation'].reshape((1,-1)))
#         tp_pool = out['perturbation'].sum(dim=0)+out['promoter'].sum(dim=0)
#         pred = sig(tp_pool.reshape(1,32))
#         correct = get_oh(data['y'])
#         loss = multilabel(pred,correct)
#         epoch_loss+=loss
#         break

with open('raw_emb7noe.pkl','wb') as f:
    pkl.dump(raw_emb,f)


# ## Model Loading and Testing

# In[12]:


#load in hb mask,     needed for htb and beyond
#NEW heterodata prep and loading
fulllist = hetlist
mask = pkl.load(open('htbtrainmask.pkl','rb'))
testlist = [fulllist[ind] for ind in range(len(fulllist)) if mask[ind]]
trainlist = [fulllist[ind] for ind in range(len(fulllist)) if not mask[ind]]

#loaderize datasets
fullloader = pyg.loader.DataLoader(fulllist,shuffle=True)
trainloader = pyg.loader.DataLoader(trainlist,shuffle=True)
testloader = pyg.loader.DataLoader(testlist,shuffle=True)

mll = []
mll.append([pkl.load(open('ht2_e55pl_bmask_iter'+str(which_iter)+'.pkl','rb')) for which_iter in range(0,68,1)])
mll.append([pkl.load(open('htp_bmask_iter'+str(which_iter)+'.pkl','rb')) for which_iter in range(0,75,1)])


# In[5]:


mll = []
mll.append([pkl.load(open('ht2_e55pl_iter'+str(which_iter)+'.pkl','rb')) for which_iter in range(0,261,1)])
# mll.append([pkl.load(open('ht2_e55pl_bmask_iter'+str(which_iter)+'.pkl','rb')) for which_iter in range(0,24,1)])
# mll.append([pkl.load(open('ht3base_iter'+str(which_iter)+'.pkl','rb')) for which_iter in range(0,121,1)])
mll.append([pkl.load(open('htp_iter'+str(which_iter)+'.pkl','rb')) for which_iter in range(0,261,1)])
# mll.append([pkl.load(open('htp_bmask_iter'+str(which_iter)+'.pkl','rb')) for which_iter in range(0,103,1)])


# In[30]:


#for getting emb accuracy scores
model = CPU_unpickler(open('hg5_udep_hsbb_iter2000.pkl','rb')).load()
model.eval()
losslist,timelist = [],[]
start = dtime()
count=0

accl = []
for data in fullloader:
    out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
    pool = out['perturbation'].sum(dim=0)+out['enhancer'].sum(dim=0)+out['promoter'].sum(dim=0)
    pred = get_trin(sig(pool.reshape(1,32)))[0]
    correct = data['y']
    acc = np.sum(list(pred==correct))/16
    accl.append(acc)
    
    count+=1
    if count%5000==0:
        print(count)
        print(dtime()-start)


# ## UMAP

# In[5]:


# model = CPU_unpickler(open('hg3_32ndo15w_iter65.pkl','rb')).load()
# raw_emb = pkl.load(open('raw_emb1.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb1noe.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb2noe.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb3noe.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb5noe.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb5.pkl','rb'))
raw_emb = pkl.load(open('raw_emb6.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb6_ran.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb6_gmf.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb6_rannopa.pkl','rb'))
# raw_emb = pkl.load(open('raw_emb7.pkl','rb'))
glist = tv25    #if using all graphs
# glist = testlist    #if using test graphs


# In[8]:


#optional, if adding synth_emb

#load and loaderize modified graphs

# tolp = pkl.load(open('synthg_tnnt2_ol_pkp1.pkl','rb'))
# tnolp = pkl.load(open('synthg_tnnt2_nol_pkp1.pkl','rb'))
# polt = pkl.load(open('synthg_pkp1_ol_tnnt2.pkl','rb'))
# pnolt = pkl.load(open('synthg_pkp1_nol_tnnt2.pkl','rb'))
# synthlist = [tolp,tnolp,polt,pnolt]
# synthloader = pyg.loader.DataLoader(synthlist,shuffle=False)

#for g motif
# tnnog = pkl.load(open('synthg_Tnnt2_nog.pkl','rb'))
# tng1 = pkl.load(open('synthg_Tnnt2_gp1.pkl','rb'))
# tng5 = pkl.load(open('synthg_Tnnt2_gp5.pkl','rb'))
# tng10 = pkl.load(open('synthg_Tnnt2_gp10.pkl','rb'))
# ttnog = pkl.load(open('synthg_Ttn_nog.pkl','rb'))
# ttg1 = pkl.load(open('synthg_Ttn_gp1.pkl','rb'))
# ttg5 = pkl.load(open('synthg_Ttn_gp5.pkl','rb'))
# ttg10 = pkl.load(open('synthg_Ttn_gp10.pkl','rb'))
# pnog = pkl.load(open('synthg_Pkp1_nog.pkl','rb'))
# pg1 = pkl.load(open('synthg_Pkp1_gp1.pkl','rb'))
# pg5 = pkl.load(open('synthg_Pkp1_gp5.pkl','rb'))
# pg10 = pkl.load(open('synthg_Pkp1_gp10.pkl','rb'))
# synthlist = [tnnog,tng1,tng5,tng10,ttnog,ttg1,ttg5,ttg10,pnog,pg1,pg5,pg10]

#for t motif
tnnot = pkl.load(open('synthg_tnnt2_not.pkl','rb'))
tnt1 = pkl.load(open('synthg_tnnt2_1t.pkl','rb'))
tnt5 = pkl.load(open('synthg_tnnt2_5t.pkl','rb'))
tnt10 = pkl.load(open('synthg_tnnt2_10t.pkl','rb'))
ttnot = pkl.load(open('synthg_ttn_not.pkl','rb'))
ttt1 = pkl.load(open('synthg_ttn_1t.pkl','rb'))
ttt5 = pkl.load(open('synthg_ttn_5t.pkl','rb'))
ttt10 = pkl.load(open('synthg_ttn_10t.pkl','rb'))
synthlist = [tnnot,tnt1,tnt5,tnt10,ttnot,ttt1,ttt5,ttt10]

synthloader = pyg.loader.DataLoader(synthlist,shuffle=False)

#load model and make synth embedding array
model = CPU_unpickler(open('hg5_udep_hsbb_iter2000.pkl','rb')).load()     #emb6 model
model.eval()
synth_emb = []
for epoch in range(0,1):
    start = dtime()
    count=0
    for data in synthloader:
        out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
        pool = torch.cat((out['perturbation'].sum(dim=0),out['enhancer'].sum(dim=0),out['promoter'].sum(dim=0)))
        synth_emb.append(pool.reshape((1,-1)))
        
raw_emb = raw_emb+synth_emb
glist = tv25+synthlist    #if using synth graphs


# In[9]:


arem = [embed.detach().numpy() for embed in raw_emb]
# arem = [embed.detach().numpy() for embed in test_emb]
# datamat = np.reshape(np.array(arem),(-1,64)) #for noe pooling
datamat = np.reshape(np.array(arem),(-1,96)) #for e pooling
num_enh = [graph['enhancer'].x.size()[0] for graph in glist]
adata = sc.AnnData(datamat)
adata.obs['num_enh'] = num_enh


# In[10]:


#do pca for neighbors
sc.tl.pca(adata)

# #remove first PC from PCA plot
# np.shape(adata.obsm['X_pca'])
# adata.obsm['X_pca'] = adata.obsm['X_pca'][:,1:]

#remove trailing empty PCs
print(np.shape(adata.obsm['X_pca']))
pc_sums = [bool(np.sum(adata.obsm['X_pca'][:,which_pc])) for which_pc in range(np.shape(adata.obsm['X_pca'])[1])]
valid_pcs = np.sum(pc_sums)
adata.obsm['X_pca'] = adata.obsm['X_pca'][:,:valid_pcs]
print(np.shape(adata.obsm['X_pca']))


# In[11]:


sc.pp.neighbors(adata,use_rep='X_pca')
sc.tl.louvain(adata,resolution = 1.8)
sc.tl.paga(adata,groups='louvain')

sc.tl.umap(
    adata,
    init_pos='X_pca',
#     min_dist = 0.5,
#     spread = 2,
)
sc.tl.tsne(adata)


# In[12]:


#add obs impromptu
# adata.obs['f4y_fltrin'] = [int(3*adata.obs['f4y'][gnum][0].item()+adata.obs['f4y'][gnum][15].item()) for gnum in range(len(adata))]
adata.obs['nc'] = [graph.y[0].item() for graph in glist]
adata.obs['g'] = [graph.y[1].item() for graph in glist]
adata.obs['h'] = [graph.y[2].item() for graph in glist]
adata.obs['m'] = [graph.y[3].item() for graph in glist]
adata.obs['t'] = [graph.y[4].item() for graph in glist]
adata.obs['gh'] = [graph.y[5].item() for graph in glist]
adata.obs['gm'] = [graph.y[6].item() for graph in glist]
adata.obs['gt'] = [graph.y[7].item() for graph in glist]
adata.obs['hm'] = [graph.y[8].item() for graph in glist]
adata.obs['ht'] = [graph.y[9].item() for graph in glist]
adata.obs['mt'] = [graph.y[10].item() for graph in glist]
adata.obs['ghm'] = [graph.y[11].item() for graph in glist]
adata.obs['ght'] = [graph.y[12].item() for graph in glist]
adata.obs['gmt'] = [graph.y[13].item() for graph in glist]
adata.obs['hmt'] = [graph.y[14].item() for graph in glist]
adata.obs['ghmt'] = [graph.y[15].item() for graph in glist]
gl = pkl.load(open('hl_gene_names.pkl','rb'))
# adata.obs['gene'] = gl
adata.obs['gene'] = gl+['synthetic_gene']*len(synthlist)


# In[13]:


min_tf = []
for graph in glist:
    tf0 = graph.y[0].item() < 2
    tf1 = max(graph.y[1:5]).item() == 2
    tf2 = max(graph.y[5:11]).item() == 2
    tf3 = max(graph.y[11:15]).item() == 2
    tf4 = graph.y[15].item() == 2
    if tf1*tf2*tf3*tf4 and tf0:
        min_tf.append(1)
    elif tf2*tf3*tf4 and tf0:
        min_tf.append(2)
    elif tf3*tf4 and tf0:
        min_tf.append(3)
    elif tf4 and tf0:
        min_tf.append(4)
    else:
        min_tf.append(5)
adata.obs['min_tf'] = min_tf
adata.obs['min_tf'] = pd.Categorical(adata.obs['min_tf'])
sb.histplot(min_tf)


# In[6]:


perts = ['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt']
gset = [perts.index(elem) for elem in perts if perts[1] in elem]
hset = [perts.index(elem) for elem in perts if perts[2] in elem]
mset = [perts.index(elem) for elem in perts if perts[3] in elem]
tset = [perts.index(elem) for elem in perts if perts[4] in elem]
l1,l1g,l1h,l1m,l1t = [],[],[],[],[]
nl1,nl1g,nl1h,nl1m,nl1t = [],[],[],[],[]
for graph in glist:
    lin,lg,lh,lm,lt = 0,0,0,0,0
    nonlin,ng,nh,nm,nt = 0,0,0,0,0
    #g linearity
    if graph.y[1].item() == 2:
        gmem = [graph.y[ind].item() == 2 for ind in gset]
        if np.sum(gmem) == len(gmem):
            lin+=1
            lg+=1
        else:
            nonlin+=1
            ng+=1
    #h linearity
    if graph.y[2].item() == 2:
        gmem = [graph.y[ind].item() == 2 for ind in hset]
        if np.sum(gmem) == len(gmem):
            lin+=1
            lh+=1
        else:
            nonlin+=1
            nh+=1
    #m linearity
    if graph.y[3].item() == 2:
        gmem = [graph.y[ind].item() == 2 for ind in mset]
        if np.sum(gmem) == len(gmem):
            lin+=1
            lm+=1
        else:
            nonlin+=1
            nm+=1
    #t linearity
    if graph.y[4].item() == 2:
        gmem = [graph.y[ind].item() == 2 for ind in tset]
        if np.sum(gmem) == len(gmem):
            lin+=1
            lt+=1
        else:
            nonlin+=1
            nt+=1
    
    l1.append(lin),l1g.append(lg),l1h.append(lh),l1m.append(lm),l1t.append(lt)
    nl1.append(nonlin),nl1g.append(ng),nl1h.append(nh),nl1m.append(nm),nl1t.append(nt)
    
adata.obs['l1'],adata.obs['lg'],adata.obs['lh'],adata.obs['lm'],adata.obs['lt'] = l1,l1g,l1h,l1m,l1t
adata.obs['nl1'],adata.obs['ng'],adata.obs['nh'],adata.obs['nm'],adata.obs['nt'] = nl1,nl1g,nl1h,nl1m,nl1t
adata.obs['l1'] = pd.Categorical(adata.obs['l1'])
adata.obs['nl1'] = pd.Categorical(adata.obs['nl1'])

perts = ['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt']
gset = [perts.index(elem) for elem in perts if perts[1] in elem]
hset = [perts.index(elem) for elem in perts if perts[2] in elem]
mset = [perts.index(elem) for elem in perts if perts[3] in elem]
tset = [perts.index(elem) for elem in perts if perts[4] in elem]
l1,l1g,l1h,l1m,l1t = [],[],[],[],[]
nl1,nl1g,nl1h,nl1m,nl1t = [],[],[],[],[]
for graph in glist:
    lin,lg,lh,lm,lt = 0,0,0,0,0
    nonlin,ng,nh,nm,nt = 0,0,0,0,0
    #g linearity
    if graph.y[1].item() == 0:
        gmem = [graph.y[ind].item() == 0 for ind in gset]
        if np.sum(gmem) == len(gmem):
            lin+=1
            lg+=1
        else:
            nonlin+=1
            ng+=1
    #h linearity
    if graph.y[2].item() == 0:
        gmem = [graph.y[ind].item() == 0 for ind in hset]
        if np.sum(gmem) == len(gmem):
            lin+=1
            lh+=1
        else:
            nonlin+=1
            nh+=1
    #m linearity
    if graph.y[3].item() == 0:
        gmem = [graph.y[ind].item() == 0 for ind in mset]
        if np.sum(gmem) == len(gmem):
            lin+=1
            lm+=1
        else:
            nonlin+=1
            nm+=1
    #t linearity
    if graph.y[4].item() == 0:
        gmem = [graph.y[ind].item() == 0 for ind in tset]
        if np.sum(gmem) == len(gmem):
            lin+=1
            lt+=1
        else:
            nonlin+=1
            nt+=1
    
    l1.append(lin),l1g.append(lg),l1h.append(lh),l1m.append(lm),l1t.append(lt)
    nl1.append(nonlin),nl1g.append(ng),nl1h.append(nh),nl1m.append(nm),nl1t.append(nt)
    
adata.obs['lr1'],adata.obs['lrg'],adata.obs['lrh'],adata.obs['lrm'],adata.obs['lrt'] = l1,l1g,l1h,l1m,l1t
adata.obs['nlr1'],adata.obs['nlrg'],adata.obs['nlrh'],adata.obs['nlrm'],adata.obs['nlrt'] = nl1,nl1g,nl1h,nl1m,nl1t
adata.obs['lr1'] = pd.Categorical(adata.obs['lr1'])
adata.obs['nlr1'] = pd.Categorical(adata.obs['nlr1'])


# In[15]:


#min _tf for repression
min_tf = []
for graph in glist:
    tf0 = graph.y[0].item() > 0
    tf1 = max(graph.y[1:5]).item() == 0
    tf2 = max(graph.y[5:11]).item() == 0
    tf3 = max(graph.y[11:15]).item() == 0
    tf4 = graph.y[15].item() == 0
    if tf1*tf2*tf3*tf4 and tf0:
        min_tf.append(1)
    elif tf2*tf3*tf4 and tf0:
        min_tf.append(2)
    elif tf3*tf4 and tf0:
        min_tf.append(3)
    elif tf4 and tf0:
        min_tf.append(4)
    else:
        min_tf.append(5)
adata.obs['min_tf_rep'] = min_tf
adata.obs['min_tf_rep'] = pd.Categorical(adata.obs['min_tf_rep'])
sb.histplot(min_tf)


# In[6]:


#add acc to obs
model = CPU_unpickler(open('hg5_udep_hsbb_iter2000.pkl','rb')).load() #emb6 model
model.eval()
tv = torch.load('pyg_hetlist_tv25_6merrchs.pt')
mask = pkl.load(open('newbtrainmask.pkl','rb'))

accl = []
for data in tv:
    correct = data['y']
    out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
    ot = out['perturbation'].sum(dim=0)
    oe = out['enhancer'].sum(dim=0)
    op = out['promoter'].sum(dim=0)
    tp_pool = ot+oe+op
    tp_pool = tp_pool.reshape(1,32)
    pred = tp_pool
    pred = torch.where(pred>=0,1,0)
    pred = get_trin(pred)[0]
    acc = np.sum(list(pred==correct))
    accl.append(acc)
adata.obs['accl'] = accl


# In[7]:


#add idx to obs
adata.obs['idx'] = list(range(35038))


# In[15]:


#find which single is sufficient for act/rep of a gene,  make bar chart or venn diagram with it
gc,hc,mc,tc = 0,0,0,0
totc = 0
glist = tv25
for graph in glist:
    if graph.y[0] < 2:
        if graph.y[1] == 2 and graph.y[2]<2 and graph.y[3]<2 and graph.y[4]<2:
            gc+=1
        if graph.y[2] == 2 and graph.y[1]<2 and graph.y[3]<2 and graph.y[4]<2:
            hc+=1
        if graph.y[3] == 2 and graph.y[1]<2 and graph.y[2]<2 and graph.y[4]<2:
            mc+=1
        if graph.y[4] == 2 and graph.y[1]<2 and graph.y[2]<2 and graph.y[3]<2:
            tc+=1
        if graph.y[1] == 2 or graph.y[2] == 2 or graph.y[3] == 2 or graph.y[4] == 2:
            totc+=1
gc,hc,mc,tc,totc


# In[ ]:


np.sum(l1g),np.sum(l1h),np.sum(l1m),np.sum(l1t)


# In[13]:


sc.tl.tsne(adata)


# In[14]:


#saves adata object
# adata.write('adata7_annotated.h5ad')
# adata.write('adata6_ran_annotated.h5ad')
# adata.write('adata6_gmf_annotated.h5ad')
# adata.write('adata6_rannopa_annotated.h5ad')
adata.write('adata6_ttn&tnnt_tpert_annotated.h5ad')


# In[4]:


#loads adata object
adata = sc.read('adata6_annotated.h5ad')
# adata = sc.read('adata6_rannopa_annotated.h5ad')
# glist = tv25


# In[4]:


#loads in ogid
ogid = pkl.load(open('dendro_ogid.pkl','rb'))


# In[5]:


#ogid analysis


# In[87]:


ogid_full = [-5000]*len(adata)
opos = 1
for elem in ogid:
    ogid_full[elem] = opos
    opos+=1
adata.obs['ogid_full'] = ogid_full


# In[42]:


gname = 'Ttn'
idiq = ogid.index(3379)
offset = 10
dendro_neighbors = ogid[idiq-offset:idiq+offset]
dendro_gene = [0]*len(adata)
for elem in dendro_neighbors:
    dendro_gene[elem] = 1
adata.obs['dendro_'+gname] = dendro_gene


# In[74]:


#end ogid analysis


# In[132]:


adata.obs['l12'] = [int(elem) for elem in list(adata.obs['louvain']=='12')]


# In[80]:


for x in range(12,13):
    l12 = [int(elem) for elem in list(adata.obs['louvain']==str(x))]
    de12 = []
    for eid,elem in enumerate(l12):
        if elem:
            if eid in ogid:
                de12.append(ogid.index(eid))
    print(len(de12))
    sb.histplot(de12,bins=70)
    plt.show()


# In[56]:


adata.obs['louvain'][1404],adata.obs['louvain'][3379],adata.obs['louvain'][35037:]


# In[72]:


sl = [0]*35038+[3,3.67,4.33,5]+[10,9.33,8.67,8]
sl[1404],sl[3379] = 7,7
adata.obs['scolor'] = sl


# In[43]:


fig, ax = plt.subplots(figsize=(30,20))
sc.pl.tsne(
    adata,
#     cdata,
#     sdata,
#     adata[adata.obs['louvain']!='32'],
#     palette = ['gold','tomato','crimson','darkorchid','darkslategray'],
#     palette = ['darkslategray','gold','tomato','crimson','darkorchid'],
    palette = ['navy','darkorchid','crimson','orangered','coral'],
#     palette = ['midnightblue','deepskyblue','aquamarine','','greenyellow'],
#     palette = ['black','red','orange','yellow','green','blue','purple','pink','brown'],
#     palette = ['navy','teal','gold','coral','crimson'],
#     cmap = 'copper',
#     color=['accl'], 
#     color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh'],
#     color=['ghmt'],
#     color=['louvain'],
    color=['nl1'],
#     color=['scolor'],
#     color = [giq],
#     color = ['ogid_full'],
#     cmap = 'twilight',
#     color = ['dendro_'+gname],
#     color = ['l12'],
    size=200,
#     vmin=-1, vmax= 14,
#     dimensions=(0,1),
    title = '',
#     legend_loc='None',
    ax=ax
)


# In[22]:


np.sum(list(adata.obs['ng'])),np.sum(list(adata.obs['nh'])),np.sum(list(adata.obs['nm'])),np.sum(list(adata.obs['nt']))


# In[23]:


np.sum(list(adata.obs['lg'])),np.sum(list(adata.obs['lh'])),np.sum(list(adata.obs['lm'])),np.sum(list(adata.obs['lt']))


# In[25]:


np.sum(list(adata.obs['lrg'])),np.sum(list(adata.obs['lrh'])),np.sum(list(adata.obs['lrm'])),np.sum(list(adata.obs['lrt']))


# In[27]:


np.sum(list(adata.obs['nlrg'])),np.sum(list(adata.obs['nlrh'])),np.sum(list(adata.obs['nlrm'])),np.sum(list(adata.obs['nlrt']))


# In[12]:


sdata = adata
sdata = sdata[sdata.obs['louvain']=='0']
# sdata = sdata[sdata.obsm['X_tsne'][:,0]>-20]
# sdata = sdata[sdata.obsm['X_tsne'][:,0]<15]
# sdata = sdata[sdata.obsm['X_tsne'][:,1]<-95]
# sdata = sdata[sdata.obsm['X_tsne'][:,1]>-97]


#finished initial check for first 100 genes in louvain=0
x=0
len(list(sdata.obs['gene'])),list(set(sdata.obs['gene']))[x:x+100]

#for saving louvain=0 gene list (cardiac synth. pert candidate cluster),    only need to run once
# fl = list(set(sdata.obs['gene']))
# with open('l0_gnames.pkl','wb') as file:
#     pkl.dump(fl, file)
newfl = pkl.load(open('l0_gnames.pkl','rb'))

#for single gene annotation
gl = pkl.load(open('hl_gene_names.pkl','rb')) # + ['synthetic_gene']*len(synthlist)
#can dwd


# In[6]:


newfl = pkl.load(open('l0_gnames.pkl','rb'))


# In[132]:


x=1120
for ind,elem in enumerate(newfl[x:x+20]):
    print(x+ind,'\t',elem)


# In[7]:


#can dwd
gl = pkl.load(open('hl_gene_names.pkl','rb'))


# In[26]:


giq = 'Mfap4'
fgl = []
for elem in gl:
    fgl.append((elem==giq)*len(set(fgl)))
adata.obs[giq] = fgl


# In[27]:


pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
print(adata[adata.obs[giq]>0].obs[:])


# In[7]:


hsrc_names = pkl.load(open('hsrc_names.pkl','rb'))
# hsrc_names = pkl.load(open('hsrc_names_rannopa.pkl','rb'))
gata_id = hsrc_names.index(('AGATAA','TTATCT')) + 1
mef_id = hsrc_names.index(('AAAATA','TATTTT')) + 1
# pola_id = hsrc_names.index(('AAAAAA','TTTTTT')) + 1


# In[22]:


#for gmf
# gata_id,pola_id = 1,2


# In[8]:


# adata.obs['motif_pola'] = [graph['enhancer'].x[:,pola_id].sum().item()/graph['enhancer'].x.shape[0] for graph in tv25]
adata.obs['motif_gata'] = [graph['enhancer'].x[:,gata_id].sum().item()/graph['enhancer'].x.shape[0] for graph in tv25]
adata.obs['motif_mef'] = [graph['enhancer'].x[:,mef_id].sum().item()/graph['enhancer'].x.shape[0] for graph in tv25]
adata.obs['motif_dist'] = [graph['enhancer'].x[:,0].sum().item()/graph['enhancer'].x.shape[0] for graph in tv25]


# In[5]:


fig, ax = plt.subplots(figsize=(30,20))
sc.pl.tsne(
    adata,
#     cdata,
#     sdata,
#     adata[adata.obs['louvain']!='32'],
#     palette = ['gold','tomato','crimson','darkorchid','darkslategray'],
#     palette = ['darkslategray','darkorchid','crimson','orangered','coral'],
#     palette = ['midnightblue','deepskyblue','aquamarine','','greenyellow'],
#     palette = ['black','red','orange','yellow','green','blue','purple','pink','brown'],
#     palette = ['navy','teal','gold'],
#     cmap = 'copper',
#     color=['g'], 
#     color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh'],
#     color=['ghmt'],
#     color=['louvain'],
    color=['min_tf_rep'],
#     color=['motif_mef'],
#     color = [giq],
#     color = ['dendro_'+gname],
#     color = ['min_tf_rep'],
    size=300,
#     vmin=0, vmax= 5,
#     dimensions=(0,1),
    title = '',
#     legend_loc='None',
    ax=ax
)


# In[22]:


adata[adata.obs['Gapdh']>0].obs


# In[174]:


#for gene series annotation
gl = pkl.load(open('hl_gene_names.pkl','rb'))
# gl = pkl.load(open('hl_gene_names.pkl','rb')) + ['synthetic_gene']*len(synthlist)
# gl = pkl.load(open('hl_gene_names.pkl','rb'))+['synthetic_gene_tn']*4+['synthetic_gene_tt']*4+['synthetic_gene_p']*4
#can dwd
giqlist = ['Dsg1','Dsg2','Dsg3','Dsg4']
# giqlist = ['Tnnt2','synthetic_gene_tn']
# giqlist = ['Ttn','synthetic_gene_tt']
# giqlist = ['Pkp1','synthetic_gene_p']
fgl = []
for elem in gl:
    fgl.append((elem in giqlist)*len(set(fgl)))
adata.obs['Dsg_series'] = fgl


# In[58]:


adata.obs['tneu'] = [int(np.prod(graph.y.tolist())==1) for graph in tv25]
adata.obs['tpos'] = [int(graph.y.tolist()==[0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]) for graph in tv25]
adata.obs['tneg'] = [int(graph.y.tolist()==[2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) for graph in tv25]


# In[15]:


adata


# In[20]:


synthlist = [tnnog,tng1,tng5,tng10,ttnog,ttg1,ttg5,ttg10,pnog,pg1,pg5,pg10]


# In[135]:


for g in synthlist:
    print(g['enhancer'].x[:,316])


# In[136]:


tv25[1404]['enhancer'].x[:,316],tv25[1408]['enhancer'].x[:,316],tv25[3379]['enhancer'].x[:,316]


# In[6]:


fig, ax = plt.subplots(figsize=(30,20))
sc.pl.tsne(
    adata,
#     cdata,
#     sdata,
#     adata[adata.obs['louvain']!='32'],
#     palette = ['gold','tomato','crimson','darkorchid','darkslategray'],
#     palette = ['darkslategray','darkorchid','crimson','orangered','coral'],
#     palette = ['midnightblue','deepskyblue','aquamarine','','greenyellow'],
#     palette = ['black','red','orange','yellow','green','blue','purple','pink','brown'],
#     cmap = 'tab10',
#     color=['min_tf'], 
#     color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh'],
    color=['ghmt'],
#     color=['louvain'],
#     color=['nlr1'],
#     color=['Pkp1'],
#     color = [giq],
#     color = ['TSP'],
    size=250,
#     vmin=0, vmax= 5,
#     dimensions=(0,1),
#     title = '',
    legend_loc='None',
    ax=ax
)


# In[20]:


synthlist = [tolp,tnolp,polt,pnolt]


# In[154]:


adata[adata.obs['TSP']>0].obs


# In[74]:


np.sum(adata.obs['ng']),np.sum(adata.obs['nh']),np.sum(adata.obs['nm']),np.sum(adata.obs['nt'])


# In[73]:


np.sum(adata.obs['lg']),np.sum(adata.obs['lh']),np.sum(adata.obs['lm']),np.sum(adata.obs['lt'])


# In[20]:


#for getting active enhancer %


# In[5]:


tv25 = torch.load('pyg_hetlist_tv25_6merrchs.pt')


# In[30]:


adata[adata.obs['gene'] == 'Des'].obs.index


# In[22]:


hl = torch.load('pyg_hetlist_co.pt')


# In[33]:


print(hl[739]['perturbation','foldchange','rna'].edge_attr[0:5])
print(hl[739]['perturbation','foldchange','rna'].edge_attr[5:11])
print(hl[739]['perturbation','foldchange','rna'].edge_attr[11:15])
print(hl[739]['perturbation','foldchange','rna'].edge_attr[15:16])


# In[22]:


thresh = 35038
cutoff = 0.25
olp_list_up,olp_list_down = [],[]
egp_list_up,egp_list_down = [],[]
egp_list_neu = []
olw_up,olw_down = [[],[]],[[],[]]
for graph in tv25[:]:
    nume = len(graph['enhancer'].x)
    g_neu = False
    #define graph as up or down by GHMT:
    if graph.y[-1] > 1:
        g_updown = True
    elif graph.y[-1] < 1:
        g_updown = False
    else:
        g_neu = True
    eg_set,pe_set = [],[]
    eg_w = []
    for enum in range(nume):
        epany = graph['enhancer','foldchange','promoter'].edge_attr[enum] - 1 > cutoff and graph['enhancer','pvalue','promoter'].edge_attr[enum] > -np.log10(0.05/thresh)
        erany = graph['enhancer','foldchange','rna'].edge_attr[enum] - 1 > cutoff and graph['enhancer','pvalue','rna'].edge_attr[enum] > -np.log10(0.05/thresh)
        if erany or epany:
            eg_set.append(enum)
            epw = graph['enhancer','foldchange','promoter'].edge_attr[enum] - 1
            erw = graph['enhancer','foldchange','rna'].edge_attr[enum] - 1
            eg_w.append(epw)
        p_set = []
        for pnum in range(15,16):
            peany = graph['perturbation','foldchange','enhancer'].edge_attr[enum::nume][pnum] - 1 > cutoff and graph['perturbation','pvalue','enhancer'].edge_attr[enum::nume][pnum] > -np.log10(0.05/14800000)
            p_set.append(peany)
        peany = np.sum(p_set)
        if peany:
            pe_set.append(enum)
    ol_set = [elem for elem in eg_set if elem in pe_set]
    if len(ol_set)==0 and len(eg_set)==0:
        continue
#     if epany:
#         o_epw = np.mean([elem for elem in range(len(eg_w)) if eg_set[elem] in ol_set])
#         no_epw = np.mean([elem for elem in range(len(eg_w)) if eg_set[elem] not in ol_set])
#     if erany:
#         o_erw = np.mean([elem for elem in range(len(eg_w)) if eg_set[elem] in ol_set])
#         no_erw = np.mean([elem for elem in range(len(eg_w)) if eg_set[elem] not in ol_set])
        
    olp = 2*len(ol_set)/(len(eg_set)+len(pe_set))
    egp = len(ol_set)/len(eg_set)
    if g_neu:
        egp_list_neu.append(egp)
    
    if g_updown:
        olp_list_up.append(olp)
        egp_list_up.append(egp)
        if epany:
            olw_up[0].append(o_epw),olw_up[1].append(no_epw)
#             olw_up[0].append(o_erw),olw_up[1].append(no_erw)
    else:
        olp_list_down.append(olp)
        egp_list_down.append(egp)
        if epany:
            olw_down[0].append(o_epw),olw_down[1].append(no_epw)
#             olw_down[0].append(o_erw),olw_down[1].append(no_erw)
print(len(olp_list_up),len(olp_list_down))


# In[23]:


#pweights


# In[24]:


print(len(egp_list_down),len(egp_list_neu),len(egp_list_up))
print(np.mean(egp_list_down),np.mean(egp_list_neu),np.mean(egp_list_up))


# In[14]:


print(len(egp_list_up),len(egp_list_down))
nc=0
for elem in egp_list_down:
    if type(elem) == np.nan:
        nc+=1
print(nc)


# In[11]:


np.mean(egp_list_up),np.median(egp_list_up),np.mean(egp_list_down),np.median(egp_list_down)


# In[169]:


nro,nrn = [],[]
for elem in range(len(olw_up[0])):
    if not np.isnan(olw_up[1][elem]):
        nro.append(olw_up[0][elem]),nrn.append(olw_up[1][elem])
np.median(nro),np.median(nrn),np.mean(nro),np.mean(nrn)


# In[170]:


nro,nrn = [],[]
for elem in range(len(olw_down[0])):
    if not np.isnan(olw_down[1][elem]):
        nro.append(olw_down[0][elem]),nrn.append(olw_down[1][elem])
np.median(nro),np.median(nrn),np.mean(nro),np.mean(nrn)


# In[ ]:


#this analysis might not be valid if I'm looking at all up or down genes, may be worth focusing on a specifc region around fib peak and card peak?
#also , only about half as much of the enhancers OL for down.reg. genes so majority of E contribution is still coming from Non-OL peaks
#also, promoter could have higher relative importance for down-reg genes, which wouldnt show up here


# In[158]:


np.median(olw_up[0]),np.nanmedian(olw_up[1])


# In[159]:


np.median(olw_down[0]),np.nanmedian(olw_down[1])


# In[128]:


#prom or RNA
np.median(olp_list_up),np.median(olp_list_down),np.mean(olp_list_up),np.mean(olp_list_down)


# In[112]:


#only look at prom
np.median(olp_list_up),np.median(olp_list_down),np.mean(olp_list_up),np.mean(olp_list_down)


# In[116]:


#only look at RNA
np.median(olp_list_up),np.median(olp_list_down),np.mean(olp_list_up),np.mean(olp_list_down)


# In[78]:


thresh = 35038
peg_up,peg_down = [0,0],[0,0] #up graph [up links, down links], down graph [up links, down links]
gc = [0,0]
for graph in tv25[:]:
    ge_flag = False
    nume = len(graph['enhancer'].x)
    #define graph as up or down by GHMT:
    if graph.y[-1] > 1:
        g_updown = True
    elif graph.y[-1] < 1:
        g_updown = False
    else:
        continue
    for enum in range(nume):
#         epany = abs(graph['enhancer','foldchange','promoter'].edge_attr[enum] - 1) > 0.25 and graph['enhancer','pvalue','promoter'].edge_attr[enum] > -np.log10(0.05/thresh)
#         erany = abs(graph['enhancer','foldchange','rna'].edge_attr[enum] - 1) > 0.25 and graph['enhancer','pvalue','rna'].edge_attr[enum] > -np.log10(0.05/thresh)
        epany = graph['enhancer','foldchange','promoter'].edge_attr[enum] - 1 > 0.25 and graph['enhancer','pvalue','promoter'].edge_attr[enum] > -np.log10(0.05/thresh)
        erany = graph['enhancer','foldchange','rna'].edge_attr[enum] - 1 > 0.25 and graph['enhancer','pvalue','rna'].edge_attr[enum] > -np.log10(0.05/thresh)
#         epany = False
        if epany or erany:
            #checks for link to any perturbation, may want to change to only GHMT though
            for pnum in range(15,16):
                peany = abs(graph['perturbation','foldchange','enhancer'].edge_attr[enum::nume][pnum] - 1) > 0.25 and graph['perturbation','pvalue','enhancer'].edge_attr[enum::nume][pnum] > -np.log10(0.05/14800000)
                if peany:
                    ge_flag = True
                    #if pe is (+)
                    if graph['perturbation','foldchange','enhancer'].edge_attr[enum::nume][pnum] > 1:
                        if g_updown:
                            peg_up[0] += 1
                        else:
                            peg_down[0] += 1
                    #if pe is (-)
                    else:
                        if g_updown:
                            peg_up[1] += 1
                        else:
                            peg_down[1] += 1
    if ge_flag:
        if g_updown:
            gc[0] += 1
        else:
            gc[1] += 1
print(peg_up,peg_down)


# In[79]:


peg_up[1]/np.sum(peg_up),peg_down[1]/np.sum(peg_down)


# In[73]:


11921/gc[0], 3251/gc[1]


# In[72]:


gc[0]/8080,gc[1]/9439


# In[62]:


peg_up[0]/np.sum(peg_up),peg_down[0]/np.sum(peg_down)


# In[60]:


peg_up[0]/np.sum(peg_up),peg_down[0]/np.sum(peg_down)


# In[54]:


peg_up[0]/np.sum(peg_up),peg_down[0]/np.sum(peg_down)


# In[66]:


np.sum(adata.obs['ghmt']==0),np.sum(adata.obs['ghmt']==1),np.sum(adata.obs['ghmt']==2)


# In[20]:


#end for getting "active enhancer %


# In[20]:


#for making min_tf barplot


# In[15]:


h = []
for num in range(1,5):
    print(num,'\t',np.sum(adata.obs['min_tf'] == num))
    h.append(np.sum(adata.obs['min_tf'] == num))
for num in range(1,5):
    print(num,'\t',np.sum(adata.obs['min_tf_rep'] == num))
    h.append(np.sum(adata.obs['min_tf_rep'] == num))


# In[62]:


fig, ax = plt.subplots(figsize=(12,10))
plt.axhline(y=1000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=2000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=3000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=4000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=5000, color='slategrey', linestyle='--',alpha=0.2)
plt.bar((1,2,3,4),h[0:4],width=0.8,color='darkred')
plt.xticks([]),plt.yticks(ticks=None)


# In[65]:


fig, ax = plt.subplots(figsize=(12,10))
plt.axhline(y=1000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=2000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=3000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=4000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=5000, color='slategrey', linestyle='--',alpha=0.2)
plt.axhline(y=6000, color='slategrey', linestyle='--',alpha=0.2)
plt.bar((1,2,3,4),h[4:8],width=0.8,color='darkslateblue')
plt.xticks([]),plt.yticks(ticks=None)


# In[20]:


#end for making min_tf barplot


# In[4]:


#for finding neighbor ranks


# In[31]:


giq = 'Tnnt2'
adata.obs[giq] = [int(elem) for elem in adata.obs['gene']==giq]
txy = adata[adata.obs[giq]==1].obsm['X_tsne'][0]

giq = 'Pkp1'
adata.obs[giq] = [int(elem) for elem in adata.obs['gene']==giq]
dxy = adata[adata.obs[giq]==1].obsm['X_tsne'][0]

td_euc = ((txy[0]-dxy[0])**2+(txy[1]-dxy[1])**2)**0.5

rad_list = []
for xy in adata.obsm['X_tsne'][:]:
    in_range = ((txy[0]-xy[0])**2+(txy[1]-xy[1])**2)**0.5<td_euc
    rad_list.append(int(in_range))
adata.obs['in_range'] = rad_list

np.sum(rad_list)


# In[32]:


fig, ax = plt.subplots(figsize=(30,20))
sc.pl.tsne(
    adata,
#     palette = ['gold','tomato','crimson','darkorchid','darkslategray'],
#     palette = ['darkslategray','gold','tomato','crimson','darkorchid'],
#     palette = ['navy','darkorchid','crimson','orangered','coral'],
#     palette = ['midnightblue','deepskyblue','aquamarine','','greenyellow'],
#     palette = ['black','red','orange','yellow','green','blue','purple','pink','brown'],
#     palette = ['navy','teal','gold','coral','crimson'],
#     cmap = 'copper',
#     color=['accl'], 
#     color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh'],
#     color=['ghmt'],
#     color = [giq],
    color = ['in_range'],
#     cmap = 'twilight',
#     color = ['dendro_'+gname],
    size=200,
#     vmin=-1, vmax= 14,
#     dimensions=(0,1),
    title = '',
#     legend_loc='None',
    ax=ax
)


# In[20]:


#end for finding neighbor ranks


# In[42]:


fig = sc.pl.tsne(
    adata,
    cmap = 'viridis',
#     color=['f4y_fltrin'], 
    color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh','nl1','l1'],
#     color=['nl1','ng','nh','nm','nt'],
#     color=['l1','lg','lh','lm','lt'],
#     color=['ghmt'],
#     color=['louvain'],
#     color=['num_enh'],
#     color=['mucpm'],
    size=10,
#     vmin=0, vmax= 1,
#     ax=ax
#     dimensions = (0,1)
    return_fig = True
)


# In[55]:


fig.axes[0].xlabel=''


# In[58]:


fig


# ### Point Embedding 

# In[35]:


#load and loaderize modified graphs
tolp = pkl.load(open('synthg_tnnt2_ol_pkp1.pkl','rb'))
tnolp = pkl.load(open('synthg_tnnt2_nol_pkp1.pkl','rb'))
polt = pkl.load(open('synthg_pkp1_ol_tnnt2.pkl','rb'))
pnolt = pkl.load(open('synthg_pkp1_nol_tnnt2.pkl','rb'))
synthlist = [tolp,tnolp,polt,pnolt]
synthloader = pyg.loader.DataLoader(synthlist,shuffle=False)

#load model and make synth embedding array
model = CPU_unpickler(open('hg5_udep_hsbb_iter2000.pkl','rb')).load()     #emb6 model
model.eval()
synth_emb = []
for epoch in range(0,1):
    start = dtime()
    count=0
    for data in synthloader:
        out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)
        pool = torch.cat((out['perturbation'].sum(dim=0),out['enhancer'].sum(dim=0),out['promoter'].sum(dim=0)))
        synth_emb.append(pool.reshape((1,-1)))

arem = [embed.detach().numpy() for embed in synth_emb]
datamat = np.reshape(np.array(arem),(-1,96)) #for e pooling


# ### < End Point Embedding >

# In[27]:


#density
np.shape(adata.obsm['X_tsne'][:,0])


# In[29]:


sb.histplot(adata.obsm['X_tsne'][:,0])


# In[30]:


sb.histplot(adata.obsm['X_tsne'][:,1])


# In[250]:


#seq stuff, may want later

#add sequence obs
#LIN REG FOR REV COMP
# psm = pkl.load(open('promotif_smatrix.pkl','rb'))
# pmm = pkl.load(open('promotif_mmatrix.pkl','rb'))
# esm = pkl.load(open('enhmotif_smatrix.pkl','rb'))
# emm = pkl.load(open('enhmotif_mmatrix.pkl','rb'))
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

seq = 'TATGAC'
revseq = rev_comp(seq)
seq_ind = adata_6mer.var_names.tolist().index(seq)
revseq_ind = adata_6mer.var_names.tolist().index(revseq)
motif_arr = []
for graph in glist:
#     motif_count = graph['enhancer'].x[:,1+seq_ind].sum()+graph['promoter'].x[:,seq_ind]
    motif_count = graph['promoter'].x[:,seq_ind]+graph['promoter'].x[:,revseq_ind]+graph['enhancer'].x[:,1+seq_ind].sum()+graph['enhancer'].x[:,1+revseq_ind].sum()
    motif_arr.append(motif_count.item())
adata.obs[seq] = motif_arr

seq = 'TATGAC'
revseq = rev_comp(seq)
seq_ind = adata_6mer.var_names.tolist().index(seq)
revseq_ind = adata_6mer.var_names.tolist().index(revseq)
motif_arr = []
for graph in glist:
#     motif_count = graph['enhancer'].x[:,1+seq_ind].sum()+graph['promoter'].x[:,seq_ind]
    motif_count = graph['promoter'].x[:,seq_ind]+graph['promoter'].x[:,revseq_ind]
    motif_arr.append(motif_count.item())
adata.obs[seq+'_prom'] = motif_arr

seqnames = list(adata_6mer.var_names)
rc_seqs = []
for sid,seq in enumerate(seqnames):
    seqtup = list(set((seqnames.index(seq),seqnames.index(rev_comp(seq)))))
    if seqtup not in rc_seqs:
        rc_seqs.append(seqtup)


# In[250]:


#differential motif analysis
arem = []
for graph in tv25:
    gmar = graph['enhancer'].x[:,1:].detach().numpy()
    gmar = np.sum(gmar,axis=0)
    arem.append(gmar)

datamat = np.array(arem)
motifdata = sc.AnnData(datamat)

hsrc_names = pkl.load(open('hsrc_names.pkl','rb'))
hsrc_names = [elem[0]+'_'+elem[-1] for elem in hsrc_names]
motifdata.var_names = hsrc_names
motifdata.obs['louvain'] = adata.obs['louvain']
# sc.pp.log1p(motifdata)


# In[147]:


np.shape(motifdata.X)


# In[233]:


sc.tl.rank_genes_groups(motifdata,groupby='louvain',method='wilcoxon')


# In[235]:


tm = []
for louv in range(len(motifdata.uns['rank_genes_groups']['names'][0])):
    lmo = []
    for motif in range(5):
        lmo.append(motifdata.uns['rank_genes_groups']['names'][motif][louv])
    tm+=lmo


# In[236]:


sc.pl.heatmap(motifdata,var_names=tm,groupby='louvain')


# In[ ]:


sc.pl.matrixplot(motifdata,['gmot','tmot'],groupby='louvain')


# In[271]:


('AAAAAA','TTTTTT') in hsrc_names


# In[270]:


'AGGTGT_ACACCT' in motifdata.var_names,('AGGTGT','ACACCT') in hsrc_names


# In[264]:


'TTATCT_AGATAA' in motifdata.var_names


# In[263]:


'AAAAAA_TTTTTT' in motifdata.var_names


# In[262]:


motifdata.var_names


# In[ ]:


sc.pl.rank_genes_groups(motifdata)


# In[20]:


#given 6mer string and hsrc_names, returns index of 6mer
def motif_indexer(motif):
    for names_idx,names_tup in enumerate(hsrc_names):
        if motif in names_tup:
            idx = names_idx
    return idx
hsrc_names = pkl.load(open('hsrc_names.pkl','rb'))


# In[255]:


mid = motif_indexer('AGGTGT')
ml = list (motifdata[:,mid].X)
ml = [float(elem) for elem in ml]
adata.obs['tmot'] = ml


# In[100]:


gl = pkl.load(open('hl_gene_names.pkl','rb'))
cgl = [int(elem in ('Tnnt2','Hcn4','Ttn','Pkp1','Nkx2-5','Ascl1','Actn2','Tbx20','Srf','Pitx2','Sall1','Sall2','Isl1','Gata6','Mesp1','Hey2','Hand1')) for elem in gl]
adata.obs['card'] = cgl
fgl = [int(elem in ('Vim','Fbn1','Fap','Col1a1')) for elem in gl]
adata.obs['fibr'] = fgl


# In[102]:


#if numbering is not needed
# giq = 'Gapdh'
# fgl = [int(elem == giq) for elem in gl]
# adata.obs[giq] = fgl

#if numbering is needed
giq = 'Pkp2'
fgl = []
for elem in gl:
    fgl.append((elem==giq)*len(set(fgl)))
adata.obs[giq] = fgl


# In[74]:


adata[adata.obs['gene']==giq]


# In[ ]:


# fig, ax = plt.subplots(figsize=(25,20),nrows=2,ncols=3)
# fig, ax = plt.subplots(figsize=(30,20))
sc.pl.umap(
    adata,
    cmap = 'viridis',
#     color=['f4y_fltrin'], 
    color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt'],
#     color=['lg','lh','lm','lt','ng','nh','nm','nt'],
#     color=['lrg','lrh','lrm','lrt'],
#     color=['louvain'],
#     color=['num_enh'],
#     color=[goid_str[0]],
    size=5,
#     vmin=0, vmax= 100,
#     ax=ax
)


# In[29]:


gl = pkl.load(open('hl_gene_names.pkl','rb'))


# In[39]:


gl = pkl.load(open('hl_gene_names.pkl','rb'))
zce_rna = pkl.load(open('conce_rna.pkl','rb'))
zce_rna.X = zce_rna.X.todense()
mucpm = []
for g in gl:
    vid = list(zce_rna.var_names).index(g)
    if vid == 122:
        print("OMG")
    if vid%10000==0:
        print(vid)
    mucpm.append(np.mean(zce_rna[:,vid].X))
adata.obs['mucpm'] = mucpm
adata.obs['log_mucpm'] = np.log10([elem+1e-6 for elem in mucpm])


# In[41]:


for gid in range(123):
#     print(np.mean(zce_rna[:,gid].X))
    if np.mean(zce_rna[:,gid].X) == 0:
        print(gid)


# In[43]:


zce_rna[:,122].var


# In[53]:


for g in gl:
    vid = list(zce_rna.var_names).index(g)
    if np.mean(zce_rna[:,vid].X) == 0:
        print(g,vid)


# In[58]:


np.sum(zce_rna[:,726].X)


# In[66]:


# fig, ax = plt.subplots(figsize=(25,20),nrows=2,ncols=3)
# fig, ax = plt.subplots(figsize=(30,20))
sc.pl.umap(
    adata,
    cmap = 'viridis',
#     color=['f4y_fltrin'], 
#     color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh'],
    color=['nl1','ng','nh','nm','nt'],
#     color=['l1','lg','lh','lm','lt'],
#     color=['ghmt'],
#     color=['louvain'],
#     color=['num_enh'],
#     color=['mucpm'],
#     size=250,
#     vmin=0, vmax= 1,
#     ax=ax
    dimensions = (0,1)
)


# In[128]:


np.max(sdata.obsm['X_umap'][:,1])


# In[168]:


np.max(sdata.obsm['X_umap'][:,1])
xy = adata[adata.obs['Tnnt2']==1].obsm['X_umap'][0]
dist = 0.1
neighbors = []
for gid,gene in enumerate(adata.obsm['X_umap']):
    close = True
    for eid,elem in enumerate(gene):
        if abs(elem-xy[eid])>dist:
            close = False
    if close:
        neighbors.append(gid)
    if gid%10000==0:
        print(gid)


# In[169]:


neighbors


# In[171]:


adata[neighbors].obs


# ## LOW UMI REMOVED

# In[53]:


sdata = adata[adata.obs['louvain']!='27']


# In[54]:


sc.tl.pca(sdata)


# In[55]:


#remove trailing empty PCs
print(np.shape(sdata.obsm['X_pca']))
pc_sums = [bool(np.sum(sdata.obsm['X_pca'][:,which_pc])) for which_pc in range(np.shape(sdata.obsm['X_pca'])[1])]
valid_pcs = np.sum(pc_sums)
sdata.obsm['X_pca'] = sdata.obsm['X_pca'][:,:valid_pcs]
print(np.shape(sdata.obsm['X_pca']))

sc.pp.neighbors(sdata,use_rep='X_pca')
sc.tl.louvain(sdata,resolution = 1.5)
sc.tl.paga(
    sdata,
    groups='louvain', 
)
sc.tl.umap(
    sdata,
    init_pos='X_pca',
    min_dist = 0.5,
    spread = 2,
)


# In[24]:


#add genes
gene = 'Gapdh'
sdata.obs[gene] = [int(elem) for elem in sdata.obs['gene'] == gene]


# In[32]:


# fig, ax = plt.subplots(figsize=(25,20),nrows=2,ncols=3)
fig, ax = plt.subplots(figsize=(30,20))
sc.pl.umap(
    sdata,
    cmap = 'plasma',
#     color=['f4y_fltrin'], 
#     color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh'],
#     color=['ghmt'],
    color=['louvain'],
#     color=['hood'],
#     color=[gene],
    size=150,
#     vmin=0, vmax= 100,
    ax=ax
)


# In[57]:


pl=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt']
# pl = ['nl1','ng','nh','nm','nt','l1','lg','lh','lm','lt']
# pl = ['lr1','lrg','lrh','lrm','lrt']
# pl = ['l1','lg','lh','lm','lt']
# fig, ax = plt.subplots(figsize=(25,20),nrows=4,ncols=4)
# fig, ax = plt.subplots(figsize=(30,20))
# for p in pl:
plt.ylabel='',
sc.pl.umap(
    sdata,
    cmap = 'viridis',
#     color=['f4y_fltrin'], 
    color=pl,
#     color=['lg','lh','lm','lt','ng','nh','nm','nt'],
#     color=['louvain'],
#     color=['num_enh'],
#     color=[goid_str[0]],
    size=5,
#     vmin=0, vmax= 100,
#     ax=ax,
    colorbar_loc=None,
    legend_loc=None,
    wspace=0,
    hspace=0.2,
    show=False,
#         colorbar_loc='none',
)
plt.ylabel = ''


# In[155]:


sdata[sdata.obsm['X_umap'][:,1]==np.max(sdata.obsm['X_umap'][:,1])].obs['gene']


# In[153]:


sdata[sdata.obsm['X_umap'][:,1]==1.6325023].obsm['X_umap'][0]


# In[158]:


sdata[sdata.obs['gene']=='Ppp2r2c'].obs


# In[293]:


sdata[sdata.obs['gene']=='Ttn'].obsm['X_umap'][1]


# In[175]:


sdata[sdata.obsm['X_umap'][:,1]==1.6325023].obsm['X_umap'][0]


# In[262]:


xy = sdata[sdata.obsm['X_umap'][:,1]==1.6325023].obsm['X_umap'][0][0:2]
xy = np.array([9,1.3])
xy


# In[324]:


# xy = sdata[sdata.obs['gene']=='Ttn'].obsm['X_umap'][1]
# xy = sdata[sdata.obsm['X_umap'][:,1]==1.6325023].obsm['X_umap'][0][0:2]
xy = np.array([3,0.4])
dist = 0.1
neighbors = []
for gid,gene in enumerate(sdata.obsm['X_umap']):
    close = True
    for eid,elem in enumerate(gene[0:2]):
        if abs(elem-xy[eid])>dist:
            close = False
    if close:
        neighbors.append(gid)
    if gid%10000==0:
        print(gid)
print(len(neighbors))


# In[325]:


nn = list(sdata[neighbors].obs['gene'])
hood = []
for gid in range(len(sdata)):
    if sdata.obs['gene'][gid] in nn:
        hood.append(1)
    else:
        hood.append(0)
sdata.obs['hood'] = hood
print(np.sum(sdata.obs['hood']))


# In[28]:


# fig, ax = plt.subplots(figsize=(25,20),nrows=2,ncols=3)
fig, ax = plt.subplots(figsize=(30,20))
sc.pl.umap(
    sdata,
    cmap = 'plasma',
#     color=['f4y_fltrin'], 
#     color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh'],
    color=['nl1'],
#     color=['louvain'],
#     color=['hood'],
#     color=[gene],
    size=150,
#     vmin=0, vmax= 100,
    ax=ax
)


# In[327]:


nn


# In[328]:


sdata[sdata.obs['hood']==1].obs['gene']

