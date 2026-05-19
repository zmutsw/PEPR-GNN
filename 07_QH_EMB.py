
# coding: utf-8

import numpy as np
import scanpy as sc
import pandas as pd
import pickle as pkl
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

import pickle
import io

#load in dataset
tv25 = torch.load('pyg_hetlist_tv25_6merrchs.pt')


# ## DATA PREP and UTILS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#use example:
#model = CPU_unpickler(open('model_iterX.pkl','rb')).load()
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

#NEW heterodata prep and loading
fulllist = tv25
mask = pkl.load(open('newbtrainmask.pkl','rb'))
testlist = [fulllist[ind] for ind in range(len(fulllist)) if mask[ind]]
trainlist = [fulllist[ind] for ind in range(len(fulllist)) if not mask[ind]]
#loaderize datasets
fullloader = pyg.loader.DataLoader(fulllist,shuffle=False)
trainloader = pyg.loader.DataLoader(trainlist[::50],shuffle=False)
testloader = pyg.loader.DataLoader(testlist,shuffle=False)

#for running existing model
model = CPU_unpickler(open('hg5_udep_hsbb_iter2000.pkl','rb')).load()     #emb6
model.eval()

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

with open('raw_emb.pkl','wb') as f:
    pkl.dump(raw_emb,f)

#for getting emb accuracy scores
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
raw_emb = pkl.load(open('raw_emb.pkl','rb'))
glist = tv25    #if using all graphs
# glist = testlist    #if using test graphs

arem = [embed.detach().numpy() for embed in raw_emb]
datamat = np.reshape(np.array(arem),(-1,96)) #for e pooling
num_enh = [graph['enhancer'].x.size()[0] for graph in glist]
adata = sc.AnnData(datamat)
adata.obs['num_enh'] = num_enh

#do pca for neighbors
sc.tl.pca(adata)

#remove trailing empty PCs
print(np.shape(adata.obsm['X_pca']))
pc_sums = [bool(np.sum(adata.obsm['X_pca'][:,which_pc])) for which_pc in range(np.shape(adata.obsm['X_pca'])[1])]
valid_pcs = np.sum(pc_sums)
adata.obsm['X_pca'] = adata.obsm['X_pca'][:,:valid_pcs]
print(np.shape(adata.obsm['X_pca']))

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

#add idx to obs
adata.obs['idx'] = list(range(35038))

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

np.sum(l1g),np.sum(l1h),np.sum(l1m),np.sum(l1t)

sc.tl.tsne(adata)

#saves adata object
# adata.write('adata7_annotated.h5ad')
# adata.write('adata6_ran_annotated.h5ad')
# adata.write('adata6_gmf_annotated.h5ad')
# adata.write('adata6_rannopa_annotated.h5ad')
adata.write('adata6_annotated.h5ad')

#loads adata object
adata = sc.read('adata6_annotated.h5ad')

fig, ax = plt.subplots(figsize=(30,20))
sc.pl.tsne(
    adata,
#     palette = ['gold','tomato','crimson','darkorchid','darkslategray'],
#     color=['nc','g','h','m','t','gh','gm','gt','hm','ht','mt','ghm','ght','gmt','hmt','ghmt','louvain','num_enh'],
    color=['ghmt'],
#     color=['louvain'],
#     color=['nlr1'],
    size=250,
#     vmin=0, vmax= 5,
#     title = '',
    legend_loc='None',
    ax=ax
)
