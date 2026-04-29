
# coding: utf-8

# In[1]:
import ctypes
libgcc_s=ctypes.CDLL('libgcc_s.so.1')

import numpy as np
import scanpy as sc
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
from torch_geometric.nn import GCNConv, GraphConv, ResGatedGraphConv, SAGEConv, pool, to_hetero, to_hetero_with_bases
import torch_geometric.transforms as T

# set a working directory for saving plots
os.chdir('/project/GCRB/Hon_lab/s437603/data/ghmt_multiome/analysis')

hetlist = torch.load('pyg_hetlist_tv25_6merrchs.pt')

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
        if oh[2*ind]==0 and oh[2*ind+1]==1:
            oh_real.append(0)
        elif oh[2*ind]==0 and oh[2*ind+1]==0:
            oh_real.append(1)
        elif oh[2*ind]==1 and oh[2*ind+1]==0:
            oh_real.append(2)
        # always erroneous, up AND down prediction
        elif oh[2*ind]==1 and oh[2*ind+1]==1:
            oh_real.append(-13) #ensures error tracing for linear model
    oh_real = tc.Tensor([oh_real])        
    return oh_real

#load in htb mask to minimize test edge exposure
#NEW heterodata prep and loading
fulllist = hetlist
mask = pkl.load(open('htbtrainmask.pkl','rb'))
testlist = [fulllist[ind] for ind in range(len(fulllist)) if mask[ind]]
trainlist = [fulllist[ind] for ind in range(len(fulllist)) if not mask[ind]]

#loaderize datasets
gpb = 100
fullloader = pyg.loader.DataLoader(fulllist,shuffle=True)
trainloader = pyg.loader.DataLoader(trainlist,shuffle=True,batch_size=gpb)
testloader = pyg.loader.DataLoader(testlist,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sig = tc.nn.Sigmoid() #define sigmoid function for transforming outputs

#define model
class GC(torch.nn.Module):
    def __init__(self):
        intra_channels,out_channels = 48,32
        super().__init__()
        self.conv1 = GraphConv(-1, intra_channels)
        self.conv2 = GraphConv(intra_channels, intra_channels)
        self.conv3 = GraphConv(intra_channels,out_channels)
        self.sig = tc.nn.Sigmoid() #define sigmoid function for transforming outputs

    def forward(self,data,data_x_dict, data_edge_index_dict, data_edge_attr_dict):
        x, a, w = data_x_dict, data_edge_index_dict, data_edge_attr_dict
        #l1
        x = self.conv1(x,a,w)
        x = F.relu(x)
        x = F.dropout(x,p=0.2,training=self.training)        
        #l2
        x = self.conv2(x,a,w)
        x = F.relu(x)
        x = F.dropout(x,p=0.2,training=self.training)
        #l3
        x = self.conv3(x,a,w)
        
        return x

#initialize model from above class
data = testlist[0].to(device)
model = GC()
model = to_hetero(model, data.metadata(), aggr='sum')
model = model.to(device)
multilabel = tc.nn.BCEWithLogitsLoss()

learn1 = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
learn2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
learn3 = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-3)
learn4 = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)
learn5 = torch.optim.Adam(model.parameters(), lr=3e-6, weight_decay=1e-3)
learn6 = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-3)
model.train()

#variables for epoch testing
pred_comp_train,corr_comp_train = [],[]
pred_comp_test,corr_comp_test = [],[]
subloader = pyg.loader.DataLoader(testlist,shuffle=True)
eval_bool = False

desig = 'hg5_udep_hsbb'
for epoch in range(0,2001):
    start = dtime()
    if epoch%10==0:
        eval_bool = True
        pred_thismodel_train,corr_thismodel_train = [],[]

    epoch_loss = 0
    if epoch<50:
        optimizer = learn1
    elif epoch<150:
        optimizer = learn2
    elif epoch<250:
        optimizer = learn3
    elif epoch<350:
        optimizer = learn4
    elif epoch<900:
        optimizer = learn5
    else:
        optimizer = learn6

    for data in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data,data.x_dict,data.edge_index_dict, data.edge_attr_dict)

        numof_g = len(data['rna']['batch'])
        tp_pool = torch.empty((1,32*numof_g))
        for g in range(numof_g):
            id_start = 16*g
            id_end = id_start + 16
            e_start,e_end = data['enhancer'].ptr[g],data['enhancer'].ptr[g+1]
            tp_pool_pergraph = out['perturbation'][id_start:id_end,:].sum(dim=0)+out['enhancer'][e_start:e_end,:].sum(dim=0)+out['promoter'][g,:].sum(dim=0)
            tp_pool_pergraph = tp_pool_pergraph.reshape(1,32)
            tp_pool[0,2*id_start:2*id_end]= tp_pool_pergraph
        pred = tp_pool.to(device)
        correct = get_oh(data['y']).to(device)
        #scales trin values to be equal across each batch and average to 1
        weight = data['y'].detach().clone()
        ud_bias = 0.1
        up_fraction,down_fraction = torch.where(weight==2,1,0).sum()/len(weight)*(1-ud_bias), torch.where(weight==0,1,0).sum()/len(weight)*(1-ud_bias)
        neu_fraction = 1-up_fraction-down_fraction
        up_weight,neu_weight,down_weight = 1/(3*up_fraction),1/(3*neu_fraction),1/(3*down_fraction)
        #NOTA BENE: if any trin categories are not present in batch (very unlikely), these weights become invalid
        weight = torch.where(weight==2,up_weight,weight)
        weight = torch.where(weight==1,neu_weight,weight)
        weight = torch.where(weight==0,down_weight,weight)
        #resize weight to match oh encoding
        weight_oh = torch.empty((2*len(weight)))
        for entry in range(len(weight)):
            weight_oh[2*entry],weight_oh[2*entry+1] = weight[entry],weight[entry]
        multilabel = torch.nn.BCEWithLogitsLoss(weight=weight_oh.to(device)) #can also add in pos_weight to, i.e. fix NC/GHMT bias
        loss = multilabel(pred,correct)
        epoch_loss+=loss
        loss.backward()
        optimizer.step()

        #save pred and corr for eval.
        if eval_bool:
            saved_pred = pred.clone().detach()
            pred_thismodel_train.append(tc.where(saved_pred>0,1,0).to('cpu'))
            corr_thismodel_train.append(correct.to('cpu'))
    if eval_bool:
        #save train data eval.
        pred_comp_train.append(pred_thismodel_train)
        corr_comp_train.append(corr_thismodel_train)
        #perform and save test data eval.
        model.eval()
        pred_thismodel_test,corr_thismodel_test = [],[]
        for data in subloader:
            data = data.to(device)
            numof_g = len(data['rna']['batch'])
            out = model(data,data.x_dict,data.edge_index_dict,data.edge_attr_dict)

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
                pred = tp_pool.to(device)
                pred = torch.where(pred>=0,1,0)
                correct = get_oh(data['y']).to(device)

                inten = pred==correct
                pred_thismodel_test.append(pred.to('cpu'))
                corr_thismodel_test.append(correct.to('cpu'))
        pred_comp_test.append(pred_thismodel_test)
        corr_comp_test.append(corr_thismodel_test)
        model.train()
        eval_bool = False
        
    duration = dtime()-start
    with open(desig+'_outputs.txt','a') as f:
        f.write('\nDURATION: '+str(duration)+'\nLOSS: '+str(float(epoch_loss)))

    with open(desig+'_iter'+str(epoch)+'.pkl', 'wb') as dl_file:
        pkl.dump(model,dl_file)

#save pred/corr compendiums
pc_train = [pred_comp_train,corr_comp_train]
#for reformatting train data results (for hg4 onward)
tpl,tcl = pc_train[0],pc_train[1]
tpl_refor,tcl_refor = [],[]
for model in tpl:
    model_refor = []
    for pelem in model:
        for gpb in range(int(len(pelem[0])/32)-1): #range up to batch size
            model_refor.append(tc.Tensor(pelem[0][gpb*32:(gpb+1)*32]).clone().detach().reshape((1,32)))
    tpl_refor.append(model_refor)
for model in tcl:
    model_refor = []
    for celem in model:
        for gpb in range(int(len(celem[0])/32)-1): #range up for batch size
            model_refor.append(tc.Tensor(celem[0][gpb*32:(gpb+1)*32]).clone().detach().reshape((1,32)))
    tcl_refor.append(model_refor)
pc_train_refor = [tpl_refor,tcl_refor]
with open(desig+'_cputrain_pclists.pkl','wb') as trainfile:
    pkl.dump(pc_train_refor,trainfile)
pc_test = [pred_comp_test,corr_comp_test]
with open(desig+'_cpu_pclists.pkl','wb') as testfile:
    pkl.dump(pc_test,testfile)
