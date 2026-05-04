
# coding: utf-8

# In[1]:


import numpy as np
import scanpy as sc
import scvi as sv
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
from timeit import default_timer
from scipy.stats import percentileofscore
import random as rn

#dwd
#pval check
k = 199      #input feature AND output feature
N = 1320                   #input feature
n = 477                  #output feature
M = 4367                #total

#find real p-val
pval = ss.hypergeom.cdf(k,M,n,N)
t1pval = min(pval,1-pval)*2
print(t1pval,-np.log10(t1pval+1e-11))


# In[25]:


#dwd
#pval check
k = 577      #input feature AND output feature
N = 1320                   #input feature
n = 1265                  #output feature
M = 4367                #total

#find real p-val
pval = ss.hypergeom.cdf(k,M,n,N)
t1pval = min(pval,1-pval)*2
print(t1pval,-np.log10(t1pval+1e-11))


# In[2]:


#BL
lib = 'bl'
#load in filtered files
rna_lib = sc.read(lib+'rna_filtered.h5ad')
atac_lib = sc.read(lib+'atac_filtered.h5ad')
pert_lib = pkl.load(open(lib+'pert_filtered.pkl','rb'))
#load in com,cel,cou
ccc = pkl.load(open(lib+'ccc.pkl','rb'))
#ccc = pkl.load(open(lib+'pdccc.pkl','rb'))

#load in new prom_overlaps and promenh_overlaps with 1000bp offset
#   [tss_ind,rna_ind,[atac_inds]] for enhol/promenhol
#   [tss_ind,rna_ind,atac_ind] for promol
promol = pkl.load(open(lib+'1000ce_prom_overlaps.pkl','rb'))
promenhol = pkl.load(open(lib+'1000ce_promenh_overlaps.pkl','rb'))
enhol = pkl.load(open(lib+'1000ce_enh_overlaps.pkl','rb'))

ce_rna = pkl.load(open(lib+'nce_rna.pkl','rb'))                   #ce_rna but CPM normalized
zce_rna = pkl.load(open(lib+'zce_rna.pkl','rb'))                  #nce_rna but z-scored
ce_atac = pkl.load(open(lib+'nce_atac.pkl','rb'))                 #ce_atac but binarized and then CPM normalized
ce_tss = pkl.load(open(lib+'ce_tss.pkl','rb'))                    #[chromosome,position,gene_name]


# In[2]:


#LN
lib = 'ln'
#load in filtered files
rna_lib = sc.read(lib+'rna_filtered.h5ad')
atac_lib = sc.read(lib+'atac_filtered.h5ad')
pert_lib = pkl.load(open(lib+'pert_filtered.pkl','rb'))
#load in com,cel,cou
ccc = pkl.load(open(lib+'ccc.pkl','rb'))
#ccc = pkl.load(open(lib+'pdccc.pkl','rb'))

#load in new prom_overlaps and promenh_overlaps with 1000bp offset
#   [tss_ind,rna_ind,[atac_inds]] for enhol/promenhol
#   [tss_ind,rna_ind,atac_ind] for promol
promol = pkl.load(open(lib+'1000ce_prom_overlaps.pkl','rb'))
promenhol = pkl.load(open(lib+'1000ce_promenh_overlaps.pkl','rb'))
enhol = pkl.load(open(lib+'1000ce_enh_overlaps.pkl','rb'))

ce_rna = pkl.load(open(lib+'nce_rna.pkl','rb'))                   #ce_rna but CPM normalized
zce_rna = pkl.load(open(lib+'zce_rna.pkl','rb'))                  #nce_rna but z-scored
ce_atac = pkl.load(open(lib+'nce_atac.pkl','rb'))                 #ce_atac but binarized and then CPM normalized
ce_tss = pkl.load(open(lib+'ce_tss.pkl','rb'))                    #[chromosome,position,gene_name]


# In[2]:


#Combined
lib = 'co'
#load in filtered files
rna_lib = sc.read(lib+'rna_filtered.h5ad')
atac_lib = sc.read(lib+'atac_filtered.h5ad')
pert_lib = pkl.load(open(lib+'pert_filtered.pkl','rb'))
#load in com,cel,cou
ccc = pkl.load(open(lib+'ccc.pkl','rb'))
#ccc = pkl.load(open(lib+'pdccc.pkl','rb'))

#load in new prom_overlaps and promenh_overlaps with 1000bp offset
#   [tss_ind,rna_ind,[atac_inds]] for enhol/promenhol
#   [tss_ind,rna_ind,atac_ind] for promol
promol = pkl.load(open(lib+'1000ce_prom_overlaps.pkl','rb'))
promenhol = pkl.load(open(lib+'1000ce_promenh_overlaps.pkl','rb'))
enhol = pkl.load(open(lib+'1000ce_enh_overlaps.pkl','rb'))

ce_rna = pkl.load(open(lib+'nce_rna.pkl','rb'))                   #ce_rna but CPM normalized
zce_rna = pkl.load(open(lib+'zce_rna.pkl','rb'))                  #nce_rna but z-scored
ce_atac = pkl.load(open(lib+'nce_atac.pkl','rb'))                 #ce_atac but binarized and then CPM normalized
ce_tss = pkl.load(open(lib+'ce_tss.pkl','rb'))                    #[chromosome,position,gene_name]


# ## Supporting Functions

# In[60]:


#for un-lning zce_rna.raw
print("original type and shape:",type(zce_rna.raw.X),np.shape(zce_rna.raw.X))
print('original values:',zce_rna.raw.X.todense())

zce_rna.raw.X.data[:] = np.e**zce_rna.raw.X.data-1

print("new type and shape:",type(zce_rna.raw.X),np.shape(zce_rna.raw.X))
print('new values:',zce_rna.raw.X.todense())


# In[24]:


#__________________________________________________________________________________________________________________________________________________________________________
#given list-like of cells in input and the index (featureout) of the feature of interest in anndata (arrayout), return foldchange using metacell
def metacellfc(cellsin,featureout,cellsall,arrayout):
    if (len(cellsin) == 0) or (len(cellsin) == len(cellsall)):
        foldchange = 0
    else:
        #split array into pos and neg
        cellsin,cellsall = np.array(cellsin),np.array(cellsall)
        notcellsin = np.setdiff1d(cellsall,cellsin)
        
        #create metacells
        metacellpos = np.sum(arrayout[cellsin,featureout].X.todense())
        metacellneg = np.sum(arrayout[notcellsin,featureout].X.todense())
        
        #record foldchange
        foldpos = metacellpos/len(cellsin)
        foldneg = metacellneg/len(notcellsin)
        
        if foldneg == 0:
            foldchange = 1e2
        else:
            foldchange = foldpos/foldneg
        
    return foldchange        
#__________________________________________________________________________________________________________________________________________________________________________
#given list-like of cells in input and output category (and total population), returns 2-sided hypergeometric p-value 
def hyperpvals(cellsin,cellsout,cellsall,cellsrand):
    #need to subset if intype is pert
    if np.sum(cellsall) < len(cellsall):
        cellsin = cellsin & cellsall
        cellsout = cellsout & cellsall
        cellsrand = [fakeout & cellsall for fakeout in cellsrand]
    
    k = np.sum(cellsin & cellsout)      #input feature AND output feature
    N = np.sum(cellsin)                   #input feature
    n = np.sum(cellsout)                  #output feature
    M = np.sum(cellsall)                  #total
    
    #temp, remove when done
    print(k,N,n,M)
    
    #find real p-val
    pval = ss.hypergeom.cdf(k,M,n,N)
    t1pval = min(pval,1-pval)*2           #doesn't distinguish between enrichment or depletion, need foldchange to do so
    
    #generate 1,000 fake p-vals
    t1rpvals = []
    for fakeout in cellsrand:
        k = np.sum(cellsin & fakeout)    #input feature AND output feature
        n = np.sum(fakeout)                #output feature 
        #N and M do not change from real
        rand_pval = ss.hypergeom.cdf(k,M,n,N)
        rand_t1pval = min(rand_pval,1-rand_pval)*2   #doesn't distinguish between enrichment or depletion, need foldchange to do so
        t1rpvals.append(rand_t1pval)
    
    #find q-value of the real p-val
    adjval = percentileofscore(t1rpvals,t1pval)
    
    return t1pval,adjval
#__________________________________________________________________________________________________________________________________________________________________________
#given type of data in and out from ['perturbation','enhancer','promoter','rna'], return cell or feature lists and matrices to use them with
def matrixgetter(intype,outtype):
    if intype == 'perturbation':
        listin = [ccc[1]]*len(promenhol)
        matrixin = []                  #subsequent functions will assume an empty matrixin means listin contains fully prepared cell lists
    elif intype == 'enhancer':         #returns list list
        listin = [elem[2] for elem in enhol]
        matrixin = ce_atac
    elif intype == 'promoter':         #returns list list
        listin = [[elem[2]] for elem in promol]
        matrixin = ce_atac
        
    if outtype == 'enhancer':          #returns list list
        listout = [elem[2] for elem in enhol]
        matrixout = ce_atac
    elif outtype == 'promoter':        #returns int list
        listout = [elem[2] for elem in promol]
        matrixout = ce_atac
    elif outtype == 'rna':             #returns int list
        listout = [elem[1] for elem in promol]
        matrixout = zce_rna
        
    return listin,matrixin,listout,matrixout
#__________________________________________________________________________________________________________________________________________________________________________
def statgetter(intype,outtype):
    #configure matrices
    listin,matrixin,listout,matrixout = matrixgetter(intype,outtype)
    if type(matrixout.X) != np.ndarray:
        splitout = np.asarray(matrixout.X.todense())
    else:
        splitout = matrixout.X
    if len(matrixin):
        splitin = np.asarray(matrixin.X.todense())
    cellsall = matrixout.obs_names #for stat gen
    
    #initialize generator for later
    rng = np.random.default_rng()
    
    #iterate through all outgroups
    total_inout_stats = []
    if type(listout[0]) == int:        #only need 1 layer of iteration if outtype is int-list
        for elemidx in range(100): #len(listout)
            featureout = listout[elemidx]                               #for stat gen
            if np.shape(matrixout.raw):                                 #
                arrayout = matrixout.raw                                #
            else:                                                       #
                arrayout = matrixout                                    #
            cellsout = matrixout.obs_names[splitout[:,featureout] > 0]  #for stat gen
            
            #generate list of random cellsout bools, pulled from features not on the same chromosome as real cellsout
            feature_chr = matrixout.var['chr'][featureout]
            rand_matrixout = matrixout[:,matrixout.var['chr'] != feature_chr]
            rand_feature_list = rng.permutation(len(matrixout.var))[:1000]
            rand_cellsout = [splitout[:,rand_feature] > 0 for rand_feature in rand_feature_list]
            
            #iterate through all ingroups
            inout_stat = []
            for elemin in listin[elemidx]:
                if len(matrixin):                                       #for stat gen
                    cellsin = matrixin.obs_names[splitin[:,elemin] > 0] #
                    tempcellsall = cellsall                             #
                else:                                                   #
                    cellsin = elemin                                    #for stat gen
                    #redefine cellsall
                    if len(elemin) != len(listin[elemidx][0]):
                        tempcellsall = cellsin+listin[elemidx][0]
                    else:
                        tempcellsall = cellsall
                
                #make boolean arrays for hyperpvals()
                #cellsin_bool = np.array([cell in cellsin for cell in cellsall])
                #cellsout_bool = np.array([cell in cellsout for cell in cellsall])
                #cellsall_bool = np.array([cell in tempcellsall for cell in cellsall])
                
                #function calls
                #pv,av = hyperpvals(cellsin_bool, cellsout_bool, cellsall_bool, rand_cellsout)
                fc = metacellfc(cellsin,featureout,tempcellsall,arrayout)
                
                inout_stat.append([fc])
            total_inout_stats.append(inout_stat)
        
    #only entered if in=pert and out=enh
    elif type(listout[0]) == list:     #need additional layer of iteration if outtype is list-list
        for tssidx in range(len(listout)):   #len(listout) #ADD INDEX HERE!!!
            print(tssidx)
            tss_stats = []
            for enhidx in listout[tssidx]:
                featureout = enhidx                                        #for stat gen
                arrayout = matrixout                                       #
                cellsout = matrixout.obs_names[splitout[:,featureout] > 0] #for stat gen
                
                #generate list of random cellsout bools, pulled from features not on the same chromosome as real cellsout
                feature_chr = matrixout.var['chr'][featureout]
                rand_matrixout = matrixout[:,matrixout.var['chr'] != feature_chr]
                rand_feature_list = rng.permutation(len(matrixout.var))[:1000]
                rand_cellsout = [splitout[:,rand_feature] > 0 for rand_feature in rand_feature_list]
            
                #iterate through all ingroups
                inout_stat = []
                for elemin in listin[tssidx]:
                    cellsin = elemin                                       #for stat gen                          
                    
                    #redefine cellsall
                    if len(elemin) != len(listin[tssidx][0]):
                        tempcellsall = cellsin+listin[tssidx][0]
                    else:
                        tempcellsall = cellsall
                    
                    #make boolean arrays for hyperpvals()
                    cellsin_bool = np.array([cell in cellsin for cell in cellsall])
                    cellsout_bool = np.array([cell in cellsout for cell in cellsall])
                    cellsall_bool = np.array([cell in tempcellsall for cell in cellsall])
                    
                    pv,av = hyperpvals(cellsin_bool, cellsout_bool, cellsall_bool, rand_cellsout)
                    fc = metacellfc(cellsin,featureout,tempcellsall,arrayout)

                    inout_stat.append([fc])
                tss_stats.append(inout_stat)
            total_inout_stats.append(tss_stats)
            
    #LEFT OFF HERE, SEE COMMENT BELOW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #delete return statement, uncomment pv&av calls and random&bool arrays, add pv&av back to total_inout_stats
    return total_inout_stats
        
    
#     with open('total_'+intype+outtype+'_stats_1_fake.pkl', 'wb') as tios_file:
#         pkl.dump(total_inout_stats,tios_file)
#__________________________________________________________________________________________________________________________________________________________________________


# In[ ]:


stype = ['perturbation','enhancer','promoter','rna']
start = default_timer()
tios = statgetter(stype[0],stype[1])
print(default_timer()-start)
print(tios)


# In[4]:


enhol


# In[21]:


enhol_inds = [enhol[which][2] for which in range(len(enhol)) if ce_rna.var_names[enhol[which][1]]=='Tnnt2']
tlist = enhol_inds[0]


# In[ ]:


len(tlist)


# ### LOAD IN FIXED RNA FCs AND SWAP OUT FOR OLD FCs, SAVING OLD FCs IN NEW FILES

# In[3]:


#load data
# lib = 'bl'
# lib = 'ln'
lib = 'co'
tr = pkl.load(open('total_'+lib+'_perturbationrna_stats.pkl','rb'))
er = pkl.load(open('total_'+lib+'_enhancerrna_stats.pkl','rb'))
pr = pkl.load(open('total_'+lib+'_promoterrna_stats.pkl','rb'))
trfc = pkl.load(open('FCredo_'+lib+'_perturbationrna_stats.pkl','rb'))
erfc = pkl.load(open('FCredo_'+lib+'_enhancerrna_stats.pkl','rb'))
prfc = pkl.load(open('FCredo_'+lib+'_promoterrna_stats.pkl','rb'))

#generate files to store old values
old_tr,old_er,old_pr = tr.copy(),er.copy(),pr.copy()
with open('old_total_'+lib+'_perturbationrna_stats.pkl', 'wb') as tios_file:
    pkl.dump(old_tr,tios_file)
with open('old_total_'+lib+'_enhancerrna_stats.pkl', 'wb') as tios_file:
    pkl.dump(old_er,tios_file)
with open('old_total_'+lib+'_promoterrna_stats.pkl', 'wb') as tios_file:
    pkl.dump(old_pr,tios_file)

#replace FCs
for locus in range(len(tr)):
    for comparison in range(len(tr[locus])):
        tr[locus][comparison][-1] = trfc[locus][comparison][0]
    for comparison in range(len(er[locus])):
        er[locus][comparison][-1] = erfc[locus][comparison][0]
    for comparison in range(len(pr[locus])):
        pr[locus][comparison][-1] = prfc[locus][comparison][0]

#save new files
with open('total_'+lib+'_perturbationrna_stats.pkl', 'wb') as tios_file:
    pkl.dump(tr,tios_file)
with open('total_'+lib+'_enhancerrna_stats.pkl', 'wb') as tios_file:
    pkl.dump(er,tios_file)
with open('total_'+lib+'_promoterrna_stats.pkl', 'wb') as tios_file:
    pkl.dump(pr,tios_file)


# ## LOAD IN AND CONCAT DATA FROM SBATCH SUBMISSIONS

# In[3]:


TE_split = 12 #based on however many jobs TE had to be split into
lib = 'bl'
blte = []
#manually add first one due to overwrite mess up, hence starting the loop from '2'
te1a = pkl.load(open('total_'+lib+'_perturbationenhancer_stats1a.pkl','rb'))
te1b = pkl.load(open('total_'+lib+'_perturbationenhancer_stats1b.pkl','rb'))
blte+=te1a
blte+=te1b
for file_num in range(2,TE_split+1):
    te = pkl.load(open('total_'+lib+'_perturbationenhancer_stats'+str(file_num)+'.pkl','rb'))
    #print(len(blte),file_num)
    blte+=te
    #print(len(blte),file_num)
bltp = pkl.load(open('total_'+lib+'_perturbationpromoter_stats.pkl','rb'))
bltr = pkl.load(open('total_'+lib+'_perturbationrna_stats.pkl','rb'))
blep = pkl.load(open('total_'+lib+'_enhancerpromoter_stats.pkl','rb'))
bler = pkl.load(open('total_'+lib+'_enhancerrna_stats.pkl','rb'))
blpr = pkl.load(open('total_'+lib+'_promoterrna_stats.pkl','rb'))


# In[3]:


TE_split = 11 #based on however many jobs TE had to be split into
lib = 'ln'
lnte = []
for file_num in range(1,TE_split+1):
    te = pkl.load(open('split_pe_files/total_'+lib+'_perturbationenhancer_stats'+str(file_num)+'.pkl','rb'))
    lnte+=te
# lntp = pkl.load(open('total_'+lib+'_perturbationpromoter_stats.pkl','rb'))
# lntr = pkl.load(open('total_'+lib+'_perturbationrna_stats.pkl','rb'))
# lnep = pkl.load(open('total_'+lib+'_enhancerpromoter_stats.pkl','rb'))
# lner = pkl.load(open('total_'+lib+'_enhancerrna_stats.pkl','rb'))
# lnpr = pkl.load(open('total_'+lib+'_promoterrna_stats.pkl','rb'))


# In[11]:


with open('total_bl_perturbationenhancer_stats.pkl', 'wb') as tios_file:
    pkl.dump(blte,tios_file)


# In[5]:


with open('total_ln_perturbationenhancer_stats.pkl', 'wb') as tios_file:
    pkl.dump(lnte,tios_file)


# In[13]:


newb = pkl.load(open('total_bl_perturbationenhancer_stats.pkl', 'rb'))


# In[6]:


newl = pkl.load(open('total_ln_perturbationenhancer_stats.pkl', 'rb'))


# In[16]:


print(len(newb),len(newl))


# In[7]:


len(newl)


# In[6]:


print(len(bltp),len(bltr),len(blep),len(bler),len(blpr))


# In[8]:


print(len(lnte),len(lntp),len(lntr),len(lnep),len(lner),len(lnpr))


# In[ ]:


blep

