
# coding: utf-8

import numpy as np
import scanpy as sc
import pandas as pd
import pickle as pkl
import scipy.sparse as csr
from scipy.stats import gaussian_kde
import scipy.stats as ss
import math
from timeit import default_timer

#Combined library
lib = 'co'
#load in filtered files
rna_lib = sc.read('corna_filtered.h5ad')
atac_lib = sc.read('coatac_filtered.h5ad')
pert_lib = pkl.load(open('copert_filtered.pkl','rb'))
#load in com,cel,cou
ccc = pkl.load(open('coccc.pkl','rb'))

#load in new prom_overlaps and promenh_overlaps with 1000bp offset
#   [tss_ind,rna_ind,[atac_inds]] for enhol/promenhol
#   [tss_ind,rna_ind,atac_ind] for promol
promol = pkl.load(open('co1000ce_prom_overlaps.pkl','rb'))
promenhol = pkl.load(open('co1000ce_promenh_overlaps.pkl','rb'))
enhol = pkl.load(open('co1000ce_enh_overlaps.pkl','rb'))

ce_rna = pkl.load(open('conce_rna.pkl','rb'))                   #ce_rna but CPM normalized
zce_rna = pkl.load(open('cozce_rna.pkl','rb'))                  #nce_rna but z-scored
ce_atac = pkl.load(open('conce_atac.pkl','rb'))                 #ce_atac but binarized and then CPM normalized
ce_tss = pkl.load(open('coce_tss.pkl','rb'))                    #[chromosome,position,gene_name]


# Supporting Functions

#for un-lning zce_rna.raw
print("original type and shape:",type(zce_rna.raw.X),np.shape(zce_rna.raw.X))
print('original values:',zce_rna.raw.X.todense())
zce_rna.raw.X.data[:] = np.e**zce_rna.raw.X.data-1
print("new type and shape:",type(zce_rna.raw.X),np.shape(zce_rna.raw.X))
print('new values:',zce_rna.raw.X.todense())

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
    N = np.sum(cellsin)                 #input feature
    n = np.sum(cellsout)                #output feature
    M = np.sum(cellsall)                #total
    
    #find p-val
    pval = ss.hypergeom.cdf(k,M,n,N)
    t1pval = min(pval,1-pval)*2           #doesn't distinguish between enrichment or depletion, need foldchange to do so
    
    return t1pval
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
                cellsin_bool = np.array([cell in cellsin for cell in cellsall])
                cellsout_bool = np.array([cell in cellsout for cell in cellsall])
                cellsall_bool = np.array([cell in tempcellsall for cell in cellsall])
                
                #function calls
                pv = hyperpvals(cellsin_bool, cellsout_bool, cellsall_bool, rand_cellsout)
                fc = metacellfc(cellsin,featureout,tempcellsall,arrayout)
                
                inout_stat.append([pv,fc])
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

                    inout_stat.append([pv,fc])
                tss_stats.append(inout_stat)
            total_inout_stats.append(tss_stats)

    #delete return statement
    return total_inout_stats
        
    
#     with open('total_'+intype+outtype+'_stats_1_fake.pkl', 'wb') as tios_file:
#         pkl.dump(total_inout_stats,tios_file)
#__________________________________________________________________________________________________________________________________________________________________________

stype = ['perturbation','enhancer','promoter','rna']

#make statgetter() calls and save all results
#te      -note that this call may take a very long time and need to be run in batches
start = default_timer()
tios = statgetter(stype[0],stype[1])
print(default_timer()-start)
with open('total_co_'+stype[0]+stype[1]+'_stats.pkl', 'wb') as tios_file:
    pkl.dump(tios,tios_file)
#tp
start = default_timer()
tios = statgetter(stype[0],stype[2])
print(default_timer()-start)
with open('total_co_'+stype[0]+stype[2]+'_stats.pkl', 'wb') as tios_file:
    pkl.dump(tios,tios_file)
#tr
start = default_timer()
tios = statgetter(stype[0],stype[3])
print(default_timer()-start)
with open('total_co_'+stype[0]+stype[3]+'_stats.pkl', 'wb') as tios_file:
    pkl.dump(tios,tios_file)
#ep
start = default_timer()
tios = statgetter(stype[1],stype[2])
print(default_timer()-start)
with open('total_co_'+stype[1]+stype[2]+'_stats.pkl', 'wb') as tios_file:
    pkl.dump(tios,tios_file)
#er
start = default_timer()
tios = statgetter(stype[1],stype[3])
print(default_timer()-start)
with open('total_co_'+stype[1]+stype[3]+'_stats.pkl', 'wb') as tios_file:
    pkl.dump(tios,tios_file)
#pr
start = default_timer()
tios = statgetter(stype[2],stype[3])
print(default_timer()-start)
with open('total_co_'+stype[2]+stype[3]+'_stats.pkl', 'wb') as tios_file:
    pkl.dump(tios,tios_file)
