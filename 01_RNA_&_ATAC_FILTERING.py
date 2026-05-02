
# coding: utf-8

#package imports
import numpy as np
import scanpy as sc
import pandas as pd
import pickle as pkl
import seaborn as sb
import scipy.sparse as csr
import doubletdetection as dd

# load in matrices___________________________________________________________________________________________________
comul = sc.read('multiome_matrix.h5')
comul_lib.var_names_make_unique()

blpert_lib = pd.read_csv('bl_cell_perturbation.csv.gz',index_col=0).T
lnpert_lib = pd.read_csv('ln_cell_perturbation.csv.gz',index_col=0).T
copert_lib = pd.concat([blpert_lib,lnpert_lib])

#split multiome into RNA and ATAC
corna_lib = comul_lib[:,comul_lib.var['modality'] == 'Gene Expression'].copy()
coatac_lib = comul_lib[:,comul_lib.var['modality'] == 'Peaks'].copy()

# rna filtering______________________________________________________________________________________________________

rna_lib = corna_lib
print('Total number of cells: {:d}'.format(rna_lib.n_obs))
rna_lib.obs['n_counts'] = [np.sum(rna_lib.X[cell,:]) for cell in range(len(rna_lib.obs_names))]

#remove doublets__________note: 10x expects 0.8% doublets per 1k cells loaded, this is a combined library so expect less, BL+LN doublet counts give ~1,000
raw_counts = rna_lib.X
clf = dd.BoostClassifier()
labels = clf.fit(raw_counts).predict(p_thresh=1e-7,voter_thresh=0.95) #default is 1e-7 and 0.9
# higher means more likely to be doublet
scores = clf.doublet_score()
rna_lib.obs['doublet'] = labels
keep_lib = rna_lib[rna_lib.obs['doublet'] == 0]
leave_lib = rna_lib[rna_lib.obs['doublet'] == 1]
rna_lib = keep_lib
print('cells predicted to be singlets: ',len(keep_lib.obs_names),'     cells predicted to be doublets: ',len(leave_lib.obs_names))

#remove cells with high mito portion
mito_genes = [varname for varname in rna_lib.var_names if 'mt-' in varname]
rna_lib.obs['mito_portion'] = [np.sum(rna_lib[cell,mito_genes].X)/(rna_lib.obs['n_counts'][cell]) for cell in rna_lib.obs_names]
rna_lib = rna_lib[rna_lib.obs['mito_portion'] < 0.25]
print('Number of cells after mito filter: {:d}'.format(rna_lib.n_obs))

#remove cells with too low or high read counts
sc.pp.filter_cells(rna_lib, min_counts = 1000)
print('Number of cells after min count filter: {:d}'.format(rna_lib.n_obs))
sc.pp.filter_cells(rna_lib, max_counts = 50000)
print('Number of cells after max count filter: {:d}'.format(rna_lib.n_obs))

#remove genes with too few cells
print('Total number of genes: {:d}'.format(rna_lib.n_vars))
sc.pp.filter_genes(rna_lib, min_cells=1)
print('Number of genes after cell filter: {:d}'.format(rna_lib.n_vars))

corna_f = rna_lib.copy()
coatac_lib = coatac_lib[corna_f.obs_names]

atac_lib = coatac_lib
rna_lib = corna_f
pert_lib = copert_lib

# atac filtering_________________________________________________________________________________________________________________________
#binarize ATAC data
nzlist = np.array(atac_lib.X.nonzero())
for elem in range(len(nzlist[0])):
    atac_lib.X[tuple(nzlist[:,elem])] = 1

#plot and prepare for min-max count filtering
atac_lib.obs['n_counts'] = [np.sum(atac_lib.X[cell,:]) for cell in range(len(atac_lib.obs_names))]
mu = np.mean(np.log10(atac_lib.obs['n_counts']))
stdev = np.std(np.log10(atac_lib.obs['n_counts']))
print(mu,stdev)
sb.violinplot(np.log10(atac_lib.obs['n_counts']))

#keep cells within 2 standard deviations of mean   (~1000 to ~80,000)
print('Total number of cells: {:d}'.format(atac_lib.n_obs),'\nmin threshold=',10**(mu-2*stdev),' max threshold=',10**(mu+2*stdev))
sc.pp.filter_cells(atac_lib, min_counts = 10**(mu-2*stdev))
print('Number of cells after min count filter: {:d}'.format(atac_lib.n_obs))
sc.pp.filter_cells(atac_lib, max_counts = 10**(mu+2*stdev))
print('Number of cells after max count filter: {:d}'.format(atac_lib.n_obs))

ol = [cid for cid in rna_lib.obs_names if (cid in pert_lib.index) and (cid in atac_lib.obs_names)]
atac_lib = atac_lib[ol]
rna_lib = rna_lib[ol]
pert_lib = pert_lib.loc[ol]

#save files at
atac_lib.write('coatac_filtered.h5ad')
rna_lib.write('corna_filtered.h5ad')
with open('copert_filtered.pkl', 'wb') as pert_file:
    pkl.dump(pert_lib,pert_file)
