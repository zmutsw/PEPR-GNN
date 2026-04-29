
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
import doubletdetection as dd

# set a working directory for saving plots
os.chdir('/project/GCRB/Hon_lab/s437603/data/ghmt_multiome/analysis/')


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load in Matrices

# In[3]:


# BOXUN MULTIOME
# RNA = BL107
# ATAC = BL105
# PERT = BL109, BL129

# LANDON MULTIOME
# RNA = LN10, LN12
# ATAC = LN9, LN11
# PERT = LW254, LW255, LW256, LW257

data_dir = '/project/GCRB/Hon_lab/s437603/data/ghmt_multiome'
mat = 'matrix_cell_identity.csv.gz'

# TM
# pc_lib = sc.read(genex_data_dir + 'tabula-muris/droplet/adata.h5ad')
# pc_lib = pc_lib[pc_lib.obs['cell_ontology_class_formatted'] == 'cardiac_muscle_cell']
# pc_lib.var_names = [gene.partition('_')[2] for gene in pc_lib.var_names]
# pc_lib.var_names_make_unique()

# BL library___________________________________________________________________________________________________
blmul_lib = sv.data.read_10x_multiome(data_dir+'/bl_multiome/BL107_BL105_10x/outs/filtered_feature_bc_matrix/')
blmul_lib.var_names_make_unique()
blpert_lib = pd.read_csv(data_dir+'/perturbation/bl_multiome_pert/demultiplexed_knee/'+mat,index_col=0).T

#split multiome into RNA and ATAC
blrna_lib = blmul_lib[:,blmul_lib.var_names[0:32285]].copy()
blatac_lib = blmul_lib[:,blmul_lib.var_names[32285:]].copy()


# LN library_________________________________________________________________________________________________
lnmul_lib = sv.data.read_10x_multiome(data_dir+'/ln_multiome/LN9_to_12_10x/outs/filtered_feature_bc_matrix/')
lnmul_lib.var_names_make_unique()
lnpert_lib = pd.read_csv(data_dir+'/perturbation/ln_multiome_pert/ln_demult/'+mat,index_col=0).T

#split multiome into RNA and ATAC
lnrna_lib = lnmul_lib[:,lnmul_lib.var_names[0:32285]].copy()
lnatac_lib = lnmul_lib[:,lnmul_lib.var_names[32285:]].copy()


# ## RNA FILTERING

# In[4]:


rna_lib = blrna_lib
print('Total number of BL cells: {:d}'.format(rna_lib.n_obs))
rna_lib.obs['n_counts'] = [np.sum(rna_lib.X[cell,:]) for cell in range(len(rna_lib.obs_names))]

#remove doublets__________note: 10x expects 0.8% doublets per 1k cells loaded__________so with 5753 cells we expect 265 doublets_______________________________________________________________
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

#remove cells with high mito portion___________________________________________________________________________________________________________________________________________________________
mito_genes = [varname for varname in rna_lib.var_names if 'mt-' in varname]
rna_lib.obs['mito_portion'] = [np.sum(rna_lib[cell,mito_genes].X)/(rna_lib.obs['n_counts'][cell]) for cell in rna_lib.obs_names]
rna_lib = rna_lib[rna_lib.obs['mito_portion'] < 0.25]
print('Number of cells after mito filter: {:d}'.format(rna_lib.n_obs))

#remove cells with too low or high read counts_________________________________________________________________________________________________________________________________________________
sc.pp.filter_cells(rna_lib, min_counts = 1000)
print('Number of cells after min count filter: {:d}'.format(rna_lib.n_obs))
sc.pp.filter_cells(rna_lib, max_counts = 50000)
print('Number of cells after max count filter: {:d}'.format(rna_lib.n_obs))

#remove genes with too few cells_______________________________________________________________________________________________________________________________________________________________
print('Total number of genes: {:d}'.format(rna_lib.n_vars))
sc.pp.filter_genes(rna_lib, min_cells=1)
print('Number of genes after cell filter: {:d}'.format(rna_lib.n_vars))

blrna_f = rna_lib.copy()
blatac_lib = blatac_lib[blrna_f.obs_names]


# In[5]:


rna_lib = lnrna_lib
print('Total number of LN cells: {:d}'.format(rna_lib.n_obs))
rna_lib.obs['n_counts'] = [np.sum(rna_lib.X[cell,:]) for cell in range(len(rna_lib.obs_names))]

#remove doublets__________note: 10x expects 0.8% doublets per 1k cells loaded__________so with 4627 cells we expect 171 doublets______________________________________________________________
raw_counts = rna_lib.X
clf = dd.BoostClassifier()
labels = clf.fit(raw_counts).predict(p_thresh=1e-7,voter_thresh=0.95) #default is 1e-7 and 0.9
# higher means more likely to be doublet
scores = clf.doublet_score()
rna_lib.obs['doublet'] = labels
keep_lib = rna_lib[rna_lib.obs['doublet'] == 0]
leave_lib = rna_lib[rna_lib.obs['doublet'] == 1]
rna_lib = keep_lib
print('cells predicted to be singlets: ',keep_lib.n_obs,'     cells predicted to be doublets: ',leave_lib.n_obs)

#remove cells with high mito portion___________________________________________________________________________________________________________________________________________________________
mito_genes = [varname for varname in rna_lib.var_names if 'mt-' in varname]
rna_lib.obs['mito_portion'] = [np.sum(rna_lib[cell,mito_genes].X)/(rna_lib.obs['n_counts'][cell]) for cell in rna_lib.obs_names]
rna_lib = rna_lib[rna_lib.obs['mito_portion'] < 0.25]
print('Number of cells after mito filter: {:d}'.format(rna_lib.n_obs))

#remove cells with too low or high read counts_________________________________________________________________________________________________________________________________________________
sc.pp.filter_cells(rna_lib, min_counts = 1000)
print('Number of cells after min count filter: {:d}'.format(rna_lib.n_obs))
sc.pp.filter_cells(rna_lib, max_counts = 50000)
print('Number of cells after max count filter: {:d}'.format(rna_lib.n_obs))

#remove genes with too few cells_______________________________________________________________________________________________________________________________________________________________
print('Total number of genes: {:d}'.format(rna_lib.n_vars))
sc.pp.filter_genes(rna_lib, min_cells=1)
print('Number of genes after cell filter: {:d}'.format(rna_lib.n_vars))

lnrna_f = rna_lib.copy()
lnatac_lib = lnatac_lib[lnrna_f.obs_names]


# ## ATAC FILTERING

# In[ ]:


#alternative code for binarizing, didn't use here but may want later
atac_mat = np.array(ce_atac.X.todense().copy())
with np.nditer(atac_mat,op_flags=['readwrite']) as it:
    for x in it:
        if x > 0:
            x[...] = 1


# In[ ]:


#binarize atac data
atac_lib = blatac_lib
rna_lib = blrna_f
pert_lib = blpert_lib
nzlist = np.array(atac_lib.X.nonzero())
for elem in range(len(nzlist[0])):
    atac_lib.X[tuple(nzlist[:,elem])] = 1

#plot and prepare for min-max count filtering
atac_lib.obs['n_counts'] = [np.sum(atac_lib.X[cell,:]) for cell in range(len(atac_lib.obs_names))]
mu = np.mean(np.log10(atac_lib.obs['n_counts']))
stdev = np.std(np.log10(atac_lib.obs['n_counts']))
print(mu,stdev)
sb.violinplot(np.log10(atac_lib.obs['n_counts']))


# In[ ]:


#keep cells within 2 standard deviations of mean   (~1000 to ~80,000)
print('Total number of cells: {:d}'.format(atac_lib.n_obs))
sc.pp.filter_cells(atac_lib, min_counts = 10**(mu-2*stdev))
print('Number of cells after min count filter: {:d}'.format(atac_lib.n_obs))
sc.pp.filter_cells(atac_lib, max_counts = 10**(mu+2*stdev))
print('Number of cells after max count filter: {:d}'.format(atac_lib.n_obs))


# In[ ]:


ol = [cid for cid in rna_lib.obs_names if (cid in pert_lib.index) and (cid in atac_lib.obs_names)]
atac_lib = atac_lib[ol]
rna_lib = rna_lib[ol]
pert_lib = pert_lib.loc[ol]

atac_lib.write('blatac_filtered.h5ad')
rna_lib.write('blrna_filtered.h5ad')
with open('blpert_filtered.pkl', 'wb') as pert_file:
    pkl.dump(pert_lib,pert_file)


# In[ ]:


#binarize atac data
atac_lib = lnatac_lib
rna_lib = lnrna_f
pert_lib = lnpert_lib
nzlist = np.array(atac_lib.X.nonzero())
for elem in range(len(nzlist[0])):
    atac_lib.X[tuple(nzlist[:,elem])] = 1

#plot and prepare for min-max count filtering
atac_lib.obs['n_counts'] = [np.sum(atac_lib.X[cell,:]) for cell in range(len(atac_lib.obs_names))]
mu = np.mean(np.log10(atac_lib.obs['n_counts']))
stdev = np.std(np.log10(atac_lib.obs['n_counts']))
print(mu,stdev)
sb.violinplot(np.log10(atac_lib.obs['n_counts']))


# In[ ]:


#keep cells within 2 standard deviations of mean   (~200 to ~20,000)
print('Total number of cells: {:d}'.format(atac_lib.n_obs))
sc.pp.filter_cells(atac_lib, min_counts = 10**(mu-2*stdev))
print('Number of cells after min count filter: {:d}'.format(atac_lib.n_obs))
sc.pp.filter_cells(atac_lib, max_counts = 10**(mu+2*stdev))
print('Number of cells after max count filter: {:d}'.format(atac_lib.n_obs))


# In[ ]:


ol = [cid for cid in rna_lib.obs_names if (cid in pert_lib.index) and (cid in atac_lib.obs_names)]
atac_lib = atac_lib[ol]
rna_lib = rna_lib[ol]
pert_lib = pert_lib.loc[ol]

atac_lib.write('lnatac_filtered.h5ad')
rna_lib.write('lnrna_filtered.h5ad')
with open('lnpert_filtered.pkl', 'wb') as pert_file:
    pkl.dump(pert_lib,pert_file)


# # COMBO LIBRARY

# In[3]:


data_dir = '/project/GCRB/Hon_lab/s437603/data/ghmt_multiome'
mat = 'matrix_cell_identity.csv.gz'

# load in combo matrices___________________________________________________________________________________________________
comul_lib = sv.data.read_10x_multiome(data_dir+'/combo_multiome/combo_10x/outs/filtered_feature_bc_matrix/')
comul_lib.var_names_make_unique()

blpert_lib = pd.read_csv(data_dir+'/perturbation/bl_multiome_pert/demultiplexed_knee/'+mat,index_col=0).T
lnpert_lib = pd.read_csv(data_dir+'/perturbation/ln_multiome_pert/ln_demult/'+mat,index_col=0).T
copert_lib = pd.concat([blpert_lib,lnpert_lib])

#split multiome into RNA and ATAC
corna_lib = comul_lib[:,comul_lib.var_names[0:32285]].copy()
coatac_lib = comul_lib[:,comul_lib.var_names[32285:]].copy()


# ## RNA FILTERING

# In[8]:


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

#remove cells with high mito portion___________________________________________________________________________________________________________________________________________________________
mito_genes = [varname for varname in rna_lib.var_names if 'mt-' in varname]
rna_lib.obs['mito_portion'] = [np.sum(rna_lib[cell,mito_genes].X)/(rna_lib.obs['n_counts'][cell]) for cell in rna_lib.obs_names]
rna_lib = rna_lib[rna_lib.obs['mito_portion'] < 0.25]
print('Number of cells after mito filter: {:d}'.format(rna_lib.n_obs))

#remove cells with too low or high read counts_________________________________________________________________________________________________________________________________________________
sc.pp.filter_cells(rna_lib, min_counts = 1000)
print('Number of cells after min count filter: {:d}'.format(rna_lib.n_obs))
sc.pp.filter_cells(rna_lib, max_counts = 50000)
print('Number of cells after max count filter: {:d}'.format(rna_lib.n_obs))

#remove genes with too few cells_______________________________________________________________________________________________________________________________________________________________
print('Total number of genes: {:d}'.format(rna_lib.n_vars))
sc.pp.filter_genes(rna_lib, min_cells=1)
print('Number of genes after cell filter: {:d}'.format(rna_lib.n_vars))

corna_f = rna_lib.copy()
coatac_lib = coatac_lib[corna_f.obs_names]


# In[ ]:


atac_lib = coatac_lib
rna_lib = corna_f
pert_lib = copert_lib

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

atac_lib.write('coatac_filtered.h5ad')
rna_lib.write('corna_filtered.h5ad')
with open('copert_filtered.pkl', 'wb') as pert_file:
    pkl.dump(pert_lib,pert_file)


# In[ ]:


#load matrix (expects adata object)
atac_lib = sc.read_h5ad(FILENAME)

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

#keep cells within 2 standard deviations of mean
print('Total number of cells: {:d}'.format(atac_lib.n_obs),'\nmin threshold=',10**(mu-2*stdev),' max threshold=',10**(mu+2*stdev))
sc.pp.filter_cells(atac_lib, min_counts = 10**(mu-2*stdev))
print('Number of cells after min count filter: {:d}'.format(atac_lib.n_obs))
sc.pp.filter_cells(atac_lib, max_counts = 10**(mu+2*stdev))
print('Number of cells after max count filter: {:d}'.format(atac_lib.n_obs))

#save filtered matrix
atac_lib.write('FILENAME_filtered.h5ad')


# # OLD CODE NOT USED FOR V1 DATA

# In[ ]:


c=0
for elem in np.nditer(bla):
    if elem != 0:
        elem = 1


# In[ ]:


bla[0:100,0:100]


# In[ ]:


bla[0:100,0:100]


# In[ ]:


nz_inds = np.nonzero(bla)
bla[nz_inds] = 1


# # RNA LIBRARY

# In[ ]:


rna_lib.write('rna_filtered.h5ad')
pc_lib.write('pc_filtered.h5ad')
atac_lib.write('atac_filtered.h5ad')
with open('pert_filtered.pkl', 'wb') as pert_file:
    pkl.dump(pert_lib,pert_file)


# ## ATAC FILTERING

# In[ ]:


rna_lib = sc.read('rna_filtered.h5ad')
pc_lib = sc.read('pc_filtered.h5ad')
atac_lib = sc.read('atac_filtered.h5ad')
pert_lib = pkl.load(open('pert_filtered.pkl','rb'))


# In[ ]:


#count reads per cell and transform before counting cells per read
peaks_per_cell = [csr.csr_matrix.getnnz(atac_lib.X[cell,:]) for cell in range(len(atac_lib.obs_names))]
atac = csr.csr_matrix.tocsc(atac_lib.X)
cells_per_peak = [csr.csc_matrix.getnnz(atac[:,peak]) for peak in range(len(atac_lib.var_names))]
#log transform both
logcl = [np.log10(cell) for cell in cellslist]
logpl = [np.log10(peak) for peak in peakslist]


# In[ ]:


sb.violinplot(logcl)


# In[ ]:


sb.violinplot(logpl)


# In[ ]:


sparsec = [cell for cell in logcl if cell <= 2]
abundantc = [cell for cell in logcl if cell not in sparsec]
glc = [cell for cell in abundantc if cell <=4.5]
print(len(sparsec))
print(len(abundantc))
print(len(glc))

sparsep = [peak for peak in logpl if peak <= 1.25]
abundantp = [peak for peak in logpl if peak not in sparsep]
glp = [peak for peak in abundantp if peak <=3.25]
print(len(sparsep))
print(len(abundantp))
print(len(glp))


# In[ ]:


sb.violinplot(rna_lib.obs['n_counts'])


# In[ ]:


sb.violinplot(glc)


# In[ ]:


sb.violinplot(glp)


# In[ ]:


#filter matrices, keeping same cutoffs as used above
keep_cells = [cellid for cellid in range(len(logcl)) if logcl[cellid] >=2 and logcl[cellid] <= 4.5]
obsl = [atac_lib.obs_names[idx] for idx in keep_cells]
keep_peaks = [peakid for peakid in range(len(logpl)) if logpl[peakid] >= 1.25 and logpl[peakid] <= 3.25]
varl = [atac_lib.var_names[idx] for idx in keep_peaks]

#update all libs, but no need for pc_lib
new_atac_lib = atac_lib[obsl,varl]
new_rna_lib = rna_lib[obsl]
new_pert_lib = pert_lib.loc[obsl]


# In[ ]:


#write to files
new_rna_lib.write('rna_ff.h5ad')
pc_lib.write('pc_ff.h5ad')                      #note: pc_filtered and pc_ff are identical
new_atac_lib.write('atac_ff.h5ad')
with open('pert_ff.pkl', 'wb') as pert_file:
    pkl.dump(new_pert_lib,pert_file)


# In[ ]:


#load in fully filtered (ff) files
rna_lib = sc.read('rna_ff.h5ad')
pc_lib = sc.read('pc_ff.h5ad')
atac_lib = sc.read('atac_ff.h5ad')
pert_lib = pkl.load(open('pert_ff.pkl','rb'))


# ## OPTIONAL DOWNSAMPLING

# In[ ]:


get_ipython().run_cell_magic('time', '', 'a_counts = [np.sum(atac_lib[cell,:].X) for cell in range(len(atac_lib.obs_names))]')


# In[ ]:


sb.scatterplot(x=rna_lib.obs['n_counts'],y=a_counts)


# In[ ]:


plt.hexbin(x=rna_lib.obs['n_counts'],y=a_counts,bins='log')


# In[ ]:


print(np.std(rna_lib.obs['n_counts']))
print(np.median(rna_lib.obs['n_counts']))


# In[ ]:


print(np.std(a_counts))
print(np.median(a_counts))


# In[ ]:


sb.violinplot(rna_lib.obs['n_counts'])


# In[ ]:


newlib = sc.pp.downsample_counts(rna_lib,counts_per_cell=10000,copy=True)

n_list = []
for cell in newlib:
    n_list.append(np.sum(cell.X))

newlib.obs['n_counts'] = n_list


# In[ ]:


sb.violinplot(newlib.obs['n_counts'])


# In[ ]:


#write newlib
newlib.write('rna_ffds10k.h5ad')

