
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
import itertools as itt

# set a working directory for saving plots
os.chdir('/project/GCRB/Hon_lab/s437603/data/ghmt_multiome/analysis')


# In[20]:


#load in filtered files
blrna_lib = sc.read('blrna_filtered.h5ad')
blatac_lib = sc.read('blatac_filtered.h5ad')
blpert_lib = pkl.load(open('blpert_filtered.pkl','rb'))

lnrna_lib = sc.read('lnrna_filtered.h5ad')
lnatac_lib = sc.read('lnatac_filtered.h5ad')
lnpert_lib = pkl.load(open('lnpert_filtered.pkl','rb'))


# In[2]:


corna_lib = sc.read('corna_filtered.h5ad')
coatac_lib = sc.read('coatac_filtered.h5ad')
copert_lib = pkl.load(open('copert_filtered.pkl','rb'))


# ## ATAC LIBRARY MODIFICATION

# In[21]:


#max([max(atac_lib.var['start']),max(atac_lib.var['end'])]) = 195,242,156   highest chromosome position is 100M place so chromosome
#number can be added as the billions place multiplied by the chromosome (and 20, 21 for X, Y)
#1B modus pos can be used to get original positions from modified positions

#specify library
atac_lib = blatac_lib
rna_lib = blrna_lib
pert_lib = blpert_lib

#create new atac lib for modifying peak positions
chratac_lib = atac_lib.copy()

#remove peaks not mapped to a chromosome
drop_list = []
for this_peak in range(len(chratac_lib.var)):
    if chratac_lib.var['chr'][this_peak][0:3] != 'chr':
        drop_list.append(chratac_lib.var_names[this_peak])
keep_list = [this_peak for this_peak in chratac_lib.var_names if this_peak not in drop_list]
chratac_lib = chratac_lib[:,keep_list]

#order chromosomes correctly
chr_list = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16',
            'chr17','chr18','chr19','chrX','chrY']
annlist = []
for which_chr in chr_list:
    this_chrlib = chratac_lib[:,chratac_lib.var['chr'] == which_chr]
    annlist.append(this_chrlib.copy())
chratac_lib = sc.concat(annlist,axis=1)

#convert chromosome to integer and add to billions place
for this_peak in range(len(chratac_lib.var)):
    chrom = chratac_lib.var['chr'][this_peak]
    chrom_str = chrom.split('r')[1]
    if chrom_str == 'X':
        chr_num_int = 20
    elif chrom_str == 'Y':
        chr_num_int = 21
    else:
        chr_num_int = int(chrom_str)
    chratac_lib.var['start'][this_peak] += chr_num_int*1000000000
    chratac_lib.var['end'][this_peak] += chr_num_int*1000000000

#save final ce_atac_lib and tsv
chratac_lib.write('blce_atac.h5ad')

#TSS TSV AND RNA LIB MODIFICATION__________________________________________________________________________________________________________
#load in gtf
gene_tsv = pd.read_csv('/project/GCRB/Hon_lab/s437603/data/references/mm10_cra/genes/genes.gtf')
gene_tsv = gene_tsv[4:]

#keep only exon 1 positions
tss_df = pd.DataFrame(columns = ['chr','pos','gid'])
for entry in gene_tsv.iloc[:,0]:
    entry_list = entry.split('\t')
    if entry_list[2] == 'exon':
        if 'exon_number 1' in entry_list[8]:
            chromosome,start,end,strand = entry_list[0],entry_list[3],entry_list[4],entry_list[6]
            gene_id = entry_list[8].split(';')[5][12:-1]
            if strand == '+':
                position = start
            else:
                position = end
            
            tss_info = [chromosome,int(position),gene_id]
            tss_df.loc[len(tss_df)] = tss_info

#only keep reads mapped to chromosomes
keep_list = [row_title for row_title in tss_df.index if tss_df.loc[row_title,'chr'] in chr_list]
clean_tsv = tss_df.loc[keep_list,:]

#convert chromosome to integer and add to billions place
for tss in clean_tsv.index:
    chrom = clean_tsv.loc[tss,'chr']
    chrom_str = chrom.split('r')[1]
    if chrom_str == 'X':
        chr_num_int = 20
    elif chrom_str == 'Y':
        chr_num_int = 21
    else:
        chr_num_int = int(chrom_str)
    clean_tsv.loc[tss,'pos'] = int(clean_tsv.loc[tss,'pos']) + chr_num_int*1000000000
    
#drops duplicated TSSs
final_tsv = clean_tsv.drop_duplicates('pos')
    
#only keep genes that appear in both rna_lib and fintsv
gl = list(final_tsv.loc[:,'gid'])
glol = [g for g in gl if g in rna_lib.var_names]
rlol = [g for g in rna_lib.var_names if g in gl]
ce_rna = rna_lib[:,rlol]
keep_list = [row_title for row_title in final_tsv.index if final_tsv.loc[row_title,'gid'] in glol]
ce_tss = final_tsv.loc[keep_list,:]

#sort ce_tss
poslist = list(ce_tss.loc[:,'pos'])
poslist.sort()
indexlist = [ce_tss[ce_tss.loc[:,'pos'] == which_pos].index[0] for which_pos in poslist]
ce_tss = ce_tss.reindex(index = indexlist)

#save final ec_rna_lib and tss tsv
ce_rna.write('blce_rna.h5ad')
with open('blce_tss.pkl', 'wb') as ec_tsv_file:
    pkl.dump(ce_tss,ec_tsv_file)


# In[ ]:


#max([max(atac_lib.var['start']),max(atac_lib.var['end'])]) = 195,242,164   highest chromosome position is 100M place so chromosome
#number can be added as the billions place multiplied by the chromosome (and 20, 21 for X, Y)
#1B modus pos can be used to get original positions from modified positions

#specify library
atac_lib = lnatac_lib
rna_lib = lnrna_lib
pert_lib = lnpert_lib

#create new atac lib for modifying peak positions
chratac_lib = atac_lib.copy()

#remove peaks not mapped to a chromosome
drop_list = []
for this_peak in range(len(chratac_lib.var)):
    if chratac_lib.var['chr'][this_peak][0:3] != 'chr':
        drop_list.append(chratac_lib.var_names[this_peak])
keep_list = [this_peak for this_peak in chratac_lib.var_names if this_peak not in drop_list]
chratac_lib = chratac_lib[:,keep_list]

#order chromosomes correctly
chr_list = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16',
            'chr17','chr18','chr19','chrX','chrY']
annlist = []
for which_chr in chr_list:
    this_chrlib = chratac_lib[:,chratac_lib.var['chr'] == which_chr]
    annlist.append(this_chrlib.copy())
chratac_lib = sc.concat(annlist,axis=1)

#convert chromosome to integer and add to billions place
for this_peak in range(len(chratac_lib.var)):
    chrom = chratac_lib.var['chr'][this_peak]
    chrom_str = chrom.split('r')[1]
    if chrom_str == 'X':
        chr_num_int = 20
    elif chrom_str == 'Y':
        chr_num_int = 21
    else:
        chr_num_int = int(chrom_str)
    chratac_lib.var['start'][this_peak] += chr_num_int*1000000000
    chratac_lib.var['end'][this_peak] += chr_num_int*1000000000

#save final ce_atac_lib and tsv
chratac_lib.write('lnce_atac.h5ad')

#TSS TSV AND RNA LIB MODIFICATION__________________________________________________________________________________________________________
#load in gtf
gene_tsv = pd.read_csv('/project/GCRB/Hon_lab/s437603/data/references/mm10_cra/genes/genes.gtf')
gene_tsv = gene_tsv[4:]

#keep only exon 1 positions
tss_df = pd.DataFrame(columns = ['chr','pos','gid'])
for entry in gene_tsv.iloc[:,0]:
    entry_list = entry.split('\t')
    if entry_list[2] == 'exon':
        if 'exon_number 1' in entry_list[8]:
            chromosome,start,end,strand = entry_list[0],entry_list[3],entry_list[4],entry_list[6]
            gene_id = entry_list[8].split(';')[5][12:-1]
            if strand == '+':
                position = start
            else:
                position = end
            
            tss_info = [chromosome,int(position),gene_id]
            tss_df.loc[len(tss_df)] = tss_info

#only keep reads mapped to chromosomes
keep_list = [row_title for row_title in tss_df.index if tss_df.loc[row_title,'chr'] in chr_list]
clean_tsv = tss_df.loc[keep_list,:]

#convert chromosome to integer and add to billions place
for tss in clean_tsv.index:
    chrom = clean_tsv.loc[tss,'chr']
    chrom_str = chrom.split('r')[1]
    if chrom_str == 'X':
        chr_num_int = 20
    elif chrom_str == 'Y':
        chr_num_int = 21
    else:
        chr_num_int = int(chrom_str)
    clean_tsv.loc[tss,'pos'] = int(clean_tsv.loc[tss,'pos']) + chr_num_int*1000000000
    
#drops duplicated TSSs
final_tsv = clean_tsv.drop_duplicates('pos')
    
#only keep genes that appear in both rna_lib and fintsv
gl = list(final_tsv.loc[:,'gid'])
glol = [g for g in gl if g in rna_lib.var_names]
rlol = [g for g in rna_lib.var_names if g in gl]
ce_rna = rna_lib[:,rlol]
keep_list = [row_title for row_title in final_tsv.index if final_tsv.loc[row_title,'gid'] in glol]
ce_tss = final_tsv.loc[keep_list,:]

#sort ce_tss
poslist = list(ce_tss.loc[:,'pos'])
poslist.sort()
indexlist = [ce_tss[ce_tss.loc[:,'pos'] == which_pos].index[0] for which_pos in poslist]
ce_tss = ce_tss.reindex(index = indexlist)

#save final ec_rna_lib and tss tsv
ce_rna.write('lnce_rna.h5ad')
with open('lnce_tss.pkl', 'wb') as ec_tsv_file:
    pkl.dump(ce_tss,ec_tsv_file)


# In[3]:


#max([max(atac_lib.var['start']),max(atac_lib.var['end'])]) = 195,242,156   highest chromosome position is 100M place so chromosome
#number can be added as the billions place multiplied by the chromosome (and 20, 21 for X, Y)
#1B modus pos can be used to get original positions from modified positions

#specify library
atac_lib = coatac_lib
rna_lib = corna_lib
pert_lib = copert_lib

#create new atac lib for modifying peak positions
chratac_lib = atac_lib.copy()

#remove peaks not mapped to a chromosome
drop_list = []
for this_peak in range(len(chratac_lib.var)):
    if chratac_lib.var['chr'][this_peak][0:3] != 'chr':
        drop_list.append(chratac_lib.var_names[this_peak])
keep_list = [this_peak for this_peak in chratac_lib.var_names if this_peak not in drop_list]
chratac_lib = chratac_lib[:,keep_list]

#order chromosomes correctly
chr_list = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16',
            'chr17','chr18','chr19','chrX','chrY']
annlist = []
for which_chr in chr_list:
    this_chrlib = chratac_lib[:,chratac_lib.var['chr'] == which_chr]
    annlist.append(this_chrlib.copy())
chratac_lib = sc.concat(annlist,axis=1)

#convert chromosome to integer and add to billions place
for this_peak in range(len(chratac_lib.var)):
    chrom = chratac_lib.var['chr'][this_peak]
    chrom_str = chrom.split('r')[1]
    if chrom_str == 'X':
        chr_num_int = 20
    elif chrom_str == 'Y':
        chr_num_int = 21
    else:
        chr_num_int = int(chrom_str)
    chratac_lib.var['start'][this_peak] += chr_num_int*1000000000
    chratac_lib.var['end'][this_peak] += chr_num_int*1000000000

#save final ce_atac_lib and tsv
chratac_lib.write('coce_atac.h5ad')

#TSS TSV AND RNA LIB MODIFICATION__________________________________________________________________________________________________________
#load in gtf
gene_tsv = pd.read_csv('/project/GCRB/Hon_lab/s437603/data/references/mm10_cra/genes/genes.gtf')
gene_tsv = gene_tsv[4:]

#keep only exon 1 positions
tss_df = pd.DataFrame(columns = ['chr','pos','gid'])
for entry in gene_tsv.iloc[:,0]:
    entry_list = entry.split('\t')
    if entry_list[2] == 'exon':
        if 'exon_number 1' in entry_list[8]:
            chromosome,start,end,strand = entry_list[0],entry_list[3],entry_list[4],entry_list[6]
            gene_id = entry_list[8].split(';')[5][12:-1]
            if strand == '+':
                position = start
            else:
                position = end
            
            tss_info = [chromosome,int(position),gene_id]
            tss_df.loc[len(tss_df)] = tss_info

#only keep reads mapped to chromosomes
keep_list = [row_title for row_title in tss_df.index if tss_df.loc[row_title,'chr'] in chr_list]
clean_tsv = tss_df.loc[keep_list,:]

#convert chromosome to integer and add to billions place
for tss in clean_tsv.index:
    chrom = clean_tsv.loc[tss,'chr']
    chrom_str = chrom.split('r')[1]
    if chrom_str == 'X':
        chr_num_int = 20
    elif chrom_str == 'Y':
        chr_num_int = 21
    else:
        chr_num_int = int(chrom_str)
    clean_tsv.loc[tss,'pos'] = int(clean_tsv.loc[tss,'pos']) + chr_num_int*1000000000
    
#drops duplicated TSSs
final_tsv = clean_tsv.drop_duplicates('pos')
    
#only keep genes that appear in both rna_lib and fintsv
gl = list(final_tsv.loc[:,'gid'])
glol = [g for g in gl if g in rna_lib.var_names]
rlol = [g for g in rna_lib.var_names if g in gl]
ce_rna = rna_lib[:,rlol]
keep_list = [row_title for row_title in final_tsv.index if final_tsv.loc[row_title,'gid'] in glol]
ce_tss = final_tsv.loc[keep_list,:]

#sort ce_tss
poslist = list(ce_tss.loc[:,'pos'])
poslist.sort()
indexlist = [ce_tss[ce_tss.loc[:,'pos'] == which_pos].index[0] for which_pos in poslist]
ce_tss = ce_tss.reindex(index = indexlist)

#save final ec_rna_lib and tss tsv
ce_rna.write('coce_rna.h5ad')
with open('coce_tss.pkl', 'wb') as ec_tsv_file:
    pkl.dump(ce_tss,ec_tsv_file)


# ## SPLIT BY PERT

# In[3]:


def pert_lister(pert_lib):
#pert_lib = dataframe of perturbation matrix (cells=rows, TFs=columns)
#assumes final column in dataframe is the negative control
    #_________________________________________________________________________________________________________________________________
    #rename columns to be TF names and create separate list of TFs
    pert_lib.columns = [tf.partition('_')[0] for tf in pert_lib.columns]     #DEPENDS ON DATAFRAME FORMAT!
    cols = list(pert_lib.columns)
    #_________________________________________________________________________________________________________________________________
    #create lists for negative control cells
    neg_combo = [(cols[-1],)]
    neg_bool = (0,)*(len(cols)-1)+(1,)            #for when only mNG is present
    null_bool = (0,)*(len(cols)-1)+(0,)           #for when no constructs are present
    neg_cells = []
    for cell in range(len(pert_lib.index)):
        pert_bool_neg = pert_lib.iloc[cell,:] == neg_bool
        pert_bool_null = pert_lib.iloc[cell,:] == null_bool
        if (pert_bool_neg.all() or pert_bool_null.all()):
            neg_cells.append(pert_lib.index[cell])
    neg_count = [len(neg_cells)]
    #_________________________________________________________________________________________________________________________________
    #create list of possible perturbations as TF tuple
    pert_combos = []
    for dimen in range(1,(len(cols))):
        dimen_combos = list(combinations(cols[:-1],dimen))
        for combo in dimen_combos:
            pert_combos.append(combo)
    #_________________________________________________________________________________________________________________________________
    #initialize and populate list of cells with each perturbation
    #index in pert_cells matches index in pert_combos
    pert_cells = [[] for combo in pert_combos]
    for cell in range(len(pert_lib.index)):
        pert_tup = ()
        for tf in range(len(pert_lib.iloc[cell,:-1])):
            #tf_mult = [1,4,1,5]                                                         #read count condition
            if pert_lib.iloc[cell,tf] >= 1:#*tf_mult[tf]:                                 #read count condition
                pert_tup+=(pert_lib.columns[tf],)
        for combo_ind in range(len(pert_combos)):
            if pert_tup == pert_combos[combo_ind]:
                pert_cells[combo_ind].append(pert_lib.index[cell])            
    #intialize and populate list of number of cells with each perturbation
    #index in pert_counts matches index in pert_combos
    pert_counts = [len(pert_cells[combo]) for combo in range(len(pert_cells))]
    #_________________________________________________________________________________________________________________________________
    #final output preparation
    pert_combos = neg_combo + pert_combos
    for combo in range(len(pert_combos)):
        temp_str = ''
        for tf_tup in pert_combos[combo]:
            temp_str += str(tf_tup)+','
        temp_str = temp_str[0:-1]
        pert_combos[combo] = temp_str
    pert_cells = [neg_cells] + pert_cells
    pert_counts = neg_count + pert_counts
    #_________________________________________________________________________________________________________________________________
    return pert_combos,pert_cells,pert_counts


# In[ ]:


# call for pert_lister()
pert_lib = blpert_lib
com,cel,cou = pert_lister(pert_lib)
ccc = [com,cel,cou]
with open('blccc.pkl', 'wb') as ccc_file:
    pkl.dump(ccc,ccc_file)
    
pert_lib = lnpert_lib
com,cel,cou = pert_lister(pert_lib)
ccc = [com,cel,cou]
with open('lnccc.pkl', 'wb') as ccc_file:
    pkl.dump(ccc,ccc_file)


# In[5]:


pert_lib = copert_lib
com,cel,cou = pert_lister(pert_lib)
ccc = [com,cel,cou]
with open('coccc.pkl', 'wb') as ccc_file:
    pkl.dump(ccc,ccc_file)


# In[6]:


len(copert_lib),len(corna_lib),len(coatac_lib)


# In[20]:


l = list(copert_lib.index)
for item in list(set(list(copert_lib.index))):
    l.remove(item)


# In[36]:


len(set(l))


# In[24]:


ccc = pkl.load(open('coccc.pkl', 'rb'))


# In[40]:


flatccc = [cell for group in ccc[1] for cell in group]
len(flatccc)


# In[41]:


inrna = list(corna_lib.obs_names)
print(len(inrna))
count = 0
for cid in inrna:
    if cid not in flatccc:
        count+=1
count


# In[27]:


np.sum(ccc[2])


# In[35]:


dupid = []
for dup in l:
    for group in ccc[1]:
        if dup in group:
            dupid.append(ccc[1].index(group))
sb.histplot(dupid,bins=16)


# ## GENERATE NEW CCC BASED ON PERTURBATION DEGREE

# In[ ]:


ccc = pkl.load(open('blccc.pkl','rb'))
com,cel,cou = ccc

pdccc = []
pdcom,pdcel,pdcou = [],[],[]
#NC
pdcom.append('Negatives')
pdcel.append(ccc[1][0])
pdcou.append(ccc[2][0])
#Singles
pdcom.append('Singles')
pdcel.append(ccc[1][1]+ccc[1][2]+ccc[1][3]+ccc[1][4])
pdcou.append(ccc[2][1]+ccc[2][2]+ccc[2][3]+ccc[2][4])
#Doubles
pdcom.append('Doubles')
pdcel.append(ccc[1][5]+ccc[1][6]+ccc[1][7]+ccc[1][8]+ccc[1][9]+ccc[1][10])
pdcou.append(ccc[2][5]+ccc[2][6]+ccc[2][7]+ccc[2][8]+ccc[2][9]+ccc[2][10])
#Triples
pdcom.append('Triples')
pdcel.append(ccc[1][11]+ccc[1][12]+ccc[1][13]+ccc[1][14])
pdcou.append(ccc[2][11]+ccc[2][12]+ccc[2][13]+ccc[2][14])
#Quadruples
pdcom.append('Quadruples')
pdcel.append(ccc[1][15])
pdcou.append(ccc[2][15])

pdccc = [pdcom,pdcel,pdcou]

with open('blpdccc.pkl','wb') as file:
    pkl.dump(pdccc,file)


# In[ ]:


ccc = pkl.load(open('lnccc.pkl','rb'))
com,cel,cou = ccc

pdccc = []
pdcom,pdcel,pdcou = [],[],[]
#NC
pdcom.append('Negatives')
pdcel.append(ccc[1][0])
pdcou.append(ccc[2][0])
#Singles
pdcom.append('Singles')
pdcel.append(ccc[1][1]+ccc[1][2]+ccc[1][3]+ccc[1][4])
pdcou.append(ccc[2][1]+ccc[2][2]+ccc[2][3]+ccc[2][4])
#Doubles
pdcom.append('Doubles')
pdcel.append(ccc[1][5]+ccc[1][6]+ccc[1][7]+ccc[1][8]+ccc[1][9]+ccc[1][10])
pdcou.append(ccc[2][5]+ccc[2][6]+ccc[2][7]+ccc[2][8]+ccc[2][9]+ccc[2][10])
#Triples
pdcom.append('Triples')
pdcel.append(ccc[1][11]+ccc[1][12]+ccc[1][13]+ccc[1][14])
pdcou.append(ccc[2][11]+ccc[2][12]+ccc[2][13]+ccc[2][14])
#Quadruples
pdcom.append('Quadruples')
pdcel.append(ccc[1][15])
pdcou.append(ccc[2][15])

pdccc = [pdcom,pdcel,pdcou]

with open('lnpdccc.pkl','wb') as file:
    pkl.dump(pdccc,file)


# In[6]:


ccc = pkl.load(open('coccc.pkl','rb'))
com,cel,cou = ccc

pdccc = []
pdcom,pdcel,pdcou = [],[],[]
#NC
pdcom.append('Negatives')
pdcel.append(ccc[1][0])
pdcou.append(ccc[2][0])
#Singles
pdcom.append('Singles')
pdcel.append(ccc[1][1]+ccc[1][2]+ccc[1][3]+ccc[1][4])
pdcou.append(ccc[2][1]+ccc[2][2]+ccc[2][3]+ccc[2][4])
#Doubles
pdcom.append('Doubles')
pdcel.append(ccc[1][5]+ccc[1][6]+ccc[1][7]+ccc[1][8]+ccc[1][9]+ccc[1][10])
pdcou.append(ccc[2][5]+ccc[2][6]+ccc[2][7]+ccc[2][8]+ccc[2][9]+ccc[2][10])
#Triples
pdcom.append('Triples')
pdcel.append(ccc[1][11]+ccc[1][12]+ccc[1][13]+ccc[1][14])
pdcou.append(ccc[2][11]+ccc[2][12]+ccc[2][13]+ccc[2][14])
#Quadruples
pdcom.append('Quadruples')
pdcel.append(ccc[1][15])
pdcou.append(ccc[2][15])

pdccc = [pdcom,pdcel,pdcou]

with open('copdccc.pkl','wb') as file:
    pkl.dump(pdccc,file)


# ## CREATE NCE_ATAC AND NCE_RNA

# In[ ]:


#load in *ce_* files
blce_rna = sc.read('blce_rna.h5ad')
blce_atac = sc.read('blce_atac.h5ad')

lnce_rna = sc.read('lnce_rna.h5ad')
lnce_atac = sc.read('lnce_atac.h5ad')


# In[2]:


coce_rna = sc.read('coce_rna.h5ad')
coce_atac = sc.read('coce_atac.h5ad')


# In[ ]:


#CPM normalzie ce_rna and ce_atac to generate nce_rna and nce_atac 
sc.pp.normalize_total(blce_rna, key_added='n_counts',target_sum=1e6)
sc.pp.normalize_total(blce_atac, key_added='n_counts',target_sum=1e6)

sc.pp.normalize_total(lnce_rna, key_added='n_counts',target_sum=1e6)
sc.pp.normalize_total(lnce_atac, key_added='n_counts',target_sum=1e6)


# In[3]:


sc.pp.normalize_total(coce_rna, key_added='n_counts',target_sum=1e6)
sc.pp.normalize_total(coce_atac, key_added='n_counts',target_sum=1e6)


# In[ ]:


#save *nce_* files
with open('blnce_rna.pkl', 'wb') as cra_file:
    pkl.dump(blce_rna,cra_file)
with open('blnce_atac.pkl', 'wb') as caa_file:
    pkl.dump(blce_atac,caa_file)

with open('lnnce_rna.pkl', 'wb') as cra_file:
    pkl.dump(lnce_rna,cra_file)
with open('lnnce_atac.pkl', 'wb') as caa_file:
    pkl.dump(lnce_atac,caa_file)


# In[4]:


with open('conce_rna.pkl', 'wb') as cra_file:
    pkl.dump(coce_rna,cra_file)
with open('conce_atac.pkl', 'wb') as caa_file:
    pkl.dump(coce_atac,caa_file)


# ## CREATE ZCE_RNA

# In[ ]:


#load in *nce_* files
blce_rna = pkl.load(open('blnce_rna.pkl','rb'))
lnce_rna = pkl.load(open('lnnce_rna.pkl','rb'))


# In[5]:


coce_rna = pkl.load(open('conce_rna.pkl','rb'))


# In[ ]:


#transform
zce_rna = blce_rna.copy()
sc.pp.log1p(zce_rna)
zce_rna.raw = zce_rna
sc.pp.scale(zce_rna)
#save
with open('blzce_rna.pkl', 'wb') as zra_file:
    pkl.dump(zce_rna,zra_file)


# In[ ]:


#transform
zce_rna = lnce_rna.copy()
sc.pp.log1p(zce_rna)
zce_rna.raw = zce_rna
sc.pp.scale(zce_rna)
#save
with open('lnzce_rna.pkl', 'wb') as zra_file:
    pkl.dump(zce_rna,zra_file)


# In[6]:


#transform
zce_rna = coce_rna.copy()
sc.pp.log1p(zce_rna)
zce_rna.raw = zce_rna
sc.pp.scale(zce_rna)
#save
with open('cozce_rna.pkl', 'wb') as zra_file:
    pkl.dump(zce_rna,zra_file)


# ## CREATE PROMOL

# In[ ]:


#load in ce_libraries
blcerna_lib = sc.read('blce_rna.h5ad')
blceatac_lib = sc.read('blce_atac.h5ad')
bl_tss = pkl.load(open('blce_tss.pkl','rb'))

lncerna_lib = sc.read('lnce_rna.h5ad')
lnceatac_lib = sc.read('lnce_atac.h5ad')
ln_tss = pkl.load(open('lnce_tss.pkl','rb'))


# In[7]:


cocerna_lib = sc.read('coce_rna.h5ad')
coceatac_lib = sc.read('coce_atac.h5ad')
co_tss = pkl.load(open('coce_tss.pkl','rb'))


# In[ ]:


#create list of index triples [tss_ind,rna_ind,atac_ind] for all available tss
atac = blceatac_lib.copy()
tss = bl_tss.copy()
ce_rna = blcerna_lib.copy()
prom_overlaps = []
offset = 1000

pos_col,gid_col = 1,2
this_peak,this_tss = 0,0 #initialize atac and tss index to beginning

while (this_tss < len(tss)) and (this_peak < len(atac.var_names)):
    this_pos = tss.iloc[this_tss,pos_col]
    if atac.var['start'][this_peak] > this_pos+offset:
        this_tss += 1
    elif atac.var['end'][this_peak] < this_pos-offset:
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

with open('bl1000ce_prom_overlaps.pkl', 'wb') as cpo_file:
    pkl.dump(prom_overlaps,cpo_file)


# In[ ]:


#create list of index triples [tss_ind,rna_ind,atac_ind] for all available tss
atac = lnceatac_lib.copy()
tss = ln_tss.copy()
ce_rna = lncerna_lib.copy()
prom_overlaps = []
offset = 1000

pos_col,gid_col = 1,2
this_peak,this_tss = 0,0 #initialize atac and tss index to beginning

while (this_tss < len(tss)) and (this_peak < len(atac.var_names)):
    this_pos = tss.iloc[this_tss,pos_col]
    if atac.var['start'][this_peak] > this_pos+offset:
        this_tss += 1
    elif atac.var['end'][this_peak] < this_pos-offset:
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

with open('ln1000ce_prom_overlaps.pkl', 'wb') as cpo_file:
    pkl.dump(prom_overlaps,cpo_file)


# In[8]:


#create list of index triples [tss_ind,rna_ind,atac_ind] for all available tss
atac = coceatac_lib.copy()
tss = co_tss.copy()
ce_rna = cocerna_lib.copy()
prom_overlaps = []
offset = 1000

pos_col,gid_col = 1,2
this_peak,this_tss = 0,0 #initialize atac and tss index to beginning

while (this_tss < len(tss)) and (this_peak < len(atac.var_names)):
    this_pos = tss.iloc[this_tss,pos_col]
    if atac.var['start'][this_peak] > this_pos+offset:
        this_tss += 1
    elif atac.var['end'][this_peak] < this_pos-offset:
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

with open('co1000ce_prom_overlaps.pkl', 'wb') as cpo_file:
    pkl.dump(prom_overlaps,cpo_file)


# ## CREATE PROMENHOL AND ENHOL

# In[2]:


#load in ce_libraries
blcerna_lib = sc.read('blce_rna.h5ad')
blceatac_lib = sc.read('blce_atac.h5ad')
bl_tss = pkl.load(open('blce_tss.pkl','rb'))
blpromol = pkl.load(open('bl1000ce_prom_overlaps.pkl','rb'))

lncerna_lib = sc.read('lnce_rna.h5ad')
lnceatac_lib = sc.read('lnce_atac.h5ad')
ln_tss = pkl.load(open('lnce_tss.pkl','rb'))
lnpromol = pkl.load(open('ln1000ce_prom_overlaps.pkl','rb'))


# In[9]:


cocerna_lib = sc.read('coce_rna.h5ad')
coceatac_lib = sc.read('coce_atac.h5ad')
co_tss = pkl.load(open('coce_tss.pkl','rb'))
copromol = pkl.load(open('co1000ce_prom_overlaps.pkl','rb'))


# In[5]:


#create list of index triples [tss_ind,rna_ind,[atac_inds]] for all available tss, where [atac_inds] is a list of all atac indexes
#within the enhancer window of a given tss
atac = blceatac_lib.copy()
tss = bl_tss.copy()
ce_rna = blcerna_lib
promol = blpromol

promenh_overlaps = []

pos_col,gid_col = 1,2
this_peak,this_tss = 0,0                      #initialize atac and tss index to beginning
window = 10**5                                #scan for enhancers within (position +/- window)
atac_index_list = []                          #initialize empty enh list for the while loop
jump_back = 0                                 #initialize tracker to rescan peaks when switching to a new tss
offset = 1000
while (this_tss < len(tss)) and (this_peak < len(atac.var_names)):                          #iterate through all tss
    this_pos = tss.iloc[this_tss,pos_col]
    if (atac.var['start'][this_peak] - window) > this_pos + offset:                         #if this peak is too far ahead of this tss
        #save any enhancers/proms for this tss
        if len(atac_index_list) > 0:                                                   
            tss_index = this_tss
            gid = tss.iloc[this_tss,gid_col]
            rna_index = list(ce_rna.var_names).index(tss.iloc[this_tss,gid_col])
            this_overlap = [tss_index,rna_index,atac_index_list]
            promenh_overlaps.append(this_overlap)
            
        #move to next tss and reset promenh list
        this_peak -= jump_back
        this_tss += 1
        atac_index_list = []
        jump_back = 0
    elif (atac.var['end'][this_peak] + window) < this_pos - offset:                        #if this tss is too far ahead of this peak
        this_peak += 1
    else:                                                                                  #if this tss and peak are in the same window
        atac_index = this_peak
        atac_index_list.append(atac_index)
        this_peak += 1
        jump_back += 1

#filter out entries with no associated promoter
peo = promenh_overlaps
tss_with_prom = [promol[which_tss][0] for which_tss in range(len(promol))]
f_peo = [peo[which_tss] for which_tss in range(len(peo)) if peo[which_tss][0] in tss_with_prom]
        
with open('bl1000ce_promenh_overlaps.pkl', 'wb') as cpeo_file:
    pkl.dump(f_peo,cpeo_file)

#remove proms from peak list so only enhancers remain
prom_col = 2
for which_tss in range(len(f_peo)):
    atac_ind = promol[which_tss][prom_col]
    f_peo[which_tss][prom_col].remove(atac_ind)
with open('bl1000ce_enh_overlaps.pkl', 'wb') as ceo_file:
    pkl.dump(f_peo,ceo_file)


# In[11]:


promenhol = pkl.load(open('ln1000ce_promenh_overlaps.pkl','rb'))
len(promenhol)


# In[4]:


#print(lnceatac_lib.var)
print(len(blpromol),len(bl_tss))
print(len(lnpromol),len(ln_tss))


# In[30]:


count = 0
missing = 0
for x in range(len(promenhol)):
    if promol[x][2] in promenhol[x][2]:
        count += 1
    if promol[x][0] == promenhol[x][0]:
        missing += 1
print(len(promenhol),len(promol),count,missing)


# In[10]:


#create list of index triples [tss_ind,rna_ind,[atac_inds]] for all available tss, where [atac_inds] is a list of all atac indexes
#within the enhancer window of a given tss
atac = lnceatac_lib.copy()
tss = ln_tss.copy()
ce_rna = lncerna_lib
promol = lnpromol

promenh_overlaps = []

pos_col,gid_col = 1,2
this_peak,this_tss = 0,0                      #initialize atac and tss index to beginning
window = 10**5                                #scan for enhancers within (position +/- window)
atac_index_list = []                          #initialize empty enh list for the while loop
jump_back = 0                                 #initialize tracker to rescan peaks when switching to a new tss
offset = 1000

while (this_tss < len(tss)) and (this_peak < len(atac.var_names)):                           #iterate through all tss
    this_pos = tss.iloc[this_tss,pos_col]
    if (atac.var['start'][this_peak] - window) > this_pos + offset:                         #if this peak is too far ahead of this tss
        #save any enhancers/proms for this tss
        if len(atac_index_list) > 0:                                                   
            tss_index = this_tss
            gid = tss.iloc[this_tss,gid_col]
            rna_index = list(ce_rna.var_names).index(tss.iloc[this_tss,gid_col])
            this_overlap = [tss_index,rna_index,atac_index_list]
            promenh_overlaps.append(this_overlap)
            
        #move to next tss and reset promenh list
        this_peak -= jump_back
        this_tss += 1
        atac_index_list = []
        jump_back = 0
    elif (atac.var['end'][this_peak] + window) < this_pos - offset:                         #if this tss is too far ahead of this peak
        this_peak += 1
    else:                                                                                   #if this tss and peak are in the same window
        atac_index = this_peak
        atac_index_list.append(atac_index)
        this_peak += 1
        jump_back += 1

#filter out entries with no associated promoter
peo = promenh_overlaps
tss_with_prom = [promol[which_tss][0] for which_tss in range(len(promol))]
f_peo = [peo[which_tss] for which_tss in range(len(peo)) if peo[which_tss][0] in tss_with_prom]
        
with open('ln1000ce_promenh_overlaps.pkl', 'wb') as cpeo_file:
    pkl.dump(f_peo,cpeo_file)

#remove proms from peak list so only enhancers remain
prom_col = 2
for which_tss in range(len(f_peo)):
    atac_ind = promol[which_tss][prom_col]
    f_peo[which_tss][prom_col].remove(atac_ind)
with open('ln1000ce_enh_overlaps.pkl', 'wb') as ceo_file:
    pkl.dump(f_peo,ceo_file)


# In[10]:


#create list of index triples [tss_ind,rna_ind,[atac_inds]] for all available tss, where [atac_inds] is a list of all atac indexes
#within the enhancer window of a given tss
atac = coceatac_lib.copy()
tss = co_tss.copy()
ce_rna = cocerna_lib
promol = copromol

promenh_overlaps = []

pos_col,gid_col = 1,2
this_peak,this_tss = 0,0                      #initialize atac and tss index to beginning
window = 10**5                                #scan for enhancers within (position +/- window)
atac_index_list = []                          #initialize empty enh list for the while loop
jump_back = 0                                 #initialize tracker to rescan peaks when switching to a new tss
offset = 1000

while (this_tss < len(tss)) and (this_peak < len(atac.var_names)):                           #iterate through all tss
    this_pos = tss.iloc[this_tss,pos_col]
    if (atac.var['start'][this_peak] - window) > this_pos + offset:                         #if this peak is too far ahead of this tss
        #save any enhancers/proms for this tss
        if len(atac_index_list) > 0:                                                   
            tss_index = this_tss
            gid = tss.iloc[this_tss,gid_col]
            rna_index = list(ce_rna.var_names).index(tss.iloc[this_tss,gid_col])
            this_overlap = [tss_index,rna_index,atac_index_list]
            promenh_overlaps.append(this_overlap)
            
        #move to next tss and reset promenh list
        this_peak -= jump_back
        this_tss += 1
        atac_index_list = []
        jump_back = 0
    elif (atac.var['end'][this_peak] + window) < this_pos - offset:                         #if this tss is too far ahead of this peak
        this_peak += 1
    else:                                                                                   #if this tss and peak are in the same window
        atac_index = this_peak
        atac_index_list.append(atac_index)
        this_peak += 1
        jump_back += 1

#filter out entries with no associated promoter
peo = promenh_overlaps
tss_with_prom = [promol[which_tss][0] for which_tss in range(len(promol))]
f_peo = [peo[which_tss] for which_tss in range(len(peo)) if peo[which_tss][0] in tss_with_prom]
        
with open('co1000ce_promenh_overlaps.pkl', 'wb') as cpeo_file:
    pkl.dump(f_peo,cpeo_file)

#remove proms from peak list so only enhancers remain
prom_col = 2
for which_tss in range(len(f_peo)):
    atac_ind = promol[which_tss][prom_col]
    f_peo[which_tss][prom_col].remove(atac_ind)
with open('co1000ce_enh_overlaps.pkl', 'wb') as ceo_file:
    pkl.dump(f_peo,ceo_file)


# ## LOAD IN PROM AND ENH OLs

# In[12]:


promenhol = pkl.load(open('bl1000ce_promenh_overlaps.pkl','rb'))     #all tss with enh AND prom peaks
promol = pkl.load(open('bl1000ce_prom_overlaps.pkl','rb'))           #all tss with with ONLY prom peaks
enhol = pkl.load(open('bl1000ce_enh_overlaps.pkl','rb'))             #all tss with ONLY enh peaks

print(len(promol),len(promenhol),len(enhol))

promenhol = pkl.load(open('ln1000ce_promenh_overlaps.pkl','rb'))     #all tss with enh AND prom peaks
promol = pkl.load(open('ln1000ce_prom_overlaps.pkl','rb'))           #all tss with with ONLY prom peaks
enhol = pkl.load(open('ln1000ce_enh_overlaps.pkl','rb'))             #all tss with ONLY enh peaks

print(len(promol),len(promenhol),len(enhol))


# In[13]:


promenhol = pkl.load(open('co1000ce_promenh_overlaps.pkl','rb'))     #all tss with enh AND prom peaks
promol = pkl.load(open('co1000ce_prom_overlaps.pkl','rb'))           #all tss with with ONLY prom peaks
enhol = pkl.load(open('co1000ce_enh_overlaps.pkl','rb'))             #all tss with ONLY enh peaks

print(len(promol),len(promenhol),len(enhol))


# ## MAIN

# In[12]:


ep = pkl.load(open('ce_enh_pvals.pkl','rb'))
rep = pkl.load(open('ce_rand_enh_pvals_1.pkl','rb'))


# In[16]:


pl = [elem[1] for elem in promol]
print(len(pl))
print(len(set(pl)))


# In[14]:


print(len(promol),len(promenhol),len(enhol))


# In[10]:


#load in new prom_overlaps and promenh_overlaps
promol = pkl.load(open('1000ce_prom_overlaps.pkl','rb'))


# In[11]:


promenhol = pkl.load(open('1000ce_promenh_overlaps.pkl','rb'))


# In[15]:


enhol = pkl.load(open('ln1000ce_enh_overlaps.pkl','rb'))


# # 1.1 Global Enhancer Scanning

# ### 1.4 MISC. Enhancer Stats

# In[14]:


#unique peaks average
elist = []
for which_tss in enhol:
    for which_peak in which_tss[2]:
        elist.append(which_peak)
print('TOTAL PEAKS =',len(elist))
siglist = set(elist)
print('UNIQUE PEAKS =',len(siglist))
print('TSS CONSIDERED =',len(enhol))


# In[10]:


#peaks passing bonferroni
#enhpvals = pkl.load(open('nce_enh_pvals.pkl','rb'))
enhpvals = pkl.load(open('nce_rand_enh_pvals_1.pkl','rb'))[0]
plist = []
for which_tss in enhpvals:
    for which_peak in which_tss:
        plist.append(which_peak)
print('TOTAL PEAKS =',len(plist))
siglist = [val for val in plist if val <= 0.05/len(plist)]
print('BONFERRONI-PASSING PEAKS =',len(siglist))
unad_siglist = [val for val in plist if val <= 0.05]
print('UNAD-PASSING PEAKS =',len(unad_siglist))


# In[7]:


#inverse log p values
ilist = [1/(value+0.0000000001) for value in plist]
llist = [np.log10(value) for value in ilist]
sb.violinplot(data=[llist],orient='v')


# In[11]:


#inverse log p values
ilist = [1/(value+0.0000000001) for value in plist]
llist2 = [np.log10(value) for value in ilist]
sb.violinplot(data = [llist,llist2])


# In[36]:


#average peaks detected per TSS
enh_nums = [len(which_tss[2]) for which_tss in enhol]

print('MEAN =',np.mean(enh_nums),'   MEDIAN =',np.median(enh_nums))
sb.violinplot(enh_nums)

