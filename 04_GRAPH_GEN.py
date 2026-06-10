
# coding: utf-8

import numpy as np
import scanpy as sc
import pandas as pd
import pickle as pkl
import seaborn as sb
import scipy.sparse as csr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.stats as ss
import math
import sys
import random
import networkx as nx

#start from tios files

#ORIGINAL FILTERED LIBS
rna_lib = sc.read('corna_filtered.h5ad')
atac_lib = sc.read('coatac_filtered.h5ad')
pert_lib = pkl.load(open('copert_filtered.pkl','rb'))
ccc = pkl.load(open('coccc.pkl','rb'))

# [tss_ind,rna_ind,[atac_inds]] for enhol
# [tss_ind,rna_ind,atac_ind] for promol
promol = pkl.load(open('co1000ce_prom_overlaps.pkl','rb'))
promenhol = pkl.load(open('co1000ce_promenh_overlaps.pkl','rb'))
enhol = pkl.load(open('co1000ce_enh_overlaps.pkl','rb'))

# LOAD IN CE LIBS
ce_rna = pkl.load(open('conce_rna.pkl','rb'))                   #ce_rna but CPM normalized
zce_rna = pkl.load(open('cozce_rna.pkl','rb'))                  #nce_rna but z-scored
ce_atac = pkl.load(open('conce_atac.pkl','rb'))                 #ce_atac but binarized and then CPM normalized
ce_tss = pkl.load(open('coce_tss.pkl','rb'))

# LOAD IN FULL STATS FILES,    each element is in form [pv,fc]
unflat_te = pkl.load(open('total_co_perturbationenhancer_stats.pkl','rb'))
te = [[stats for pert in enh for stats in pert] for enh in unflat_te] #flatten by one dimension
tp = pkl.load(open('total_co_perturbationpromoter_stats.pkl','rb'))
tr = pkl.load(open('total_co_perturbationrna_stats.pkl','rb'))
ep = pkl.load(open('total_co_enhancerpromoter_stats.pkl','rb'))
er = pkl.load(open('total_co_enhancerrna_stats.pkl','rb'))
pr = pkl.load(open('total_co_promoterrna_stats.pkl','rb'))

#create binary matrix from all perturbation combos__________________________________________________________________________________________________________________________________________________________
tf_combos = ccc[0].copy()
tf_combos[0] = ''
tf_te = torch.zeros((len(tf_combos),len(tf_combos[len(tf_combos)-1].split(','))))     #initializes tensor of (# of pert combos) by (# of component perts)       #            G   H   M   T
for combo_index,combo in enumerate(tf_combos):                                                                                                                  # combo 0    -   -   -   -
    #populate tf_te, assume GHMT ordering                                                                                                                       # combo 1    -   -   -   -
    component_perts = combo.split(',')                                                                                                                          # combo n    -   -   -   -
    if 'Gata4' in component_perts:
        tf_te[combo_index,0] = 1
    if 'Hand2' in component_perts:
        tf_te[combo_index,1] = 1
    if 'Mef2c' in component_perts:
        tf_te[combo_index,2] = 1
    if 'Tbx5' in component_perts:
        tf_te[combo_index,3] = 1
tf_tensors=[tf_te]*len(pr)
 
#save tf_te
with open('component_tfs_tensor.pkl', 'wb') as file:
    pkl.dump(tf_tensors,file)

#save mm10 genome fasta as tuple list______________________________________________________________________________________________________________________________
#load in reference genome from 10x
with open('genome.fa','r') as file: #use the mm10 genome.fa file from 10x Genomics
    file_lines = file.readlines()
chr_info,chr_l,chr_str = [],0,''
for line_ind,line in enumerate(file_lines):
    if line[0] == '>':
        if chr_l != 0:
            chr_info.append((chr_name,chr_l,chr_str))
        chr_name = line[1:].partition(' ')[0]
        chr_l = 0
        chr_str = ''
    chr_l += len(line)
    chr_str += line
chr_info
#clean out newlines and chr name characters from chr_str
real_chr_info = []
for which_chr in chr_info:
    real_chr_info.append((which_chr[0],which_chr[1],which_chr[2][7:].replace('\n','')))
#save file
with(open('mm10_genome_by_chr.pkl','wb')) as ref_file:
    pkl.dump(real_chr_info,ref_file)

#initialize and populate df and adata for 6mers____________________________________________________________________________________________________________________
rci = pkl.load(open('mm10_genome_by_chr.pkl','rb'))
all6mers = [('').join(elem) for elem in product('ATCG',repeat=6)]
df_6mer = pd.DataFrame(data=np.zeros(shape=(len(ce_atac.var_names),len(all6mers))),index=ce_atac.var_names,columns=all6mers)
for region_index,region in enumerate(df_6mer.index):                                                         #scan through each region in df
    this_chr = region.split(':')[0]
    this_start = int(region.split(':')[1].split('-')[0])
    this_end = int(region.split(':')[1].split('-')[1])
    chr_index = [which_index for which_index in range(len(rci)) if rci[which_index][0] == this_chr]
    reference_region = rci[chr_index[0]][2][this_start:this_end] 
    for index1,char1 in enumerate(reference_region[:-5]):                                                    #define each 6mer and populate
        this_6mer = reference_region[index1:index1+6]
        if 'N' in this_6mer:
            continue
        df_6mer.loc[region,this_6mer] += 1
adata_6mer = sc.AnnData(df_6mer)
sc.tl.pca(adata_6mer,n_comps=2000)
#save file
adata_6mer.write('6mer_adata_co.h5ad')

#create list of tensors for distance to promoter of e nodes followed by 6mer tiling__________________________________________________________________________________
adata_6mer = sc.read('6mer_adata_co.h5ad')
e_diseq_tensors = []
for tss_id,enh_info in enumerate(enhol):
    numof_e = len(enh_info[2])
    #initialize tensor (num_nodes x [prom.distance,6mer adata])
    this_tss_diseq = torch.zeros([numof_e,1+adata_6mer.n_vars])
    #get promol/enhol indices and calculate e-p distance in bp
    prom_pos = int((ce_atac.var.iloc[promol[tss_id][2],3]+ce_atac.var.iloc[promol[tss_id][2],4])/2)                                               #finds midpoint of promoter peak
    enh_pos_list = [int((ce_atac.var.iloc[enh_info[2][enhol_id],3]+ce_atac.var.iloc[enh_info[2][enhol_id],4])/2) for enhol_id in range(numof_e)]  #finds midpoint of enhancer peaks
    enh_dist_tensor = torch.Tensor([abs(enh_pos-prom_pos) for enh_pos in enh_pos_list])
    #add distances to 0th column of this_tss_diseq
    this_tss_diseq[:,0] = enh_dist_tensor
    #add 6mer to each row of this_tss_diseq, iteratively
    for which_e,row in enumerate(this_tss_diseq):
        row[1:] = torch.from_numpy(adata_6mer.X[enh_info[2][which_e]])
    e_diseq_tensors.append(this_tss_diseq)
    
#save file
with(open('e_diseq_tensors.pkl','wb')) as file:
    pkl.dump(e_diseq_tensors,file)

#create list of tensors for 6mer tiling of all p nodes_________________________________________________________________________________________________________________________
p_seq_tensors = []
for tss_id,prom_info in enumerate(promol):
    #initialize tensor (1 x 6mer adata)
    this_tss_seq = torch.from_numpy(adata_6mer.X[prom_info[2]]).reshape(1,-1)
    p_seq_tensors.append(this_tss_seq)
    
#save file
with(open('p_seq_tensors.pkl','wb')) as file:
    pkl.dump(p_seq_tensors,file)
    
#create list of placeholder tensors for RNA node (needed for node initialization but won't influence learning)____________________________________________________________________
r_ph_tensors = [torch.ones((1,1))]*len(pr)

#save file
with(open('r_ph_tensors.pkl','wb')) as file:
    pkl.dump(r_ph_tensors,file)

#if loading from files
t_nodes = pkl.load(open('component_tfs_tensor.pkl', 'rb'))
e_nodes = pkl.load(open('e_diseq_tensors.pkl', 'rb'))
p_nodes = pkl.load(open('p_seq_tensors.pkl', 'rb'))
r_nodes = pkl.load(open('r_ph_tensors.pkl', 'rb'))

#if running straight from previous block
# t_nodes = tf_tensors
# e_nodes = e_diseq_tensors
# p_nodes = p_seq_tensors
# r_nodes = r_ph_tensors

#make 1 graph per tss
glist = []
for tss_id in range(len(pr)):
    g=pyg.data.HeteroData()
    #add node features as defined previously
    g['perturbation'].x = t_nodes[tss_id]
    g['enhancer'].x = e_nodes[tss_id]
    g['promoter'].x = p_nodes[tss_id]
    g['rna'].x = r_nodes[tss_id]
    #add edge information
    numt = len(g['perturbation'].x)
    nume = len(g['enhancer'].x)
    #te
    te_index1 = [int(elem) for tlist in [[elem]*nume for elem in range(numt)] for elem in tlist] #excuse the mess, this line iterates AND flattens
    te_index2 = [int(elem) for elem in range(nume)]*numt
    te_index = torch.Tensor((te_index1,te_index2)).to(torch.int32)
    te_p = torch.Tensor([max(elem,10e-16) for whicht in [[elem[0] for elem in te[tss_id][whicht::numt]] for whicht in range(numt)] for elem in whicht]) #max() is to remove infs by capping
    te_p = -torch.log10(te_p)
    te_f = torch.Tensor([elem for whicht in [[elem[1] for elem in te[tss_id][whicht::numt]] for whicht in range(numt)] for elem in whicht])
    g['perturbation','pvalue','enhancer'].edge_index = te_index
    g['perturbation','pvalue','enhancer'].edge_attr = te_p
    g['perturbation','foldchange','enhancer'].edge_index = te_index
    g['perturbation','foldchange','enhancer'].edge_attr = te_f
    #tp
    tp_index = torch.Tensor((range(numt),[0]*numt)).to(torch.int32)
    tp_p = torch.Tensor([max(elem[0],10e-16) for elem in tp[0]])
    tp_p = -torch.log10(tp_p)
    tp_f = torch.Tensor([elem[1] for elem in tp[0]])
    g['perturbation','pvalue','promoter'].edge_index = tp_index
    g['perturbation','pvalue','promoter'].edge_attr = tp_p
    g['perturbation','foldchange','promoter'].edge_index = tp_index
    g['perturbation','foldchange','promoter'].edge_attr = tp_f
    #ep
    ep_index = torch.Tensor((range(nume),[0]*nume)).to(torch.int32)
    ep_p = torch.Tensor([max(elem[0],10e-16) for elem in ep[0]])
    ep_p = -torch.log10(ep_p)
    ep_f = torch.Tensor([elem[1] for elem in ep[0]]) 
    g['enhancer','pvalue','promoter'].edge_index = ep_index
    g['enhancer','pvalue','promoter'].edge_attr = ep_p
    g['enhancer','foldchange','promoter'].edge_index = ep_index
    g['enhancer','foldchange','promoter'].edge_attr = ep_f
    #er
    er_index = torch.Tensor((range(nume),[0]*nume)).to(torch.int32)
    er_p = torch.Tensor([max(elem[0],10e-16) for elem in er[0]])
    er_p = -torch.log10(er_p)
    er_f = torch.Tensor([elem[1] for elem in er[0]])
    g['enhancer','pvalue','rna'].edge_index = er_index
    g['enhancer','pvalue','rna'].edge_attr = er_p
    g['enhancer','foldchange','rna'].edge_index = er_index
    g['enhancer','foldchange','rna'].edge_attr = er_f
    #pr
    pr_index = torch.Tensor(([0],[0])).to(torch.int32)
    pr_p = torch.Tensor([max(elem[0],10e-16) for elem in pr[10]])
    pr_p = -torch.log10(pr_p)
    pr_f = torch.Tensor([elem[1] for elem in pr[0]])
    g['promoter','pvalue','rna'].edge_index = pr_index
    g['promoter','pvalue','rna'].edge_attr = pr_p
    g['promoter','foldchange','rna'].edge_index = pr_index
    g['promoter','foldchange','rna'].edge_attr = pr_f
    #transform to undirected graph (adds reverse edges)
    g = T.ToUndirected()(g)
    #add tensor of t-r response, cutoff designates ratio of fc to call neutral/no regulation,  25% expression difference in this case
    cutoff = 1.25
    g.y = torch.Tensor([float(elem[1]>(1/cutoff))+float(elem[1]>(cutoff)) for elem in tr[0]])
    glist.append(g)

#remove graphs with no associated enhancers
glist = [elem for elem in glist if elem['enhancer'].x.size()[0] > 0]

#save dataset as .pt
torch.save(glist,'pyg_hetlist_tv25.pt')

# pyg_hetlist_tv25 to pyg_hetlist_tv25_6merrc_____________________________________________________________________________________________________________________
tv25 = torch.load('pyg_hetlist_tv25.pt')
adata_6mer = sc.read('6mer_adata_co.h5ad') #adata of all atac 6mer vectors

def rev_comp(motif): #returns reverse compliment of given DNA motif
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
    seqtup = sorted(list(set((seqnames.index(seq),seqnames.index(rev_comp(seq))))))
    if seqtup not in rc_seqs:
        rc_seqs.append(seqtup)

#combine 6mer arrays into reverse-compliment-agnostic arrays
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

torch.save(new_tv,'pyg_hetlist_tv25_6merrc.pt')

# pyg_hetlist_tv25_6merrc to pyg_hetlist_tv25_6merrchs_________________________________________________________________________________________________________________
#define and then remove the lower standard-deviation-psm 50% of reverse-compliment-agnostic arrays
tv25 = torch.load('pyg_hetlist_tv25_6merrc.pt')

#score all t->motif relationships among promoters, summation
promotif_smatrix = torch.zeros((16,len(rc_seqs)))
for g in tv25:
    for pert_num,pv in enumerate(g[('perturbation','pvalue','promoter')]['edge_attr']):
        if pv>=2:
            fc = np.log2(g[('perturbation','foldchange','promoter')]['edge_attr'][pert_num]+1/1024)
            promotif_smatrix[pert_num,:]+=fc*g['promoter']['x'][0]
with open('promotif_smatrix_rc.pkl','wb') as pfile:
    pkl.dump(promotif_smatrix,pfile)

mat = pkl.load(open('promotif_smatrix_rc.pkl','rb'))
stl = []
for seq_ind,rctup in enumerate(rc_seqs):
    seq_vals = []
    for t in range(len(mat)):
        val = mat[t][seq_ind].item()
        seq_vals.append(pos(mat[t],val))
    st_thisseq = np.std(seq_vals)
    stl.append(st_thisseq)

high_std = [elem>np.median(stl) for elem in stl]

hse = torch.Tensor([True]+high_std) #add starting True to account for distance element of enhancer.x
hse = hse.reshape((1,-1))>0 #reshape and cast to bool
hsp = torch.Tensor(high_std)
hsp = hsp.reshape((1,-1))>0 #reshape and cast to bool

for graph in tv25:
    graph['promoter'].x = graph['promoter'].x[hsp].reshape(1,-1)
    enum = len(graph['enhancer'].x)
    hse_scaled = hse.repeat(enum,1)
    graph['enhancer'].x = graph['enhancer'].x[hse_scaled].reshape(enum,-1)    

torch.save(tv25,'pyg_hetlist_tv25_6merrchs.pt')

#training mask is separated into blocks of 100 to minimize shared enhancers appearing in both the training and test sets
block_size = 100
gblocks = list(range(int(len(glist)/block_size)+1))
testblocks = random.sample(gblocks,int(len(gblocks)/5))
bmask = [False]*len(glist)
for block in testblocks:
    for idx in range(block_size*block,block_size*block+block_size):
        bmask[idx] = True
with open('htbtrainmask.pkl','wb') as mfile:
    pkl.dump(bmask,mfile)
