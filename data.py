import torch
from torch_geometric.data import Dataset,Data

import json
from biotite.structure.io.pdb import PDBFile
import numpy as np
import pickle
from itertools import combinations
import math

import time
import itertools
import os
import torch
# from torch.utils.data import Dataset, DataLoader
import multiprocessing
from multiprocessing import Pool

class InMemorySSEDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graphs = []
        self.load_data()

    def load_data(self):
        file_paths = [os.path.join(self.data_dir, file_name) 
                      for file_name in os.listdir(self.data_dir) 
                      if file_name.endswith('.pt')]

        with Pool(processes=multiprocessing.cpu_count()) as pool:
            self.graphs = pool.map(torch.load, file_paths)

    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.graphs)

class SSEDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.data_dir=data_dir
        self.processed_files=self.processed_file_names
    @property
    def processed_file_names(self):
        file_names= []
        for i in os.listdir(self.data_dir):
            if '.pt' in i:
                file_names.append(i)
        return file_names

    def len(self):
        return len(self.processed_files)

    def get(self, idx):
        data = torch.load(osp.join(self.data_dir, self.processed_file_names[idx]))
        return data

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class CG_struct:
    def __init__(self,pdbpath,sspath,filename):
        self.name = filename
        self.sspath = sspath
        # CG_crd, CG_ss, CG_cid = self.PDB2CG(pdbpath,filename)
        self.pair_CG_crd, self.pair_CG_ss,self.pair_CG_cid,self.interact_chain = self.PDBpair2CG(pdbpath,filename)
        self.pair_CG_cid_bn= self.convert_to_binary(self.pair_CG_cid)
        self.graph =[]
        for CG_crd,CG_ss,CG_cid in zip(self.pair_CG_crd, self.pair_CG_ss, self.pair_CG_cid_bn):
            self.graph.append(self.cg2graph(CG_crd,CG_ss,CG_cid))

    def between(self,i,j,k):
        if i >= j and i <= k:
            return True
        else :
            return False
    def convert_to_binary(self,lists):
        binary_lists = []
        
        for lst in lists:
            # Find the most frequent character
            most_frequent_char = max(set(lst), key=lst.count)
            
            # Convert the list to binary: 0 for most frequent char, 1 for others
            binary_list = [0 if char == most_frequent_char else 1 for char in lst]
            binary_lists.append(binary_list)
        return binary_lists
        
    def PDB2CG(self,datapath,filename):
        structure = PDBFile.read(datapath+filename)
        # chain = structure.get_structure()[0]
        chains = structure.get_structure()[0]
        if '.pdb' in filename:
            ssinfo = self.readSS(self.sspath ,filename[0:-4]+'.ss')
        else:
            ssinfo = self.readSS(self.sspath ,filename+'.ss')
        total_mass_center=chains[(chains.atom_name == 'CA')].coord.mean(axis=0)
        CG_crd=[]
        CG_ss=[]
        CG_cid=[]
        # only use one pair 
        for cid in list(ssinfo.keys()):
            calist=chains[(chains.chain_id == cid) & (chains.atom_name == 'CA')]
            # tmp_cg=[]
            for ss in ssinfo[cid]:
                tmp_ss=calist[(calist.res_id>=ss[1]) &(calist.res_id<=ss[2])]
                ss_crd = tmp_ss.coord - total_mass_center
                CG_ss.append(ss[0])
                CG_cid.append(cid)
                if ss[0] == 'H':
                    CG_crd.append([ss_crd[0:3].mean(axis=0),ss_crd.mean(axis=0),ss_crd[-3:].mean(axis=0)])
                else:
                    CG_crd.append([ss_crd[0:2].mean(axis=0),ss_crd.mean(axis=0),ss_crd[-2:].mean(axis=0)])
        CG_crd =torch.from_numpy( np.array(CG_crd).reshape(-1,3))
        return CG_crd, CG_ss,CG_cid
    def PDBpair2CG(self,datapath,filename):
        structure = PDBFile.read(datapath+filename)
        # chain = structure.get_structure()[0]
        chains = structure.get_structure()[0]
        chain_ids = list(set(chains.chain_id))
        if len(chain_ids)==1:
            CG_crd, CG_ss,CG_cid = self.PDB2CG(datapath,filename)
            return [CG_crd],[CG_ss],[CG_cid],[(CG_cid[0],'')]
        else:
            chain_pairs = list(itertools.combinations(chain_ids, 2))
            interact_chain=[]
            for pair in chain_pairs:
                chain_A = chains[(chains.chain_id == pair[0]) & (chains.hetero==False) ]
                chain_B = chains[(chains.chain_id == pair[1]) & (chains.hetero==False) ]
                coords_A = chain_A.coord
                coords_B = chain_B.coord
                dist_matrix = np.linalg.norm(coords_A[:, np.newaxis, :] - coords_B[np.newaxis, :, :], axis=-1)
                interaction_cutoff = 5.0
                interacting_pairs = np.where(dist_matrix < interaction_cutoff)
                interacting_residues_A = set(chain_A.res_id[interacting_pairs[0]])
                interacting_residues_B = set(chain_B.res_id[interacting_pairs[1]])
                if len(interacting_residues_A)>20 or len(interacting_residues_B)>20:
                    interact_chain.append(pair)
            if '.pdb' in filename:
                ssinfo = self.readSS(self.sspath ,filename[0:-4]+'.ss')
            else:
                ssinfo = self.readSS(self.sspath ,filename+'.ss')
            total_mass_center=chains[(chains.atom_name == 'CA')].coord.mean(axis=0)
    
            pairs_crd = []
            pairs_ss = []
            pairs_cid = []
            for pair in interact_chain:
                pair_CG_crd=[]
                pair_CG_ss=[]
                pair_CG_cid=[]
                pair_mass_center=chains[(chains.atom_name == 'CA') & ((chains.chain_id==pair[0])|(chains.chain_id==pair[1]))].coord.mean(axis=0)
                for cid in pair:
                    calist=chains[(chains.chain_id == cid) & (chains.atom_name == 'CA')]
                    for ss in ssinfo[cid]:
                        tmp_ss=calist[(calist.res_id>=ss[1]) &(calist.res_id<=ss[2])]
                        ss_crd = tmp_ss.coord - pair_mass_center
                        pair_CG_ss.append(ss[0])
                        pair_CG_cid.append(cid)
                        if ss[0] == 'H':
                            pair_CG_crd.append([ss_crd[0:3].mean(axis=0),ss_crd.mean(axis=0),ss_crd[-3:].mean(axis=0)])
                        else:
                            pair_CG_crd.append([ss_crd[0:2].mean(axis=0),ss_crd.mean(axis=0),ss_crd[-2:].mean(axis=0)])  
                pair_CG_crd =torch.from_numpy( np.array(pair_CG_crd).reshape(-1,3))
                pairs_crd.append(pair_CG_crd)
                pairs_ss.append(pair_CG_ss)
                pairs_cid.append(pair_CG_cid)
            return pairs_crd, pairs_ss,pairs_cid,interact_chain
    
    def readSS(self,datapath,filename):
        SS_dict={'310Helix':'H','AlphaHelix':'H','PiHelix':'H','Strand':'E'}
        lines = open(datapath+'/'+filename,'r').readlines()
        SSinfo={} # data structure: chain:[sstype, start residue, end residue]
        for line in lines:
            s = line.split()
            if s[0] == 'LOC':
                if s[1] in SS_dict.keys():
                    if s[3].isdigit() and s[6].isdigit():
                        if s[4] not in SSinfo.keys():
                            if SS_dict[s[1]] =='H' and int(s[6])-int(s[3])>5: # do clean
                                SSinfo[s[4]] = []
                                SSinfo[s[4]].append([SS_dict[s[1]],int(s[3]),int(s[6])])
                            elif SS_dict[s[1]] =='E' and int(s[6])-int(s[3])>2:
                                SSinfo[s[4]] = []
                                SSinfo[s[4]].append([SS_dict[s[1]],int(s[3]),int(s[6])])
                            else:
                                continue
                        else:
                            if SS_dict[s[1]] =='H' and int(s[6])-int(s[3])>4: # do clean
                                SSinfo[s[4]].append([SS_dict[s[1]],int(s[3]),int(s[6])])
                            elif SS_dict[s[1]] =='E' and int(s[6])-int(s[3])>2:
                                SSinfo[s[4]].append([SS_dict[s[1]],int(s[3]),int(s[6])])
                            else:
                                continue
                    else:
                        continue
        #re order
        for key in SSinfo.keys():
            chain = SSinfo[key]
            chain.sort(key=lambda x:x[1])
            SSinfo[key] = chain
        return SSinfo
        
    def chain_index(self,string):
        chain_order = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        char_to_index = {char: i for i, char in enumerate(chain_order)}
        encoded = [ char_to_index[i] for i in string]
        return encoded
        
    def generate_range_list(self,data):
        range_list = []
        current_chain = data[0]
        count = 0
        for item in data:
            if item != current_chain:
                current_chain = item
                count += 30
            range_list.append(count)
            count += 1
        return range_list
        
    def fully_connected_edge_index(self,num_nodes):
        node_indices = torch.arange(num_nodes)
        edge_indices = list(combinations(node_indices, 2))
        edge_indices += [(j, i) for (i, j) in edge_indices]  # Adding reverse pairs for undirected edges
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return edge_index
        
    def fully_connected_edge_index_with_block_attr(self,num_nodes, points_per_block=3):
        num_blocks = int(num_nodes/points_per_block)
        edge_index = self.fully_connected_edge_index(num_nodes)
        block_indices = torch.arange(num_blocks).repeat_interleave(points_per_block)
    
        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.float)
        for i, (src, dest) in enumerate(edge_index.t()):
            src_block = block_indices[src]
            dest_block = block_indices[dest]
            if src_block == dest_block:
                edge_attr[i] = 1.0
        return edge_index, edge_attr
        
    def cg2graph(self,CG_crd,CG_ss,CG_cid):
        feature_dict={'H':[1.,0.],'E':[0.,1]}
        ss_feat=torch.tensor([feature_dict[i]*3 for i in CG_ss]).reshape(-1,2)
        # chain_feat=torch.tensor([[self.chain_index(i)]*3 for i in CG_cid]).reshape(-1,1)
        chain_feat=torch.tensor([[i]*3 for i in CG_cid]).reshape(-1,1)
        pos_embed=SinusoidalPosEmb(8)
        pos = torch.tensor(self.generate_range_list(chain_feat))/len(chain_feat) 
        pos_embed=pos_embed(pos)
        node_feat=torch.concat([ss_feat,chain_feat,pos_embed],dim=-1)
        num_nodes = len(CG_crd)
        edges,edge_sse_attr = self.fully_connected_edge_index_with_block_attr(num_nodes,3)
        node_row = CG_crd[edges[0, :]]
        node_col = CG_crd[edges[1, :]]
        edge_distance =1/ (torch.linalg.norm(node_row - node_col, axis=1)+1e-5)
        edge_neibor_attr = torch.where(torch.abs(edges[0]-edges[1])==1,1.,0.)
        edge_attr=torch.cat([edge_distance[:,None],edge_sse_attr[:,None],edge_neibor_attr[:,None]],dim=1)
        graph = Data(node_feat, edges, edge_attr, pos=CG_crd)
        return graph


if __name__=='__main__':
    data_path = '/Users/miaoyangyang/Desktop/Code/GraphAF_EGNN/12SSE_node_distmat_edgeattr_onehot_len.npy'
