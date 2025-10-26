import torch
import os
import json
from biotite.structure.io.pdb import PDBFile
import numpy as np
import pickle

class CG_struct:
    def __init__(self,pdbpath,sspath,filename):
        self.name = filename
        self.sspath = sspath
        self.CG_crd, self.sslist = self.PDB2CG(pdbpath,filename)
        self.length,self.h,self.x,self.node_mask,self.edge_mask = self.cg2graph()
    def between(self,i,j,k):
        if i >= j and i <= k:
            
            return True
        else :
            return False
        
    def PDB2CG(self,datapath,filename):
        structure = PDBFile.read(datapath+filename)
        chain = structure.get_structure()[0]
        ssinfo = self.readSS(self.sspath ,filename+'.ss')
        cid = list(ssinfo.keys())[0]
        ss_crd_list = []
        ss_type_list = []
        CG_list =[]
        total_ca_list=[]
        for line in chain:
            if line.chain_id == cid and line.atom_name == 'CA':
                total_ca_list.append(line.coord)
        total_ca_list = np.array(total_ca_list)
        mass_center = total_ca_list.mean(axis=0)
        for ss in ssinfo[cid]:
            tmp_cg =[]
            ss_crd = []
            for line in chain:
                if line.chain_id == cid and line.atom_name == 'CA':
                    
                    if self.between(line.res_id,ss[1],ss[2]):
                        ss_crd.append(line.coord)
            ss_crd = np.array(ss_crd) - mass_center

            # print(ss_crd.mean(axis=0))
            
            if ss[0] == 'H':
                tmp_cg.append([ss_crd[0:3].mean(axis=0),ss_crd.mean(axis=0),ss_crd[-3:].mean(axis=0)])
                
            else:
                tmp_cg.append([ss_crd[0:2].mean(axis=0),ss_crd.mean(axis=0),ss_crd[-2:].mean(axis=0)])
            for i in range(3):
                ss_type_list.append(ss[0])
            CG_list.append(tmp_cg)    
        CG_list = np.array(CG_list).reshape(-1,3)
        if len(CG_list)> 36:
            CG_list = CG_list[0:36]
            ss_type_list = ss_type_list[0:36]
        return CG_list, ss_type_list
    
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

    def cg2graph(self,maxsse = 12):
        cgcrd = self.CG_crd
        sslist = self.sslist
        feature_dict={'H':[1,0],'E':[0,1]}
        graph = {}
        h= np.zeros([maxsse*3,2],dtype = int)
        x = np.zeros([maxsse*3,3])
        for i in range(0,len(sslist)):
            h[i] = feature_dict[sslist[i]]
        x[0:len(cgcrd)]+= cgcrd
        length = torch.tensor(len(cgcrd),dtype=int)
        h = torch.tensor(torch.from_numpy(h),dtype=torch.long)
        x = torch.tensor(torch.from_numpy(x),dtype=torch.float32)
        node_mask = torch.zeros(maxsse*3,dtype=torch.float32)
        node_mask[0:length] = 1.
        node_mask = node_mask.reshape(36,1)
        diag_mask = 1-torch.eye(36)
        edge_mask = (node_mask*node_mask.T)*diag_mask     
        return length,h,x,node_mask,edge_mask

from multiprocessing.dummy import Pool as ThreadPool
import argparse
import os
import torch
from multiprocessing.pool import ThreadPool

def run_process(args_tuple):
    """Process a single file with provided paths"""
    filename, pdbpath, sspath = args_tuple
    try:
        tmp_data = {}
        cg = CG_struct(pdbpath, sspath, filename)
        tmp_data['h'] = cg.h
        tmp_data['x'] = cg.x
        tmp_data['length'] = cg.length
        tmp_data['id'] = filename
        tmp_data['node_mask'] = cg.node_mask
        tmp_data['edge_mask'] = cg.edge_mask
        return tmp_data
    except Exception as e:
        print(f'{filename} failed! Error: {e}')
        return None

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='E3 coarse grained Topology Diffusion Data Preprocessing')
    p.add_argument('--pdbdir', action='store', type=str, default="/work/lpdi/databases/CATH_S40/cath/dompdb/")
    p.add_argument('--pdbssdir', action='store', type=str, default="/work/lpdi/databases/CATH_S40/cath/pdbss/")
    p.add_argument('--workers', action='store', type=int, default=5)
    p.add_argument('--savedir', action='store', type=str, default='/work/lpdi/users/ymiao/CATH.pt')
    
    # Remove args=[] to use actual command line arguments
    args = p.parse_args()
    
    pdbpath = args.pdbdir
    sspath = args.pdbssdir
    filenames = os.listdir(pdbpath)
    
    # Create tuples of (filename, pdbpath, sspath) for each file
    process_args = [(filename, pdbpath, sspath) for filename in filenames]
    
    # Process with thread pool
    pool = ThreadPool(args.workers)
    results = pool.map(run_process, process_args)
    pool.close()
    pool.join()
    
    # Filter out None results (failed processes)
    results = [r for r in results if r is not None]
    
    # Save results
    torch.save(results, args.savedir)
    print(f'Saved {len(results)} structures to {args.savedir}')


