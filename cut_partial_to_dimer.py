import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import argparse
def cut_linker(file_path,output_path,length_A):
    file_path = file_path #"./test_dimer/test_rfdiff/Onechainlinker20partialdiff_0.pdb"
    structure = strucio.load_structure(file_path)
    
    # Get mask for residues with ID 128–148
    mask_delete = (structure.res_id > length_A) & (structure.res_id <=length_A+20)
    
    # # Remove residues 128–148
    structure = structure[~mask_delete]
    
    # # Get mask for residues with ID > 149
    mask_chain_b = structure.res_id >length_A+20
    
    # # Set the chain ID for these residues to 'B'
    structure.chain_id[mask_chain_b] = "B"
    
    renumbered_structure = structure.copy()
    
    # Get unique residue IDs for each chain
    chain_ids = structure.chain_id
    residue_ids = structure.res_id
    unique_chains = set(chain_ids)
    
    # Define a new numbering for chain B
    new_residue_ids_chain_b = range(1, len(set(residue_ids[chain_ids == "B"])) + 1)
    unique_res_ids_chain_b = sorted(set(residue_ids[chain_ids == "B"]))
    
    # Create a mapping for chain B
    res_id_mapping_chain_b = dict(zip(unique_res_ids_chain_b, new_residue_ids_chain_b))
    
    # Apply the renumbering only to chain B
    for i, (chain_id, old_res_id) in enumerate(zip(chain_ids, residue_ids)):
        if chain_id == "B":  # Only renumber chain B
            renumbered_structure.res_id[i] = res_id_mapping_chain_b[old_res_id]
    strucio.save_structure(output_path, renumbered_structure)

parser = argparse.ArgumentParser()
parser.add_argument('--info', action='store', type=str)
parser.add_argument('--input_dir', action='store', type=str)
parser.add_argument('--num', action='store', type=str)
args = parser.parse_args()

info = np.load(args.info,allow_pickle=True)#('./test_dimer/gen_1round_repeat/new_sketch.npy',allow_pickle=True)
input_dir= args.input_dir#'/work/lpdi/users/ymiao/genesis/test_dimer/gen_1round_repeat'
# for i in range(0,len(info)):
i = int(args.num)
try:
    length_A= info[i]['length'][0]-20
    for j in range(0,3):
        cut_linker(input_dir+'/new_sketch'+str(i)+'/partialdiff_'+str(j)+'.pdb',input_dir+'/new_sketch'+str(i)+'/partialdiff_'+str(j)+'_cut.pdb',length_A)
except:
    print(i,'not exist')
