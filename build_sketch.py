import os
import sys
from minitopobuilder.form.Form import Form
from minitopobuilder.virtual.VirtualBeta import VirtualBeta
from minitopobuilder.virtual.VirtualHelix import VirtualHelix
import numpy as np
from biotite.structure.io.pdb import PDBFile
import json 
import argparse
import warnings
import re
from pathlib import Path

warnings.filterwarnings("ignore")


def createss(ss1, sstype, chain_id, mass_centroid=None):
    """
    Create secondary structure elements
    
    Parameters:
    ss1: Secondary structure coordinates
    sstype: Secondary structure type ('H' for helix, 'E' for beta sheet)
    chain_id: Chain ID
    mass_centroid: Center of mass coordinates
    """
    tcentorid = ss1.mean(axis=0)
    dis = np.linalg.norm(ss1[0] - ss1[-1])
    ss1 = ss1 - tcentorid
    
    if sstype == 'H':
        aalen = np.random.randint(int(dis/1.59)+3, int(dis/1.59)+5, 1)
        h1 = VirtualHelix(aalen, [0., 0., 0.], chain=chain_id)
    elif sstype == 'E':
        aalen = np.random.randint(int(dis/3.2)+2, int(dis/3.2)+4, 1)
        h1 = VirtualBeta(aalen, [0., 0., 0.], chain=chain_id)    
    
    h1.create_val_sequence()
    h1data = h1.atom_data()
    cacrd_o = h1data[h1data['atomtype']=='CA'][['x','y','z']].to_numpy()
    ss1o = np.array([cacrd_o[0:3].mean(axis=0), cacrd_o[:].mean(axis=0), cacrd_o[-3:].mean(axis=0)])
    centroid = ss1o.mean(axis=0)
    ss1o = ss1o - centroid
    h1.shift(-centroid[0], -centroid[1], -centroid[2])
    h1.rotate(ss1o, ss1)
    h1data = h1.atom_data()
    newcacrd_o = h1data[h1data['atomtype']=='CA'][['x','y','z']].to_numpy()
    h1.shift(tcentorid[0], tcentorid[1], tcentorid[2])
    
    if mass_centroid is not None:
        h1.shift(mass_centroid[0], mass_centroid[1], mass_centroid[2])
    
    ss_ = [sstype for i in range(0, aalen[0])]
    return h1, ss_


def get_contig(data):
    """generate contig string"""
    result = ['3/']
    current_group = [data[0]]
    
    for i in range(1, len(data)):
        if data[i][0] == data[i - 1][0] + 1 and data[i][1:] == data[i - 1][1:]:
            current_group.append(data[i])
        else:
            start_idx = current_group[0][0]
            end_idx = current_group[-1][0]
            next_start_idx = data[i][0]
            length = next_start_idx - end_idx - 1
            letter = current_group[0][1]
            
            if current_group[0][1] != data[i][1]:
                result.append(f"{letter}{start_idx}-{end_idx}/23/ ")
            else:
                result.append(f"{letter}{start_idx}-{end_idx}/{length}/")
            current_group = [data[i]]
    
    if current_group:
        start_idx = current_group[0][0]
        end_idx = current_group[-1][0]
        letter = current_group[0][1]
        result.append(f"{letter}{start_idx}-{end_idx}/3")
    
    output = ''.join(result)
    return output


def get_length(input_string):
    """get contig length"""
    matches = re.findall(r'(\d+)|([A-Z])(\d+)-(\d+)', input_string)
    totals = 0
    for match in matches:
        if match[0]:
            length = int(match[0])
            totals += length
        elif match[1]:
            start = int(match[2])
            end = int(match[3])
            range_length = end - start + 1
            totals += range_length
    return totals


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='generate protein sketch from CG topo and generate scripts for rfdiffusion',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    

    parser.add_argument('--input', required=True, type=str,
                        help='input npy file ')
    parser.add_argument('--dimer',type=bool,default = False, help='input is single chain or dimer')
    parser.add_argument('--output', type=str, default='./filtpdbs',
                        help='output')
    parser.add_argument('--filename', type=str, default='sketch',
                        help='output file name ')
    

    parser.add_argument('--filt_rg', type=bool, default=True,
                        help='filt too large radius gyration')
    parser.add_argument('--filt_longloop', type=bool, default=True,
                        help='filt too long loop fold')
    parser.add_argument('--looplimit', type=int, default=15,
                        help='max loop length limitation')
    

    parser.add_argument('--downstream', type=bool, default=True,
                        help='generate downstream script for rfdiff')
    
    # 工具路径参数
    parser.add_argument('--rfdiffusion_env', type=str, default='/home/ymiao/mambaforge/envs/RFAA/',
                        help='rfdiffusion env (exmple: /path/to/mambaforge/envs/RFAA/)')
    parser.add_argument('--rfdiffusion_path', type=str, default='/work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/',
                        help='RFdiffusion path ( /path/to/RFdiffusion/)')
    parser.add_argument('--colabdesign_env', type=str, default='/work/lpdi/users/ymiao/colabenv/',
                        help='ColabDesign dir ( /path/to/ColabDesign/)')
    parser.add_argument('--colabdesign_path', type=str, default='/work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py',
                        help='ColabDesign script for mpnn and af2 ( /path/to/ColabDesign/)')
    # SLURM parameter
    parser.add_argument('--partition', type=str, default='gpu',
                        help='SLURM partition')
    parser.add_argument('--gpu_type', type=str, default='1',
                        help='GPU number')
    parser.add_argument('--mem', type=str, default='50gb',
                        help='asked memory for downstream')
    parser.add_argument('--time', type=str, default='72:00:00',
                        help='slurm ')
    parser.add_argument('--cpus', type=int, default=1,
                        help='CPU num')
    
    # 模块加载参数
    parser.add_argument('--gcc_module', type=str, default='gcc',
                        help='GCC module')
    parser.add_argument('--cuda_module', type=str, default='cuda/12.1.1',
                        help='CUDA module')
    parser.add_argument('--cudnn_module', type=str, default='cudnn/8.9.7.29-12',
                        help='cuDNN module')
    
    # RFdiffusion参数
    parser.add_argument('--num_designs', type=int, default=1,
                        help='RFdiffusion design num ')
    parser.add_argument('--partial_designs', type=int, default=3,
                        help='partial design num ')
    parser.add_argument('--partial_T', type=int, default=10,
                        help='partial diffusion step')
    
    # MPNN参数
    parser.add_argument('--mpnn_seqnum', type=int, default=16,
                        help='MPNN seq num')
    parser.add_argument('--mpnn_temp', type=float, default=0.2,
                        help='MPNN temperature')
    
    return parser.parse_args()


def validate_paths(args):
    if args.downstream:
        required_paths = {
            'rfdiffusion_env': args.rfdiffusion_env,
            'rfdiffusion_path': args.rfdiffusion_path,
            'colabdesign_env':args.colabdesign_env,
            'colabdesign_path':args.colabdesign_path
        }
        
        missing_paths = [name for name, path in required_paths.items() if not path]
        
        if missing_paths:
            print(f"\n error: if downstream, we need following:")
            for name in missing_paths:
                print(f"  --{name}")
            print("\nexample:")
            print("  --rfdiffusion_env /path/to/mambaforge/envs/RFAA/")
            print("  --rfdiffusion_path /path/to/RFdiffusion/")
            print("  --colabdesign_env /path/to/mambaforge/envs/colabdesignenv/")
            print("  --colabdesign_path /path/to/colabdesign/")
            sys.exit(1)

        if not Path(args.rfdiffusion_env).exists():
            print(f"warning: rfdiffusion env dosent exist: {args.conda_env}")
        if not Path(args.rfdiffusion_path).exists():
            print(f"warning:  RFdiffusion dosent exist: {args.rfdiffusion_path}")
        if not Path(args.colabdesign_env).exists():
            print(f"warning: colabdesign env dosent exist: {args.conda_env}")
        if not Path(args.colabdesign_path).exists():
            print(f"warning:  colabdesign path dosent exist: {args.rfdiffusion_path}")

def generate_downstream_script(args, sketchdata, outputdir):

    script_path = os.path.join(outputdir, 'run_downstream.sh')
    
    # get python
    python_bin = os.path.join(args.conda_env, 'bin', 'python')
    
    # get script path
    rfdiffusion_script = os.path.join(args.rfdiffusion_path, 'run_inference.py')
    
    with open(script_path, 'w') as f:
        # SLURM head
        f.write('#!/bin/bash\n')
        f.write(f'#SBATCH --nodes 1\n')
        f.write(f'#SBATCH --ntasks 1\n')
        f.write(f'#SBATCH --cpus-per-task {args.cpus}\n')
        f.write(f'#SBATCH --partition={args.partition}\n')
        f.write(f'#SBATCH --gres=gpu:{args.gpu_type}\n')
        f.write(f'#SBATCH --mem {args.mem}\n')
        f.write(f'#SBATCH --time {args.time}\n')
        f.write(f'#SBATCH --mail-type=ALL\n\n')
        
        # activate Conda env
        conda_base = str(Path(args.conda_env).parent.parent / 'bin' / 'activate')
        f.write(f'source {conda_base} {args.conda_env}\n')
        
        # load module
        f.write(f'module load {args.gcc_module}\n')
        f.write(f'module load {args.cuda_module}\n')
        f.write(f'module load {args.cudnn_module}\n\n')
        
        # generate cmds
        for i in range(len(sketchdata)):
            length = sketchdata[i]['length']
            if len(length) > 1:
                newlength = length[0] + length[1]
            else:
                newlength = length[0]
            
            contig = sketchdata[i]['contig']
            currentdir = os.path.abspath(outputdir)
            rfoutdir = os.path.join(currentdir, f'new_sketch{i}')
            
            f.write(f'# generate sketch {i}\n')
            f.write(f'mkdir -p {rfoutdir}\n')
            f.write(f'cd {currentdir}\n')
            
            # RFdiffusion loop设计
            f.write(f'{python_bin} {rfdiffusion_script} ')
            f.write(f'inference.output_prefix={rfoutdir}/sketch_loop ')
            f.write(f'inference.input_pdb=./sketch{i}.pdb ')
            f.write(f"'contigmap.contigs=[{contig}]' ")
            f.write(f'inference.num_designs={args.num_designs}\n')
            
            # RFdiffusion partial diffusion
            f.write(f'cd {rfoutdir}\n')
            f.write(f'{python_bin} {rfdiffusion_script} ')
            f.write(f'inference.output_prefix=./partialdiff ')
            f.write(f'inference.input_pdb=./sketch_loop_0.pdb ')
            f.write(f"'contigmap.contigs=[{newlength}-{newlength}]' ")
            f.write(f'inference.num_designs={args.partial_designs} ')
            f.write(f'diffuser.partial_T={args.partial_T}\n')
            
            if args.dimer:            
                genesis_script = os.path.join('./', 'cut_partial_to_dimer.py')
                f.write(f'{python_bin} {genesis_script} ')
                f.write(f'--info {outputdir}/new_sketch.npy ')
                f.write(f'--input_dir {outputdir}\n')
            
            if args.colabdesign_path:
                colabdesign_script = os.path.join(args.colabdesign_path, 'run_mpnnaf2_dimer.py')
                for j in range(args.partial_designs):
                    f.write(f'{python_bin} {colabdesign_script} ')
                    f.write(f"--fixpos '' ")
                    if args.dimer:  
                        f.write(f'--designpdb ./partialdiff_{j}_cut.pdb ')
                    else:
                        f.write(f'--designpdb ./partialdiff_{j}.pdb ')
                    f.write(f"--chains 'A,B' ")
                    f.write(f'--outdir ./partialdiff_{j} ')
                    f.write(f'--seqnum {args.mpnn_seqnum} ')
                    f.write(f'--temp {args.mpnn_temp} ')
                    f.write(f'--bias False\n')
            
            f.write('\n')
    
    print(f'generate downstream script: {script_path}')
    print(f'use slurm to submit script: sbatch {script_path}')


def main():

    args = parse_arguments()

    validate_paths(args)

    outputdir = args.output
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    else:
        print(f'clean : {outputdir}')
        os.system(f'rm -rf {outputdir}/*')
    
    # 加载输入数据
    print(f'load input : {args.input}')
    try:
        info = np.load(args.input, allow_pickle=True)
    except Exception as e:
        print(f'error: cant load input: {e}')
        sys.exit(1)
    
    crd = info[:, 0:3]
    sse = info[:, 3:5]
    chain_id = info[:, 5]
    batch = info[:, 6]
    
    filtrg = args.filt_rg
    filt_longloop = args.filt_longloop
    
    data = []
    success = 0
    
    print(f'start {len(np.unique(batch))} structures...')
    
    for num in np.unique(batch):
        tmp_chain = chain_id[batch == num]
        tmp_sse = sse[batch == num]
        tmp_crd = crd[batch == num]
        new_sketch_dict = {}
        sse_num = int(tmp_sse.sum() / 3)
        pflag = 0
        sslist = []
        ss_str = []
        name = 'test' + str(num)
        
        loop_max_len = args.looplimit
        looplen = []

        #calclulate loop length
        for i in range(0, int(len(tmp_sse)/3) - 1):
            if tmp_chain[3*i+2] == tmp_chain[3*(i+1)]:
                looplen.append(np.linalg.norm(tmp_crd[3*i+2] - tmp_crd[3*(i+1)]))
            else:
                interchain_length = np.linalg.norm(tmp_crd[3*i+2] - tmp_crd[3*(i+1)])
        
        # filt long loop
        if filt_longloop:
            if not looplen or max(looplen) > loop_max_len or len(looplen) < 5:
                continue
        
        # create sse
        for i in range(0, sse_num):
            ss1 = np.array(tmp_crd[3*i:3*i+3])
            if (tmp_sse[3*i+1] == np.array([1., 0.])).all():
                ssetype = 'H'
            elif (tmp_sse[3*i+1] == np.array([0., 1.])).all():
                ssetype = 'E'
            
            if tmp_chain[3*i+1] == 0.:
                chaintype = 'A'
            else:
                chaintype = 'B'
            
            name += ssetype
            vs, ss_ = createss(ss1, ssetype, chaintype)
            ss_str += ss_
            sslist.append(vs)
            
            if ssetype == 'H' and vs.residues >= 6:
                pflag += 1
            if ssetype == 'E' and vs.residues >= 4:
                pflag += 1
        
        # create pdb
        f = Form('test', sslist, None)
        sspos = f.prepare_coords()
        df = f.to_frame()
        np_data = df.to_numpy()
        
        tmp_pdb = os.path.join(outputdir, 'tmp.pdb')
        with open(tmp_pdb, "w") as fd:
            fd.write(f.to_pdb())
        
        # calculate rg
        structure = PDBFile.read(tmp_pdb)
        chain = structure.get_structure()[0]
        pdbcrd = chain[(chain.atom_name == 'CA')].coord
        N = len(pdbcrd)
        
        if filtrg:
            rf_rg = 0.395 * N**(3/5) + 8.0
        else:
            rf_rg = 1000
        
        RG = np.mean(np.linalg.norm(pdbcrd - pdbcrd.mean(axis=0), axis=1))
        
        # filt
        if RG < rf_rg * 1.2:
            print(f'sucess {success}: SSE number={len(looplen)+2}, '
                  f'max loop length={max(looplen):.2f}, '
                  f'interchain ={interchain_length:.2f}')
            
            output_pdb = os.path.join(outputdir, f'sketch{success}.pdb')
            with open(output_pdb, "w") as fd:
                fd.write(f.to_pdb())
            
            success += 1
            
            # prepare data 
            reshapedata = np_data[:, 0:6].reshape(int(len(np_data)/4), 4, 6)
            ss_str_ = np.repeat(np.array(ss_str)[:, None], [4], axis=1)
            new_data = np.concatenate([reshapedata, ss_str_[:, :, None]], axis=-1)
            new_sketch_dict['info'] = new_data
            
            # generate contig
            contig_need = []
            for iline in new_data:
                contig_need.append([iline[1][3], iline[1][4], iline[1][6]])
            contig = get_contig(contig_need)
            
            new_sketch_dict['contig'] = ''.join(contig.split(' '))
            new_sketch_dict['dir'] = output_pdb
            
            if len(contig.split(' ')) > 1:
                length1 = get_length(contig.split(' ')[0])
                length2 = get_length(contig.split(' ')[1])
                new_sketch_dict['length'] = [length1, length2]
            else:
                length = get_length(contig.split(' ')[0])
                new_sketch_dict['length'] = [length]
            
            data.append(new_sketch_dict)
    
    # save data
    data = np.array(data)
    output_npy = os.path.join(outputdir, 'new_sketch.npy')
    np.save(output_npy, data)
    print(f'\n generate {success} sketches')
    print(f'data saved to: {output_npy}')
    
    # generate downstream
    if args.downstream and success > 0:
        print('\n generate downstream ...')
        sketchdata = np.load(output_npy, allow_pickle=True)
        generate_downstream_script(args, sketchdata, outputdir)


if __name__ == '__main__':
    main()