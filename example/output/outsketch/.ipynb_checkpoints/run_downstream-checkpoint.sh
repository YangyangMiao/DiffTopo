#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem 50gb
#SBATCH --time 72:00:00
#SBATCH --mail-type=ALL

source /home/ymiao/mambaforge/bin/activate /home/ymiao/mambaforge/envs/RFAA/
module load gcc
module load cuda/12.1.1 
module load cudnn/8.9.7.29-12 
cd /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch
/home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/run_inference.py inference.output_prefix=/work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch0/sketch_loop inference.input_pdb=./sketch0.pdb 'contigmap.contigs=[3/A1-6/7/A14-22/6/A29-35/5/A41-45/6/A52-55/4/A60-66/23/B72-76/5/B82-92/8/B101-106/8/B115-125/7/B133-137/6/B144-151/3]' inference.num_designs=1 
cd /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch0 && /home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/run_inference.py inference.output_prefix=./partialdiff inference.input_pdb=./sketch_loop_0.pdb 'contigmap.contigs=[175-175]' inference.num_designs=3 diffuser.partial_T=10 
/home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/genesis/cut_partial_to_dimer.py --info /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch.npy --input_dir /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_0_cut.pdb --chains 'A,B' --outdir ./partialdiff_0 --seqnum 16 --temp 0.2 --bias False 
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_1_cut.pdb --chains 'A,B' --outdir ./partialdiff_1 --seqnum 16 --temp 0.2 --bias False 
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_2_cut.pdb --chains 'A,B' --outdir ./partialdiff_2 --seqnum 16 --temp 0.2 --bias False 
cd /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch
/home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/run_inference.py inference.output_prefix=/work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch1/sketch_loop inference.input_pdb=./sketch1.pdb 'contigmap.contigs=[3/B1-4/6/B11-15/2/B18-22/2/B25-31/6/B38-41/5/B47-52/4/B57-66/5/B72-86/23/A92-95/5/A101-107/4/A112-117/3/A121-128/6/A135-138/6/A145-154/5/A160-162/7/A170-175/8/A184-192/3]' inference.num_designs=1 
cd /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch1 && /home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/run_inference.py inference.output_prefix=./partialdiff inference.input_pdb=./sketch_loop_0.pdb 'contigmap.contigs=[216-216]' inference.num_designs=3 diffuser.partial_T=10 
/home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/genesis/cut_partial_to_dimer.py --info /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch.npy --input_dir /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_0_cut.pdb --chains 'A,B' --outdir ./partialdiff_0 --seqnum 16 --temp 0.2 --bias False 
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_1_cut.pdb --chains 'A,B' --outdir ./partialdiff_1 --seqnum 16 --temp 0.2 --bias False 
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_2_cut.pdb --chains 'A,B' --outdir ./partialdiff_2 --seqnum 16 --temp 0.2 --bias False 
cd /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch
/home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/run_inference.py inference.output_prefix=/work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch2/sketch_loop inference.input_pdb=./sketch2.pdb 'contigmap.contigs=[3/A1-3/3/A7-10/2/A13-20/2/A23-27/3/A31-35/3/A39-48/5/A54-57/3/A61-67/3/A71-73/2/A76-78/4/A83-87/4/A92-94/23/B100-102/4/B107-110/6/B117-119/6/B126-130/4/B135-141/5/B147-150/6/B157-164/4/B169-171/5/B177-181/3/B185-188/3/B192-195/4/B200-202/3]' inference.num_designs=1 
cd /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch2 && /home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/run_inference.py inference.output_prefix=./partialdiff inference.input_pdb=./sketch_loop_0.pdb 'contigmap.contigs=[226-226]' inference.num_designs=3 diffuser.partial_T=10 
/home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/genesis/cut_partial_to_dimer.py --info /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch.npy --input_dir /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_0_cut.pdb --chains 'A,B' --outdir ./partialdiff_0 --seqnum 16 --temp 0.2 --bias False 
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_1_cut.pdb --chains 'A,B' --outdir ./partialdiff_1 --seqnum 16 --temp 0.2 --bias False 
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_2_cut.pdb --chains 'A,B' --outdir ./partialdiff_2 --seqnum 16 --temp 0.2 --bias False 
cd /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch
/home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/run_inference.py inference.output_prefix=/work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch3/sketch_loop inference.input_pdb=./sketch3.pdb 'contigmap.contigs=[3/A1-4/2/A7-10/2/A13-15/3/A19-21/8/A30-32/3/A36-39/6/A46-48/3/A52-55/3/A59-61/4/A66-69/2/A72-73/2/A76-78/5/A84-87/4/A92-94/6/A101-102/2/A105-108/23/B112-113/4/B118-122/6/B129-131/5/B137-141/7/B149-150/7/B158-162/4/B167-171/4/B176-182/6/B189-194/4/B199-203/4/B208-213/6/B220-222/6/B229-231/6/B238-242/7/B250-252/5/B258-261/3]' inference.num_designs=1 
cd /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch3 && /home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/rfdiffusion_cloab/RFdiffusion/run_inference.py inference.output_prefix=./partialdiff inference.input_pdb=./sketch_loop_0.pdb 'contigmap.contigs=[287-287]' inference.num_designs=3 diffuser.partial_T=10 
/home/ymiao/mambaforge/envs/RFAA/bin/python /work/lpdi/users/ymiao/genesis/cut_partial_to_dimer.py --info /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch/new_sketch.npy --input_dir /work/lpdi/users/ymiao/code/DiffTopo/example/output/outsketch
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_0_cut.pdb --chains 'A,B' --outdir ./partialdiff_0 --seqnum 16 --temp 0.2 --bias False 
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_1_cut.pdb --chains 'A,B' --outdir ./partialdiff_1 --seqnum 16 --temp 0.2 --bias False 
/work/lpdi/users/ymiao/colabenv/bin/python /work/lpdi/users/ymiao/new_colabdesign/ColabDesign/run_mpnnaf2_dimer.py --fixpos '' --designpdb ./partialdiff_2_cut.pdb --chains 'A,B' --outdir ./partialdiff_2 --seqnum 16 --temp 0.2 --bias False 
