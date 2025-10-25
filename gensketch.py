import lightning
import data
import yaml   
import argparse
import os 
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store', type=str,default= './ckpt1121/diffsketch_epoch=587.ckpt',)
parser.add_argument('--data', action='store', type=str, default='../12SSE_node_distmat_edgeattr_onehot_len.pt')
parser.add_argument('--outputdir', action='store', type=str, default='./sample_output')
parser.add_argument('--keep_frames', action='store', type=int, default=2)
parser.add_argument('--device', action='store', type=str,default='cuda')
args = parser.parse_args(args=[])
model = lightning.DDPM.load_from_checkpoint(args.checkpoint,map_location=args.device,data_path=args.data,torch_device=args.device)


from data import *
model.setup(stage='val')
dataset = SketchDataset(args.data,'torch',args.device)
dataloader = get_dataloader(dataset, batch_size=64, shuffle=False,torch_device=args.device)
full_length_h=[]
for data in dataloader:
#     print(data['length'])
    if len(data['length']) <63:
        break
    for i in range(64):
        if data['length'][i]==36:
            full_length_h.append(data['h'][i])
    print(len(full_length_h))
    if len(full_length_h)>128:
        break
h0=full_length_h[0]
for i in range(1,128):
    h0 = torch.cat([h0,full_length_h[i]],dim=0)
h0=h0.reshape(128,36,2)


fake_data={}
fake_data['h']=h0.repeat(2,1,1)
fake_data['x']=torch.zeros([256,36,3]).to(h0.device)
fake_data['node_mask'] = torch.ones(36).repeat(256,1).to(h0.device)
fake_data['edge_mask'] = torch.ones(36,36).repeat(256,1,1).to(h0.device)
fake_data['length'] = torch.tensor(36).repeat(256).to(h0.device)

import time
# for data in dataloader:
t0 = time.time()
chain0_list=[]
for i in range(0,1000):
    chain_batch,chain0_batch, node_mask_batch = model.sample_chain(fake_data, keep_frames=args.keep_frames)
    chain0_list.append(chain0_batch)
    print(i,time.time() - t0)
    t0 = time.time()
chain0_list_tensor =chain0_list[0]
for i in range(1,10):
     chain0_list_tensor= torch.cat((chain0_list_tensor,chain0_list[i]))
crds= chain0_list_tensor[:,:,0:3]
sse = chain0_list_tensor[:,:,3:]
np.save('sketch12_1000_crd',crds.cpu().numpy())
np.save('sketch12_1000_sse',sse.cpu().numpy())