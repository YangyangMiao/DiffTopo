import math
from tkinter import E
from unittest.mock import NonCallableMagicMock
import numpy as np
import torch
import torch.nn as nn
import sys 
sys.path.append('/work/lpdi/users/ymiao/geometric-vector-perceptron/geometric_vector_perceptron/')
from geometric_vector_perceptron import GVP_Network
import utils
from pdb import set_trace
from gvp import *
import json

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method, activation,
                 edges_in_d=0, nodes_att_dim=0, attention=False, normalization=None):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation)

        if normalization is None:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf)
            )
        elif normalization == 'batch_norm':
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
                nn.BatchNorm1d(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.BatchNorm1d(output_nf),
            )
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        # set_trace()
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            #set_trace()
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        #set_trace()
        #edge_attr = edge_attr.reshape(edge_attr.shape[0]*edge_attr.shape[1],-1)
        #print('shape x edge attr',x.shape, edge_attr.shape)
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, activation=nn.SiLU(), tanh=False, coords_range=10.0,reflection_equiv=True):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.reflection_equiv = reflection_equiv
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            layer)
        self.cross_product_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation,
            layer
        ) if not self.reflection_equiv else None
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff,coord_cross,edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        
        if not self.reflection_equiv:
            phi_cross = self.cross_product_mlp(input_tensor)
            if self.tanh:
                phi_cross = torch.tanh(phi_cross) * self.coords_range
            trans = trans + coord_cross * phi_cross
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        # if motif_mask is not None:
        #     agg = agg * motif_mask

        coord = coord + agg
        return coord

    def forward(
            self, h, coord, edge_index, coord_diff,coord_cross, edge_attr=None, node_mask=None, edge_mask=None
    ):
        # set_trace()
        coord = self.coord_model(h, coord, edge_index, coord_diff,coord_cross, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', activation=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum',reflection_equiv=True):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              activation=activation, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, activation=activation,
                                                       tanh=tanh, coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method,reflection_equiv=self.reflection_equiv))
        self.to(self.device)

    def forward(self, h, x, edge_index,batch_size, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        distances = 1/(distances + 1e-10)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        # set_trace()
        if self.reflection_equiv:
            coord_cross = None
        else:
            coord_cross = coord2cross(x, edge_index,node_mask,batch_size,self.norm_constant)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask
            )
        x = self._modules["gcl_equiv"](
            h, x,
            edge_index=edge_index,
            coord_diff=coord_diff,
            coord_cross=coord_cross,
            edge_attr=edge_attr,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', activation=nn.SiLU(), n_layers=3,
                 attention=False, norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1,
                 inv_sublayers=2, sin_embedding=False,normalization_factor=100, aggregation_method='sum',reflection_equiv=True):#, pos_embedding=True
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.reflection_equiv = reflection_equiv
        # if pos_embedding :
        #     self.pos_embedding = SinusoidalPosEmb(8)

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2 *2 +12
        else:
            self.sin_embedding = None
            edge_feat_nf = 2+2+8
        
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               activation=activation, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,reflection_equiv=reflection_equiv))
        self.to(self.device)

    def forward(self, h, x, edge_index, batch_size,node_mask=None, edge_mask=None,edge_attr=None):
        # Edit Emiel: Remove velocity as input
        
        distances, _ = coord2diff(x, edge_index)
        distances = 1/(distances + 1e-10)
        # set_trace()
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        # edge_space_attr = torch.where(distances<15.,1.,0.).reshape(-1,1)
        # self.edge_onesse_attr = torch.zeros()
        edge_neibor_attr = torch.where(torch.abs(edge_index[0]-edge_index[1])==1,1.,0.).reshape(-1,1)
        edge_neibor_attr = edge_neibor_attr* distances
        edge_no_neibor_attr = (1-edge_neibor_attr)* distances
        h = self.embedding(h)
        if edge_attr is not None:
            edge_sse_attr= edge_attr[...,0].reshape(-1,1)
            # edge_rlative_pos_attr = edge_attr[...,1:]
            edge_sse_attr = edge_sse_attr* distances
            edge_attr = torch.cat([distances,edge_neibor_attr,edge_sse_attr,edge_no_neibor_attr],dim=1)#edge_rlative_pos_attr
        else:    
            edge_attr = torch.cat([distances,edge_neibor_attr],dim=1)

        for i in range(0, self.n_layers):
            # x = x.reshape(batch_size,-1,3)
            # centermass = x.mean(axis=1).reshape(batch_size,1,3)
            # x = x - centermass.repeat(1,x.shape[1],1)
            # x = x.reshape(-1,3)
            # # set_trace()
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=edge_attr,
                batch_size=batch_size
            )
        
        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 activation=nn.SiLU(), n_layers=4, attention=False, normalization_factor=1,
                 out_node_nf=None, normalization=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, activation=activation,
                attention=attention, normalization=normalization))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()

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
    

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff

def coord2cross(x, edge_index, node_mask,batch_size, norm_constant=1):

    row, col = edge_index
    node_num= node_mask.reshape(batch_size,-1,1).shape[1]
    pt_num = torch.sum(node_mask.reshape(batch_size,-1,1),dim=1)
    masscenterx= torch.sum(x.reshape(batch_size,-1,3),dim=1)/pt_num # B,3
    rp_masscenterx = masscenterx.reshape(batch_size,1,3).repeat(1,node_num,1)
    new_x = x.reshape(batch_size,-1,3) - rp_masscenterx
    new_x = new_x.reshape(-1,3)
    cross = torch.cross(new_x[row],new_x[col],dim=1)
    norm = torch.linalg.norm(cross, dim=1, keepdim=True)
    cross = cross / (norm + norm_constant)
    return cross

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    #set_trace()
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


class Dynamics(nn.Module):
    def __init__(
            self, in_node_nf, context_node_nf, n_dims, hidden_nf=64, device='cpu', activation=nn.SiLU(),
            n_layers=4, attention=False, condition_time=True, tanh=False, norm_constant=0, inv_sublayers=2,
            sin_embedding=False, normalization_factor=100, aggregation_method='sum', model='egnn_dynamics',
            normalization=None,reflection_equiv=True,parameterfile=None
    ):
        super().__init__()
        self.device = device
        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.condition_time = condition_time
        self.model = model
        self.edge_relative_pos_emb = SinusoidalPosEmb(8) # add on 10.25 for relative pos embedding

        in_node_nf = in_node_nf + context_node_nf + condition_time
        if self.model == 'egnn_dynamics':
            self.dynamics = EGNN(
                in_node_nf=in_node_nf,
                in_edge_nf=1,
                hidden_nf=hidden_nf, device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                tanh=tanh,
                norm_constant=norm_constant,
                inv_sublayers=inv_sublayers,
                sin_embedding=sin_embedding,
                # pos_embedding=True,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                reflection_equiv=reflection_equiv,
            )
        elif self.model == 'gnn_dynamics':
            self.dynamics = GNN(
                in_node_nf=in_node_nf+3,
                in_edge_nf=0,
                hidden_nf=hidden_nf,
                out_node_nf=in_node_nf+3,
                device=device,
                activation=activation,
                n_layers=n_layers,
                attention=attention,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                normalization=normalization,
            )
        elif self.model == 'gvp_dynamics':
            # print(self.device)
            self.dynamics = GVP_Network(
                n_layers=n_layers, 
                feats_x_in=21, vectors_x_in=1,
                feats_x_out=21, vectors_x_out=1,
                feats_edge_in=9, vectors_edge_in=0,
                feats_edge_out=9, vectors_edge_out=0,
                embedding_nums=[], embedding_dims=[],
                edge_embedding_nums=[], edge_embedding_dims=[],vector_dim=3,recalc=2,device=self.device).to(self.device)
            # self.dynamics = self.dynamics.to(self.device)
            # import pdb
            # pdb.set_trace()
        elif self.model == 'gvp_dynamics_new':
            if parameterfile!=None:
                p= json.load(open(parameterfile,'r'))
            
                self.dynamics = GVPModel(node_in_dim=(int(21),int(0)), node_h_dim=p['node_h_dim'], node_out_nf=int(21),
                     edge_in_nf=int(0), edge_h_dim=(int(128),int(32)), edge_out_nf=(int(0),int(0)),
                     num_layers=n_layers, drop_rate=0.1, reflection_equiv=False,
                     d_max=20.0, num_rbf=16, update_edge_attr=False).to(device)
            else:
                self.dynamics = GVPModel(node_in_dim=(int(23),int(0)), node_h_dim=(256,32), node_out_nf=int(23),
                 edge_in_nf=int(0), edge_h_dim=(int(128),int(32)), edge_out_nf=(int(0),int(0)),
                 num_layers=n_layers, drop_rate=0.1, reflection_equiv=False,
                 d_max=40.0, num_rbf=32, update_edge_attr=False).to(device)
            
        else:
            raise NotImplementedError

        self.edge_cache = {}

    def forward(self, t, xh, node_mask, edge_mask, context):
        """
        - t: (B)
        - xh: (B, N, D), where D = 3 + nf
        - node_mask: (B, N, 1)
        - edge_mask: (B*N*N, 1)
        - context: (B, N, C)
        """
        has_batch_dim = len(xh.size()) > 2
        #set_trace()
        if has_batch_dim:
            bs, n_nodes = xh.shape[0], xh.shape[1]
            edges = self.get_edges(n_nodes, bs)  # (2, B*N)
            node_mask = node_mask.view(bs * n_nodes, 1)  # (B*N, 1)

            # Reshaping node features
            xh = xh.view(bs * n_nodes, -1).clone() * node_mask  # (B*N, D)

            # if motif_mask is not None:
            #     motif_mask = motif_mask.view(bs * n_nodes, 1)  # (B*N, 1)
            if context is not None:
                context = context.view(bs * n_nodes, self.context_node_nf)
            edge_sse_attr=torch.zeros(n_nodes,n_nodes)
            for i in range(0,int(n_nodes/3)):
                edge_sse_attr[3*i:3*i+3,3*i:3*i+3] = 1.
            edge_sse_attr = edge_sse_attr.repeat(bs,1,1)
            edge_sse_attr= edge_sse_attr.reshape(-1,1).to(xh.device)
            # add relative pos info in edge
            num_mat = torch.arange(n_nodes).repeat(n_nodes,1)
            edge_r_pos = (torch.abs(num_mat - num_mat.T)).repeat(bs,1,1)
            edge_r_pos = edge_r_pos.reshape(-1,1).to(xh.device)
            edge_r_pos = edge_r_pos/n_nodes
            edge_r_pos_emb = self.edge_relative_pos_emb(edge_r_pos)
            edge_attr = torch.cat([edge_sse_attr,edge_r_pos_emb],dim=1)
        else:
            edges = self.get_edges_with_mask(node_mask)
            # batch_mask = node_mask
            # adj= batch_mask[:, None] == batch_mask[None, :]
            # edge_index=torch.stack(torch.where(adj), dim=0)
            # ssecenter= torch.arange(int(len(batch_mask)/3))*3+1
            # sse_idx= [torch.vstack([ssecenter,ssecenter-1]),torch.vstack([ssecenter,ssecenter+1]),torch.vstack([ssecenter-1,ssecenter+1]),torch.vstack([ssecenter-1,ssecenter]),torch.vstack([ssecenter+1,ssecenter]),torch.vstack([ssecenter+1,ssecenter-1])]
            # sse_idx= torch.cat(sse_idx,dim=1)
            # sse_attr= torch.eye(len(adj))
            # sse_attr[sse_idx[0],sse_idx[1]]=1
            # sse_attr = sse_attr[edge_index[0],edge_index[1]].float()

        # Adding time feature
        x = xh[:, :self.n_dims].clone()  # (B*N, 3)
        h = xh[:, self.n_dims:].clone()  # (B*N, nf)
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                #h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
                h_time = torch.empty(x.shape[0],1).fill_(t.item())
            else:
                # t is different over the batch dimension.
                if has_batch_dim:
                    h_time = t.view(bs, 1).repeat(1, n_nodes)
                    h_time = h_time.view(bs * n_nodes, 1)
                else:
                    h_time = t[node_mask]
            h = torch.cat([h, h_time], dim=1)  # (B*N, nf+1)
        if context is not None:
            h = torch.cat([h, context], dim=1)
        
        # Forward EGNN
        # Output: h_final (B*N, nf), x_final (B*N, 3), vel (B*N, 3)
        if self.model == 'egnn_dynamics':
            h_final, x_final = self.dynamics(
                h,
                x,
                edges,
                node_mask=node_mask if has_batch_dim else None,
                edge_mask=edge_mask,
                batch_size = bs,
                edge_attr= edge_attr,
            )
            vel = (x_final - x)
            if has_batch_dim:
                vel = vel * node_mask  # This masking operation is redundant but just in case
        elif self.model == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1).to()
            output = self.dynamics(
                xh, edges, node_mask=node_mask if has_batch_dim else None)
            vel = output[:, 0:3]
            if has_batch_dim:
                vel = vel * node_mask
            h_final = output[:, 3:]
        elif self.model == 'gvp_dynamics':
            gvpinput = (torch.cat([x,h],dim=-1)*node_mask).to(self.device)
            gvp_edges = torch.cat([edges[0].reshape(1,-1), edges[1].reshape(1,-1)],dim=0)
            gvpout= self.dynamics(
                x=gvpinput,edge_index=gvp_edges,edge_attr=(edge_attr*edge_mask).to(self.device),batch=bs 
            )
            h_final = gvpout[:,3:]
            x_final = gvpout[:,:3]
            vel = (x_final - x)
            if has_batch_dim:
                vel = vel * node_mask
        elif self.model == 'gvp_dynamics_new':
            batch_mask = node_mask
            # mask = torch.arange(node_counts.max(), device=self.device) < node_counts.unsqueeze(1)
            edges = self.get_edges_with_mask(batch_mask).to(x.device)
            out= self.dynamics(h,x,edges,v=None,edge_attr=None,batch_mask=batch_mask)
            h_final = out[0]
            vel = out[1]
            has_batch_dim=False
            
                                  
        else:
            raise NotImplementedError
        # Slice off context size
        if context is not None:
            h_final = h_final[:, :-self.context_node_nf]

        # Slice off last dimension which represented time.
        if self.condition_time:
            h_final = h_final[:, :-1]

        if has_batch_dim:
            vel = vel.view(bs, n_nodes, -1)  # (B, N, 3)
            h_final = h_final.view(bs, n_nodes, -1)  # (B, N, D)
       
        # print('xproblem',torch.any(torch.isnan(vel)) ,'hproblem',torch.any(torch.isnan(h_final)))
        # if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
        #     print(vel,h_final)
        #     set_trace()
        # vel = torch.where(torch.isnan(vel), torch.full_like(vel, 0), vel).to(vel.device)
        # h_final = torch.where(torch.isnan(h_final), torch.full_like(h_final, 0), h_final).to(vel.device)
        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final)):
            # import pdb
            # pdb.set_trace()
            raise utils.FoundNaNException(vel, h_final)

        return torch.cat([vel, h_final], dim=-1)

    def get_edges(self, n_nodes, batch_size):
        if n_nodes in self.edge_cache:
            edges_dic_b = self.edge_cache[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self.edge_cache[n_nodes] = {}
            return self.get_edges(n_nodes, batch_size)

    def get_edges_with_mask(self, batch_mask):
        # TODO: cache batches for each example -> speed-up?
        adj = batch_mask[:, None] == batch_mask[None, :]
        edges = torch.stack(torch.where(adj), dim=0)
        return edges
