import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pytorch_lightning as pl
import torch
import wandb

import utils
from dynamics import Dynamics
from edm import EDM
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional
from torch_scatter import scatter_sum
from torch.utils.data import Dataset#,DataLoader
from pdb import set_trace
from torch_geometric.data import Batch


def get_activation(activation):
    # print(activation)
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")
class DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention, norm_constant,
        inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        normalize_factors, model,
        data_path, batch_size, lr, test_epochs, n_stability_samples,torch_device='cuda',
        normalization=None, log_iterations=None, samples_dir=None, data_augmentation=False, add_batch_dim=False,pos_embdim=8,reflection_equiv=True,pinder_path=None,train_pinder=False,num_workers=20
    ):
        super(DDPM, self).__init__()
        
        self.save_hyperparameters()
        self.data_path = data_path
        self.train_pinder=train_pinder
        self.pinder_path=pinder_path
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch.device(torch_device)
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.loss_type = diffusion_loss_type
        self.add_batch_dim = add_batch_dim

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf+pos_embdim # pos 1 + pos_emb 8 
        self.training_step_outputs=[]
        self.validation_step_outputs=[]
        if type(activation) is str:
            activation = get_activation(activation)

        egnn = Dynamics(
            in_node_nf=self.in_node_nf ,
            n_dims=n_dims,
            context_node_nf=context_node_nf,
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            reflection_equiv=reflection_equiv
        ).to(self.torch_device)
        self.ddpm = EDM(
            dynamics=egnn,
            in_node_nf=self.in_node_nf ,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
            pos_embdim=pos_embdim,
        ).to(self.torch_device)
        cathdataset= torch.load(self.data_path)# '/work/lpdi/users/ymiao/TopoFlow/filtcathgraphlist_simple.pt'
        cath_train_dataset=cathdataset[0:int(len(cathdataset)*0.95)]
        cath_val_dataset=cathdataset[int(len(cathdataset)*0.95):]
        if self.train_pinder:
            pinder_train_dataset= torch.load(self.pinder_path[0]) #'/work/lpdi/users/ymiao/TopoFlow/pinder_cluster_train_120_12.pt'
            pinder_val_dataset= torch.load(self.pinder_path[1]) #'/work/lpdi/users/ymiao/TopoFlow/pinder_cluster_val.pt'
            self.train_dataset = cath_train_dataset+pinder_train_dataset
            self.val_dataset = pinder_val_dataset+ cath_val_dataset
        else:
            self.train_dataset = cath_train_dataset
            self.val_dataset = cath_val_dataset
        self.num_workers= num_workers


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size= self.batch_size,shuffle=True, num_workers=self.num_workers,pin_memory=True,prefetch_factor=2)
        # return self.train_loader

    def val_dataloader(self):
        return  DataLoader(self.val_dataset, batch_size= self.batch_size,shuffle=True, num_workers=self.num_workers,pin_memory=True,prefetch_factor=2)
        # return self.val_loader


    def forward(self, data, training):
        check_batch_size=True
        if check_batch_size:
            if (data.pos.size(0))>1900:
                print('batch_size too big skip this')
                return None,None,None,None,None,None,None
        if self.data_augmentation:
            unique_batches = data.batch.unique() 
            for b in unique_batches:
                batch_indices = (data.batch == b)
                rotation_matrix = random_rotation_matrix(dim=data.pos.size(1)).to(data.pos.device)
                translation_vector = random_translation_vector(dim=data.pos.size(1)).to(data.pos.device)
                data.pos[batch_indices] = torch.matmul(data.pos[batch_indices], rotation_matrix)+translation_vector

        h = data.x.to(self.torch_device) # is actual context 
        x = data.pos.to(self.torch_device) # have already been processed to substract centre of mass
        batch_mask= data.batch.to(self.torch_device)

        # # Applying random rotation
        # if training and self.data_augmentation:
        #     if not self.add_batch_dim:
        #         raise NotImplementedError
        #     x = utils.random_rotation(x)
        return self.ddpm.forward(
            x=x,
            h=h,
            batch_mask=batch_mask,
        )

    def training_step(self, data, *args):
        
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = \
            self.forward(data, training=True)
        if l2_loss==None:
            return None
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        training_metrics = {
            'loss': loss.detach().cpu(),
            'delta_log_px': delta_log_px.detach().cpu(),
            'kl_prior': kl_prior.detach().cpu(),
            'loss_term_t': loss_term_t.detach().cpu(),
            'loss_term_0': loss_term_0.detach().cpu(),
            'l2_loss': l2_loss.detach().cpu(),
            'vlb_loss': vlb_loss.detach().cpu(),
            'noise_t': noise_t.detach().cpu(),
            'noise_0': noise_0.detach().cpu()
        }
        if torch.isnan(loss).any():
            return None 
        # if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
        for metric_name, metric in training_metrics.items():
            self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
            self.log(f'{metric_name}/train', metric, prog_bar=True)
        self.training_step_outputs.append(training_metrics)
        return loss

    def validation_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        if l2_loss==None:
            return None
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if torch.isnan(vlb_loss).any():
            return None
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        self.validation_step_outputs.append( {
            'loss': loss.detach().cpu(),
            'delta_log_px': delta_log_px.detach().cpu(),
            'kl_prior': kl_prior.detach().cpu(),
            'loss_term_t': loss_term_t.detach().cpu(),
            'loss_term_0': loss_term_0.detach().cpu(),
            'l2_loss': l2_loss.detach().cpu(),
            'vlb_loss': vlb_loss.detach().cpu(),
            'noise_t': noise_t.detach().cpu(),
            'noise_0': noise_0.detach().cpu()
        })
        return loss

    def test_step(self, data, *args):
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(data, training=False)
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)
        return {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

    def on_training_epoch_end(self,):#on_train_epoch_end training_epoch_end
        for metric in self.training_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.training_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/train', []).append(avg_metric)
            self.log(f'{metric}/train', avg_metric, prog_bar=True)


    def on_validation_epoch_end(self,):
        for metric in self.validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            self.log(f'{metric}/val', avg_metric, prog_bar=True)


    def on_test_epoch_end(self, test_step_outputs):
        for metric in test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(test_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/test', []).append(avg_metric)
            self.log(f'{metric}/test', avg_metric, prog_bar=True)

    def sample_chain(self, data, sample_size=False, keep_frames=None,condition_mask=None,gvp=None):
        # template_data = create_templates_for_sketch_generation(data)
        x = data.pos
        h = data.x
        if gvp==True:
            chain,chain0 = self.ddpm.sample_chain_gvp(
                x=x,
                h=h,
                batch_mask=data.batch,
                keep_frames=keep_frames
            )
        elif condition_mask!=None:
            chain,chain0 = self.ddpm.sample_chain_gvp_inpaint(
                x=x,
                h=h,
                batch_mask=data.batch,
                keep_frames=keep_frames,
                condition_mask=condition_mask

            )
        else:
            raise NotImplementedError
        return chain, chain0

    def on_after_backward(self):
        # Check if any gradient has NaN or Inf values
        nan_or_inf_detected = False
        for param in self.ddpm.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    # print(f"NaN gradient detected in")
                    nan_or_inf_detected = True
        if nan_or_inf_detected:
            print("Skipping optimizer step due to NaN/Inf in gradients")
            self.zero_grad(set_to_none=True)
    def configure_optimizers(self):
        return torch.optim.AdamW(self.ddpm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-5)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()

def random_rotation_matrix(dim=3):
    # Generate a random rotation matrix (around z-axis for simplicity)
    theta = torch.rand(1) * 2 * torch.pi  # random rotation angle
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    rotation_matrix = torch.eye(dim)
    rotation_matrix[0, 0] = cos_theta
    rotation_matrix[0, 1] = -sin_theta
    rotation_matrix[1, 0] = sin_theta
    rotation_matrix[1, 1] = cos_theta
    
    return rotation_matrix

def random_translation_vector(dim=3):
    # Generate a random translation vector in 3D
    return torch.rand(dim) * 60  # scale translation as desired