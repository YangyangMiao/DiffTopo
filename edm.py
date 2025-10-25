# 10.19 version add sse type to output 
import torch
import torch.nn.functional as F
import numpy as np
import math

import utils
from dynamics import Dynamics,SinusoidalPosEmb
from noise import GammaNetwork, PredefinedNoiseSchedule
from typing import Union
from torch_scatter import scatter_add
# import torch_geometric.utils.to_dense_batch as to_dense_batch
from pdb import set_trace
import json




class EDM(torch.nn.Module):
    def __init__(
            self,
            dynamics: Union[Dynamics],
            in_node_nf: int,
            n_dims: int,
            timesteps: int = 1000,
            noise_schedule='learned',
            noise_precision=1e-4,
            loss_type='vlb',
            norm_values=(1., 1.),
            norm_biases=(None, 0.),
            pos_embdim=8,
    ):
        super().__init__()
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned with a vlb objective'
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)
        self.pos_embdim= pos_embdim
        self.pos_embedding = SinusoidalPosEmb(pos_embdim)
        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases

    def forward(self, x, h, batch_mask):
        # Normalization and concatenation
        x, h = self.normalize(x, h)
        # 
        # add positional encoding
        node_feat_dim = h.shape[-1]

        # determine type of data representation
        has_batch_dim = len(x.size()) > 2
        context_h = h#[:,:,:node_feat_dim]
        # pos_h = h[:,:,node_feat_dim:]
        xh = torch.cat([x, h], dim=-1)
        # TODO: we could store this in a variable to avoid torch.unique()
        batch_size = x.size(0) if has_batch_dim \
            else len(torch.unique(batch_mask))

        # Volume change loss term
        num_motif_atoms = scatter_add(torch.ones_like(batch_mask), batch_mask, dim=0)
        delta_log_px = self.delta_log_px(num_motif_atoms).mean()

        # Sample t
        t_int = torch.randint(0, self.T + 1, size=(batch_size, 1), device=x.device).float()
        s_int = t_int - 1
        t = t_int / self.T
        s = s_int / self.T

        # Masks for t=0 and t>0
        t_is_zero = (t_int == 0).squeeze().float()
        t_is_not_zero = 1 - t_is_zero
        # Compute gamma_t and gamma_s according to the noise schedule
        gamma_t = self.inflate_batch_array(self.gamma(t), x)
        gamma_s = self.inflate_batch_array(self.gamma(s), x)

        # Compute alpha_t and sigma_t from gamma
        alpha_t = self.alpha(gamma_t, x).to(x.device)
        sigma_t = self.sigma(gamma_t, x).to(x.device)

        # broadcast per-batch values
        if not has_batch_dim:
            alpha_t = alpha_t[batch_mask]
            sigma_t = sigma_t[batch_mask]

        # Sample noise
        # Note: only for motif
        sample_dim = (x.size(0), x.size(1)) if has_batch_dim else (x.size(0),)
        eps_t = torch.randn((*sample_dim,xh.shape[1])).to(x.device)
        # eps_t = self.sample_combined_position_feature_noise(
        #     sample_dim, mask=node_mask)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # Note: keep interface unchanged
        z_t = alpha_t * xh + sigma_t * eps_t
        # z_t = xh * interface_mask + z_t * motif_mask
        
        # change my code to batch_mask input
        # node_mask=node_mask.bool().reshape(bs,36)
        # batch_mask=[]
        # for i in range(0,bs): batch_mask.append( [i]*node_mask.sum(dim=1)[i])
        # batch_mask= torch.cat([torch.tensor(i) for i in batch_mask].to(self.device)

        # Neural net prediction

        eps_t_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            node_mask=batch_mask,
            edge_mask=None,
            context=context_h,
        )
        if len(eps_t_hat.size())==2:       
            has_batch_dim=False
            error_t = self.sum_except_batch((eps_t - eps_t_hat ) ** 2, batch_mask)
            normalization = (self.n_dims + self.in_node_nf) * num_motif_atoms # 
            # set_trace()
            l2_loss = error_t / normalization
            l2_loss = l2_loss.mean()

            # The KL between q(z_T | x) and p(z_T) = Normal(0, 1) (should be close to zero)
            kl_prior = self.kl_prior(xh, batch_mask).mean()
            # Computing NLL middle term
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze().to(xh.device)
            loss_term_t = self.T * 0.5 * SNR_weight * error_t
            loss_term_t = (loss_term_t * t_is_not_zero).sum() / t_is_not_zero.sum()

            # Computing noise returned by dynamics
            if has_batch_dim:
                noise = torch.sum(eps_t_hat**2, dim=[1, 2])**0.5
            else:
                noise = scatter_add(torch.sum(eps_t_hat ** 2, -1),
                                    batch_mask, dim=0) ** 0.5
            noise_t = (noise * t_is_not_zero).sum() / t_is_not_zero.sum()

            if t_is_zero.sum() > 0:
                # The _constants_ depending on sigma_0 from the
                # cross entropy term E_q(z0 | x) [log p(x | z0)]
                neg_log_constants = -self.log_constant_of_p_x_given_z0(
                    x, num_motif_atoms, batch_size)

                # Computes the L_0 term (even if gamma_t is not actually gamma_0)
                # and selected only relevant via masking
                
                # only calculate things related to x  not about h since h is condition
                # loss_term_0 = 0.
                # h = h.reshape(bs*36,-1)[node_mask]
                # z_t = z_t.reshape(bs*36,-1)[node_mask]
                gamma_t = self.gamma(t)
                loss_term_0 = -self.log_p_xh_given_z0_without_constants(
                    h, z_t, gamma_t, eps_t, eps_t_hat,  batch_mask)
                loss_term_0 = loss_term_0 + neg_log_constants
                loss_term_0 = (loss_term_0 * t_is_zero).sum() / t_is_zero.sum()

                # Computing noise returned by dynamics
                noise_0 = (noise * t_is_zero).sum() / t_is_zero.sum()
                loss_term_0 = loss_term_0
                noise_0 = noise_0
            else:
                loss_term_0 =torch.tensor( 0.).to(l2_loss.device)
                noise_0 = torch.tensor( 0.).to(l2_loss.device)
        return delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0
        # else:
        #     eps_t_hat = eps_t_hat * node_mask    

        #     # Computing basic error (further used for computing NLL and L2-loss)
        #     error_t = self.sum_except_batch((eps_t - eps_t_hat ) ** 2, node_mask)
        #     # error_t = self.sum_except_batch((eps_t[:,:,0:3] - eps_t_hat[:,:,0:3] ) ** 2, node_mask)

        #     # Computing L2-loss for t>0
        #     normalization = (self.n_dims + self.in_node_nf) * num_motif_atoms # 
        #     # set_trace()
        #     l2_loss = error_t / normalization
        #     l2_loss = l2_loss.mean()

        #     # The KL between q(z_T | x) and p(z_T) = Normal(0, 1) (should be close to zero)
        #     kl_prior = self.kl_prior(xh, node_mask).mean()

        #     # Computing NLL middle term
        #     SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze().to(xh.device)
        #     loss_term_t = self.T * 0.5 * SNR_weight * error_t
        #     loss_term_t = (loss_term_t * t_is_not_zero).sum() / t_is_not_zero.sum()

        #     # Computing noise returned by dynamics
        #     if has_batch_dim:
        #         noise = torch.sum(eps_t_hat**2, dim=[1, 2])**0.5
        #     else:
        #         noise = scatter_add(torch.sum(eps_t_hat ** 2, -1),
        #                             node_mask, dim=0) ** 0.5
        #     noise_t = (noise * t_is_not_zero).sum() / t_is_not_zero.sum()

        #     if t_is_zero.sum() > 0:
        #         # The _constants_ depending on sigma_0 from the
        #         # cross entropy term E_q(z0 | x) [log p(x | z0)]
        #         neg_log_constants = -self.log_constant_of_p_x_given_z0(
        #             x, num_motif_atoms, batch_size)

        #         # Computes the L_0 term (even if gamma_t is not actually gamma_0)
        #         # and selected only relevant via masking
                
        #         # only calculate things related to x  not about h since h is condition
        #         # loss_term_0 = 0.
        #         loss_term_0 = -self.log_p_xh_given_z0_without_constants(
        #             h, z_t, gamma_t, eps_t, eps_t_hat,  node_mask)
        #         loss_term_0 = loss_term_0 + neg_log_constants
        #         loss_term_0 = (loss_term_0 * t_is_zero).sum() / t_is_zero.sum()

        #         # Computing noise returned by dynamics
        #         noise_0 = (noise * t_is_zero).sum() / t_is_zero.sum()
        #     else:
        #         loss_term_0 = 0.
        #         noise_0 = 0.

        # return delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0

    @torch.no_grad()
    def sample_chain(self, x, h, node_mask, edge_mask,  keep_frames=None):
        has_batch_dim = len(x.size()) > 2
        sample_dim = (x.size(0), x.size(1)) if has_batch_dim else (x.size(0),)
        n_samples = x.size(0) if has_batch_dim \
            else len(torch.unique(node_mask))
        
        pos = torch.arange(h.shape[1]).repeat(h.shape[0]).reshape(-1).to(h.device)
        pos = pos/h.shape[1]
        # pos = torch.arange(total_pos_num).to(h.device)
        pos_emb = self.pos_embedding(pos)
        pos_emb = pos_emb.reshape(h.shape[0],h.shape[1],-1)
        pos = pos.reshape(h.shape[0],h.shape[1],-1)
        
        # Normalization and concatenation
        un_norm_h = h
        x, h, = self.normalize(x, h)
        h = torch.cat([h,pos_emb],dim=2)
        #xh = torch.cat([x, h], dim=-1)
        xh = torch.cat([x, h], dim=-1)
        # Initial motif sampling from N(0, I)
        z = torch.randn(sample_dim)#self.sample_combined_position_feature_noise(sample_dim, node_mask)
        z = z * node_mask
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + (xh.size()[0],)+(xh.size()[1],)+(xh.size()[2]+self.in_node_nf,), device=z.device)

        # Sample p(z_s | z_t)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                z_t=z,
                node_mask=node_mask,
                edge_mask=edge_mask,
                context= h,
            )
            write_index = (s * keep_frames) // self.T
            # set_trace()
            chain[write_index] = torch.cat([self.unnormalize_z(z),un_norm_h,pos_emb],dim=-1)

        # Finally sample p(x, h | z_0)
        
        x = self.sample_p_xh_given_z0(   #, h
            z_0=z,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context= h,
        )
        # set_trace()
        # chain[0] = torch.cat([x, un_norm_h], dim=-1)
        chain0=torch.cat([x, un_norm_h], dim=-1)
        return chain,chain0
    @torch.no_grad()
    def sample_chain_gvp(self, x, h, batch_mask, keep_frames=None):
        # Calculate position embeddings correctly for each batch
        unique_batches = batch_mask.unique()
        pos_list = []
        for i in unique_batches:
            batch_size = (batch_mask == i).sum()
            pos = torch.arange(batch_size, device=h.device) / batch_size
            pos_list.append(pos)
        pos = torch.cat(pos_list).to(h.device)
        pos_emb = self.pos_embedding(pos)

        # Normalization and concatenation
        un_norm_h = h
        h = torch.cat([h, pos_emb], dim=-1)
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=-1)

        # Initialize noise for sparse batch
        n_samples = len(unique_batches)
        sample_dim = (x.size(0),)  # Sparse dimension
        z = torch.cat([
            torch.randn((*sample_dim, self.n_dims)),
            torch.randn((*sample_dim, self.in_node_nf))
        ], dim=-1).to(x.device)

        # Setup chain storage
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + (xh.size(0),) + (xh.size(1)+self.in_node_nf,), device=z.device)

        # Sampling loop
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                z_t=z,
                batch_mask=batch_mask,
                context=h,
            )
            write_index = (s * keep_frames) // self.T
            chain[write_index] = torch.cat([self.unnormalize_z(z), un_norm_h, pos_emb], dim=-1)

        # Final sampling
        x = self.sample_p_xh_given_z0(
            z_0=z,
            batch_mask=batch_mask,
            context=h,
        )
        chain0 = torch.cat([x, un_norm_h], dim=-1)
        return chain, chain0
    @torch.no_grad()
    def sample_chain_with_condition(self, x, h, node_mask, edge_mask,condition_mask,  keep_frames=None):
        has_batch_dim = len(x.size()) > 2
        sample_dim = (x.size(0), x.size(1)) if has_batch_dim else (x.size(0),)
        n_samples = x.size(0) if has_batch_dim \
            else len(torch.unique(node_mask))
        
        pos = torch.arange(h.shape[1]).repeat(h.shape[0]).reshape(-1).to(h.device)
        pos = pos/h.shape[1]
        # pos = torch.arange(total_pos_num).to(h.device)
        pos_emb = self.pos_embedding(pos)
        pos_emb = pos_emb.reshape(h.shape[0],h.shape[1],-1)
        pos = pos.reshape(h.shape[0],h.shape[1],-1)
        
        # Normalization and concatenation
        un_norm_h = h
        un_norm_condition_x = x * condition_mask
        x, h, = self.normalize(x, h)
        h = torch.cat([h,pos_emb],dim=2)
        #xh = torch.cat([x, h], dim=-1)
        xh = torch.cat([x, h], dim=-1)
        condition_x = x * condition_mask
        # Initial motif sampling from N(0, I)
        z = self.sample_combined_position_feature_noise(sample_dim, node_mask)
        z = z * node_mask
        # set_trace()
        z_h= z[:,:,3:]
        z_x = z[:,:,0:3]*(1-condition_mask)+condition_x
        z = torch.cat([z_x, z_h], dim=-1)
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + (xh.size()[0],)+(xh.size()[1],)+(xh.size()[2]+self.in_node_nf,), device=z.device)

        # Sample p(z_s | z_t)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s=s_array,
                t=t_array,
                z_t=z,
                node_mask=node_mask,
                edge_mask=edge_mask,
                context= h,
            )
            z_h= z[:,:,3:]
            z_x = z[:,:,0:3]*(1-condition_mask)+condition_x
            z = torch.cat([z_x, z_h], dim=-1)
            # z = z*(1-condition_mask)+condition_x
            write_index = (s * keep_frames) // self.T
            # set_trace()
            chain[write_index] = torch.cat([self.unnormalize_z(z),un_norm_h,pos_emb],dim=-1)

        # Finally sample p(x, h | z_0)
        
        x = self.sample_p_xh_given_z0(   #, h
            z_0=z,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context= h,
        )
        x = x*(1-condition_mask)+un_norm_condition_x
        # set_trace()
        # chain[0] = torch.cat([x, un_norm_h], dim=-1)
        chain0=torch.cat([x, un_norm_h], dim=-1)
        return chain,chain0
    @torch.no_grad()
    def sample_chain_gvp_inpaint(self, x, h, batch_mask,gt_keep_mask=None,interface_mask=None,keep_frames=None):
        #get idea from RePaint https://github.com/andreas128/RePaint/blob/main/guided_diffusion/gaussian_diffusion.py
        #gt_keep_mask 1 to keep 0 to inpaint
        # Calculate position embeddings correctly for each batch
        unique_batches = batch_mask.unique()
        pos_list = []
        for i in unique_batches:
            batch_size = (batch_mask == i).sum()
            pos = torch.arange(batch_size, device=h.device) / batch_size
            pos_list.append(pos)
        pos = torch.cat(pos_list).to(h.device)
        pos_emb = self.pos_embedding(pos)

        # Normalization and concatenation
        un_norm_h = h
        h = torch.cat([h, pos_emb], dim=-1)
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=-1)

        # Initialize noise for sparse batch
        n_samples = len(unique_batches)
        sample_dim = (x.size(0),)  # Sparse dimension
        z = torch.cat([
            torch.randn((*sample_dim, self.n_dims)),
            torch.randn((*sample_dim, self.in_node_nf))
        ], dim=-1).to(x.device)
        # Setup chain storage
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + (xh.size(0),) + (xh.size(1)+self.in_node_nf,), device=z.device)

        # Sampling loop
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            z = self.sample_p_zs_given_zt_inpaint(
                s=s_array,
                t=t_array,
                z_t=z,
                batch_mask=batch_mask,
                context=h,
                gt= xh,
                gt_keep_mask=gt_keep_mask,
                interface_mask=interface_mask,
            )
            write_index = (s * keep_frames) // self.T
            chain[write_index] = torch.cat([self.unnormalize_z(z), un_norm_h, pos_emb], dim=-1)

        # Final sampling
        x = self.sample_p_xh_given_z0_inpaint(
            z_0=z,
            batch_mask=batch_mask,
            context=h,
            gt= xh,
            gt_keep_mask=gt_keep_mask,
        )
        chain0 = torch.cat([x, un_norm_h], dim=-1)
        return chain, chain0
    def sample_p_zs_given_zt(self, s, t, z_t, batch_mask, context):
        """Samples from zs ~ p(zs | zt) for sparse batch."""
        # Generate gammas
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        
        # Map to sparse representation
        gamma_s = gamma_s[batch_mask]
        gamma_t = gamma_t[batch_mask]
        
        # Calculate diffusion parameters
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(
            gamma_t, gamma_s, z_t
        )
        sigma_s = self.sigma(gamma_s, target_tensor=z_t).to(z_t.device)
        sigma_t = self.sigma(gamma_t, target_tensor=z_t).to(z_t.device)
        
        # Get model prediction
        eps_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            node_mask=batch_mask,
            edge_mask=None,
            context=context,
        )
        
        # Calculate mean and variance
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample
        noise = torch.cat([
            utils.sample_gaussian(size=(mu.size(0), self.n_dims), device=z_t.device),
            utils.sample_gaussian(size=(mu.size(0), self.in_node_nf), device=z_t.device)
        ], dim=-1)
        z_s = mu + sigma * noise

        # Remove mean correctly for sparse batch
        N = scatter_add(torch.ones_like(batch_mask), batch_mask, dim=0)
        mean = scatter_add(z_s[..., :self.n_dims], batch_mask, dim=0)
        mean = mean / N.unsqueeze(-1)
        z_s[..., :self.n_dims] = z_s[..., :self.n_dims] - mean[batch_mask]

        return z_s

    def sample_p_xh_given_z0(self, z_0, batch_mask, context):
        """Samples x ~ p(x|z0) for sparse batch."""
        zeros = torch.zeros(size=(z_0.size(0), 1), device=z_0.device)
        gamma_0 = self.gamma(zeros).to(z_0.device)
        
        # Compute sigma
        sigma_x = self.SNR(-0.5 * gamma_0).to(z_0.device)
        
        # Get model prediction
        eps_hat = self.dynamics.forward(
            t=zeros,
            xh=z_0,
            node_mask=batch_mask,
            edge_mask=None,
            context=context,
        )

        # Generate samples
        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)
        noise = torch.cat([
            utils.sample_gaussian(size=(mu_x.size(0), self.n_dims), device=z_0.device),
            utils.sample_gaussian(size=(mu_x.size(0), self.in_node_nf), device=z_0.device)
        ], dim=-1)
        xh = mu_x + sigma_x * noise

        # Split and unnormalize
        x, h = xh[..., :self.n_dims], xh[..., self.n_dims:]
        x, h = self.unnormalize(x, h)

        return x
    # def sample_p_zs_given_zt_inpaint(self, s, t, z_t, batch_mask, context,gt,gt_keep_mask):
    #     """Samples from zs ~ p(zs | zt) for sparse batch."""
    #     # Generate gammas
    #     gamma_t = self.gamma(t)
    #     gamma_s = self.gamma(s)
        
    #     # Map to sparse representation
    #     gamma_s = gamma_s[batch_mask]
    #     gamma_t = gamma_t[batch_mask]
        
    #     # Calculate diffusion parameters
    #     sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(
    #         gamma_t, gamma_s, z_t
    #     )
    #     ## add gt and gt_keep_mask
    #     sample_dim = (gt.size(0),)
    #     eps_t = torch.randn((*sample_dim,gt.shape[1])).to(gt.device)
    #     weighed_gt = alpha_t_given_s * gt + sigma_t_given_s * eps_t
    #     z_t = (gt_keep_mask*weighed_gt)+(1-gt_keep_mask)*z_t
    #     ##

    #     sigma_s = self.sigma(gamma_s, target_tensor=z_t).to(z_t.device)
    #     sigma_t = self.sigma(gamma_t, target_tensor=z_t).to(z_t.device)
        
    #     # Get model prediction
    #     eps_hat = self.dynamics.forward(
    #         xh=z_t,
    #         t=t,
    #         node_mask=batch_mask,
    #         edge_mask=None,
    #         context=context,
    #     )
        
    #     # Calculate mean and variance
    #     mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat
    #     sigma = sigma_t_given_s * sigma_s / sigma_t

    #     # Sample
    #     noise = torch.cat([
    #         utils.sample_gaussian(size=(mu.size(0), self.n_dims), device=z_t.device),
    #         utils.sample_gaussian(size=(mu.size(0), self.in_node_nf), device=z_t.device)
    #     ], dim=-1)
    #     z_s = mu + sigma * noise
    #     # Remove mean correctly for sparse batch
    #     N = scatter_add(torch.ones_like(batch_mask), batch_mask, dim=0)
    #     mean = scatter_add(z_s[..., :self.n_dims], batch_mask, dim=0)
    #     mean = mean / N.unsqueeze(-1)
    #     z_s[..., :self.n_dims] = z_s[..., :self.n_dims] - mean[batch_mask]

    #     return z_s
    # 
    def sample_p_zs_given_zt_inpaint(self, s, t, z_t, batch_mask, context, gt, gt_keep_mask,interface_mask):
        """Samples from zs ~ p(zs | zt) for sparse batch."""
        # Generate gammas
        gamma_t = self.gamma(t)
        gamma_s = self.gamma(s)
        
        # Map to sparse representation
        gamma_s = gamma_s[batch_mask]
        gamma_t = gamma_t[batch_mask]
        
        # Calculate diffusion parameters
        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(
            gamma_t, gamma_s, z_t
        )
        
        # Add gt and gt_keep_mask
        sample_dim = (gt.size(0),)
        eps_t = torch.randn((*sample_dim, gt.shape[1])).to(gt.device)
        weighed_gt = alpha_t_given_s * gt + sigma_t_given_s * eps_t
        z_t = (gt_keep_mask * weighed_gt) + (1 - gt_keep_mask) * z_t
        
        # Center the fixed part (gt) to the origin
        # import pdb
        # pdb.set_trace()
        centroid = scatter_add(gt[..., :self.n_dims] * interface_mask[..., :self.n_dims], batch_mask, dim=0)
        centroid = centroid / scatter_add(interface_mask[..., :self.n_dims], batch_mask, dim=0)
        centroid= centroid+0.25*torch.randn_like(centroid).to(centroid.device)
        gt_centered = gt.clone()
        gt_centered[..., :self.n_dims] = gt[..., :self.n_dims] - centroid[batch_mask]
        
        # Update z_t with centered fixed part
        z_t[..., :self.n_dims] = (gt_keep_mask[..., :self.n_dims] * gt_centered[..., :self.n_dims]) + \
                                (1 - gt_keep_mask[..., :self.n_dims]) * z_t[..., :self.n_dims]
        
        sigma_s = self.sigma(gamma_s, target_tensor=z_t).to(z_t.device)
        sigma_t = self.sigma(gamma_t, target_tensor=z_t).to(z_t.device)
        # Get model prediction
        eps_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            node_mask=batch_mask,
            edge_mask=None,
            context=context,
        )
        
        # Calculate mean and variance
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat
        sigma = sigma_t_given_s * sigma_s / sigma_t
        
        # Sample
        noise = torch.cat([
            utils.sample_gaussian(size=(mu.size(0), self.n_dims), device=z_t.device),
            utils.sample_gaussian(size=(mu.size(0), self.in_node_nf), device=z_t.device)
        ], dim=-1)
        z_s = mu + sigma * noise
        
        # Remove mean correctly for sparse batch
        N = scatter_add(torch.ones_like(batch_mask), batch_mask, dim=0)
        mean = scatter_add(z_s[..., :self.n_dims], batch_mask, dim=0)
        mean = mean / N.unsqueeze(-1)
        z_s[..., :self.n_dims] = z_s[..., :self.n_dims] - mean[batch_mask]
        
        # Translate the generated part back to the original position
        z_s[..., :self.n_dims] = z_s[..., :self.n_dims] + centroid[batch_mask]
        
        return z_s
    def sample_p_xh_given_z0_inpaint(self, z_0, batch_mask, context,gt,gt_keep_mask):
        """Samples x ~ p(x|z0) for sparse batch."""
        zeros = torch.zeros(size=(z_0.size(0), 1), device=z_0.device)
        gamma_0 = self.gamma(zeros).to(z_0.device)
        
        # Compute sigma
        sigma_x = self.SNR(-0.5 * gamma_0).to(z_0.device)
        ## add gt and gt_keep_mask
        z_0 = gt_keep_mask*gt+(1-gt_keep_mask)*z_0
        ##
        # Get model prediction
        eps_hat = self.dynamics.forward(
            t=zeros,
            xh=z_0,
            node_mask=batch_mask,
            edge_mask=None,
            context=context,
        )

        # Generate samples
        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)
        noise = torch.cat([
            utils.sample_gaussian(size=(mu_x.size(0), self.n_dims), device=z_0.device),
            utils.sample_gaussian(size=(mu_x.size(0), self.in_node_nf), device=z_0.device)
        ], dim=-1)
        xh = mu_x + sigma_x * noise

        # Split and unnormalize
        x, h = xh[..., :self.n_dims], xh[..., self.n_dims:]
        x, h = self.unnormalize(x, h)

        return x
    def compute_x_pred(self, eps_t, z_t, gamma_t):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.sigma(gamma_t, target_tensor=eps_t).to(z_t.device)
        alpha_t = self.alpha(gamma_t, target_tensor=eps_t).to(z_t.device)
        x_pred = 1. / alpha_t * (z_t - sigma_t * eps_t)
        return x_pred

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def kl_prior(self, xh, node_mask):
        # for our case since we have nothing in node feature we just dont calculate  kl _h 
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones).to(xh.device)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[..., :self.n_dims], mu_T[..., self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = self.gaussian_kl(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d=len(node_mask)*self.n_dims
        kl_distance_x = self.gaussian_kl_for_dimension(mu_T_x, sigma_T_x, zeros, ones,subspace_d,node_mask)
        
        if torch.isnan(kl_distance_x).any() or torch.isnan(kl_distance_h).any():
            import pdb
            pdb.set_trace()
        return kl_distance_x + kl_distance_h


    def log_constant_of_p_x_given_z0(self, x, num_nodes, batch_size):

        # TODO: Shouldn't this be: degrees_of_freedom_x = num_nodes * self.n_dims ?
        degrees_of_freedom_x = num_nodes
        zeros = torch.zeros((batch_size, 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0)
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_p_xh_given_z0_without_constants(self, h, z_0, gamma_0, eps, eps_hat, batch_mask, epsilon=1e-10):
        has_batch_dim = len(h.size()) > 2

        # Discrete properties are predicted directly from z_0
        z_h = z_0[..., self.n_dims:]

        # Take only part over x
        eps_x = eps[..., :self.n_dims]
        eps_hat_x = eps_hat[..., :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0) * self.norm_values[1]

        if not has_batch_dim:
            sigma_0 = sigma_0[batch_mask]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'
        log_p_x_given_z_without_constants = -0.5 * self.sum_except_batch((eps_x - eps_hat_x) ** 2, batch_mask)

        # Categorical features
        # Compute delta indicator masks
        h = h * self.norm_values[1] + self.norm_biases[1]
        estimated_h = z_h * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded
        centered_h = estimated_h - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=centered_h_cat, stdev=sigma_0_cat)
        log_p_h_proportional = torch.log(
            self.cdf_standard_gaussian((centered_h + 0.5) / sigma_0) -
            self.cdf_standard_gaussian((centered_h - 0.5) / sigma_0) +
            epsilon
        )

        # Normalize the distribution over the categories
        log_Z = torch.logsumexp(log_p_h_proportional, dim=-1, keepdim=True)
        log_probabilities = log_p_h_proportional - log_Z

        # Select the log_prob of the current category using the onehot representation
        log_p_h_given_z = self.sum_except_batch(log_probabilities * h , batch_mask)

        # Combine log probabilities for x and h
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def sample_combined_position_feature_noise(self, sample_dim, mask):
        z_x = utils.sample_gaussian_with_mask(
            size=(*sample_dim, self.n_dims),
            device=mask.device,
            node_mask=mask
        )
        z_h = utils.sample_gaussian_with_mask(
            size=(*sample_dim, self.in_node_nf),
            device=mask.device,
            node_mask=mask
        )
        z = torch.cat([z_x, z_h], dim=-1)
        return z

    def sample_normal(self, mu, sigma, node_mask):
        """Samples from a Normal distribution."""
        has_batch_dim = len(mu.size()) > 2
        sample_dim = (mu.size(0), mu.size(1)) if has_batch_dim else (mu.size(0),)
        eps = self.sample_combined_position_feature_noise(sample_dim, node_mask)
        return mu + sigma * eps

    def normalize(self, x, h):
        new_x = x / self.norm_values[0]
        new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        return new_x, new_h

    def unnormalize(self, x, h):
        new_x = x * self.norm_values[0]
        new_h = h * self.norm_values[1] + self.norm_biases[1]
        return new_x, new_h

    def unnormalize_z(self, z):
        assert z.size(-1) == self.n_dims + self.in_node_nf
        x, h = z[..., :self.n_dims], z[..., self.n_dims:]
        x, h = self.unnormalize(x, h)
        return torch.cat([x, h], dim=-1)

    def delta_log_px(self, num_nodes):
        return -self.dimensionality(num_nodes) * np.log(self.norm_values[0])

    def dimensionality(self, num_nodes):
        return num_nodes * self.n_dims

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor).to(target_tensor.device)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor).to(target_tensor.device)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -self.expm1(self.softplus(gamma_s) - self.softplus(gamma_t)),
            target_tensor
        ).to(target_tensor.device)

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s.to(target_tensor.device), sigma_t_given_s.to(target_tensor.device), alpha_t_given_s.to(target_tensor.device)

    @staticmethod
    def numbers_of_nodes(binary_mask, has_batch_dim, batch_mask):
        if has_batch_dim:
            return torch.sum(binary_mask.squeeze(2), dim=1)#
        else:
            return scatter_add(binary_mask, batch_mask, dim=0)

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    @staticmethod
    def sum_except_batch(x, batch_mask=None):
        if len(x.size()) > 2:
            return x.view(x.size(0), -1).sum(-1)
        else:
            return scatter_add(x.sum(-1), batch_mask, dim=0)

    @staticmethod
    def expm1(x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def gaussian_kl(q_mu, q_sigma, p_mu, p_sigma, node_mask):
        """
        Computes the KL distance between two normal distributions.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
        kl = torch.log(p_sigma / q_sigma) + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2) - 0.5
        return EDM.sum_except_batch(kl, node_mask)

    @staticmethod
    def gaussian_kl_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d, node_mask):
        """
        Computes the KL distance between two normal distributions taking the dimension into account.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
            d: dimension
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
        mu_norm_2 = EDM.sum_except_batch((q_mu - p_mu) ** 2, node_mask)
        assert len(q_sigma.size()) == 1
        assert len(p_sigma.size()) == 1
        q_sigma=EDM.sum_except_batch(q_sigma.unsqueeze(1),node_mask)
        p_sigma=EDM.sum_except_batch(p_sigma.unsqueeze(1),node_mask)
        return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma ** 2 + mu_norm_2) / (p_sigma ** 2) - 0.5 * d

def sum_except_batch(x):
        return x.view(x.size(0), -1).sum(-1)

def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )
def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)

    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d
    