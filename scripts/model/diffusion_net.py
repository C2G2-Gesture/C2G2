import math
import torch
import torch.nn.functional as F
import tqdm
from torch.nn import Module
from .diffusion_util import VarianceSchedule, TransformerModel

class DiffusionNet(Module):

    def __init__(self, net:TransformerModel, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):

        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.contiguous().view(-1, point_dim), e_rand.contiguous().view(-1, point_dim), reduction='mean')

        return loss

    def sample(self, num_pose, context, pose_dim, flexibility=0.0, ret_traj=False, uncondition_embedding=None):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_pose, pose_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in tqdm.tqdm(range(self.var_sched.num_steps, 0, -1)):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]

            if uncondition_embedding is not None:
                x_in = torch.cat([x_t] * 2)
                beta_in = torch.cat([beta] * 2)
                uncond_emb = uncondition_embedding.repeat(x_t.shape[0],1,1)
                context_in = torch.cat([uncond_emb, context])
                e_theta_uncond, e_theta = self.net(x_in, beta=beta_in, context=context_in).chunk(2)
                e_theta = e_theta_uncond + 1.15 * (e_theta - e_theta_uncond)
            else:
                e_theta = self.net(x_t, beta=beta, context=context)

            t0 = 25
            if t < t0 and t > 1:
                sigma_a = 1/t0/t0*(t-t0)*(t-t0)
                z0 = sigma_a * torch.randn_like(x_T[:,0,:].unsqueeze(1))

                res = torch.zeros_like(z)
                for n in range(num_pose):
                    zn = math.sqrt((1-sigma_a*sigma_a)) * torch.randn_like(x_T[:,0,:].unsqueeze(1))
                    res[:,n:n+1,:] = zn + z0
                z = res

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]
        
    def repaint_sample(self, num_pose, context, pose_dim, gt, mask_x0, flexibility=0.0, ret_traj=False, uncondition_embedding=None):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_pose, pose_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in tqdm.tqdm(range(self.var_sched.num_steps, 0, -1)):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
            
            x_t = traj[t]
            
            # use noised gt as condition for last piece generation
            c01 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
            c11 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

            e_rand_gt = torch.randn_like(gt)  # (B, N, d)
            gt_noised = c01 * gt + c11 * e_rand_gt
            
            x_t = mask_x0*gt_noised + (1-mask_x0)*x_t
            

            beta = self.var_sched.betas[[t]*batch_size]

            if uncondition_embedding is not None:
                x_in = torch.cat([x_t] * 2)
                beta_in = torch.cat([beta] * 2)
                uncond_emb = uncondition_embedding.repeat(x_t.shape[0],1,1)
                context_in = torch.cat([uncond_emb, context])
                e_theta_uncond, e_theta = self.net(x_in, beta=beta_in, context=context_in).chunk(2)
                e_theta = e_theta_uncond + 1.15 * (e_theta - e_theta_uncond)
            else:
                e_theta = self.net(x_t, beta=beta, context=context)

            t0 = 25
            if t < t0 and t > 1:
                sigma_a = 1/t0/t0*(t-t0)*(t-t0)
                z0 = sigma_a * torch.randn_like(x_T[:,0,:].unsqueeze(1))

                res = torch.zeros_like(z)
                for n in range(num_pose):
                    zn = math.sqrt((1-sigma_a*sigma_a)) * torch.randn_like(x_T[:,0,:].unsqueeze(1))
                    res[:,n:n+1,:] = zn + z0
                z = res

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]
        
    def ddim_sample(self, num_pose, context, pose_dim, ddim_timesteps = 20, flexibility=0.0, ret_traj=False, uncondition_embedding=None):
        c =  self.var_sched.num_steps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.var_sched.num_steps, c)))
       
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        seq = ddim_timestep_seq + 1
        # previous sequence
        seq_next = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_pose, pose_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        with torch.no_grad():
            seq_next = [-1]+list(seq[:-1])
            for i, j in zip(reversed(seq), reversed(seq_next)):
                beta = self.var_sched.betas[[i]*batch_size]
                beta_1 = self.var_sched.betas[[j]*batch_size]
                
                alpha_bar = self.var_sched.alpha_bars[i]
                alpha_bar_1 = self.var_sched.alpha_bars[j]
                
                beta_t = 1 - alpha_bar / alpha_bar_1
                
                x_t = traj[t]
                
                if uncondition_embedding is not None:
                    x_in = torch.cat([x_t] * 2)
                    beta_in = torch.cat([beta] * 2)
                    uncond_emb = uncondition_embedding.repeat(x_t.shape[0],1,1)
                    context_in = torch.cat([uncond_emb, context])
                    e_theta_uncond, e_theta = self.net(x_in, beta=beta_in, context=context_in).chunk(2)
                    e_theta = e_theta_uncond + 1.15 * (e_theta - e_theta_uncond)
                else:
                    e_theta = self.net(x_t, beta=beta, context=context)
                    
                e = e_theta

                x0_from_e = (1.0 / alpha_bar).sqrt() * x - (1.0 / alpha_bar - 1).sqrt() * e
                x0_from_e = torch.clamp(x0_from_e, -1, 1)
                
                mean_eps = (
                    (alpha_bar_1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - alpha_bar_1)) * x
                ) / (1.0 - alpha_bar)
                
                sigma = beta_t.log()
                
                x_next = mean_eps + sigma * z
                traj[t-1] = x_next.detach()
                traj[t] = traj[t].cpu()
                if not ret_traj:
                    del traj[t]

            if ret_traj:
                return traj
            else:
                return traj[0]
                                
                
                
                
                
                 