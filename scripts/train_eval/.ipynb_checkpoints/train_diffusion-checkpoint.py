import torch
from torch.nn.utils import clip_grad_norm_

def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise


def train_iter_diffusion(args, in_audio, target_poses, pose_diffusion, optimizer):

    # make pre seq input
    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    optimizer.zero_grad()
    pose_diffusion.train()
    #print(f'input poses: {target_poses.shape}, pre_seq: {pre_seq.shape}, in_audio: {in_audio.shape}')

    loss,recon_loss = pose_diffusion.get_loss(target_poses, pre_seq, in_audio)
    total_loss = loss+recon_loss
    total_loss.backward()
    clip_grad_norm_(pose_diffusion.parameters(), 10)
    optimizer.step()

    ret_dict = {'loss':loss.item(), 'recon_loss':recon_loss.item()}
    return ret_dict



def train_iter_ae(args, target_poses, pose_ae, optimizer):

    # make pre seq input
    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    l1loss = torch.nn.L1Loss()

    optimizer.zero_grad()
    pose_ae.train()
    #print(f'input poses: {target_poses.shape}, pre_seq: {pre_seq.shape}, in_audio: {in_audio.shape}')

    output,_ = pose_ae(target_poses)
    loss = l1loss(output,target_poses).mean()
    total_loss = loss
    total_loss.backward()
    clip_grad_norm_(pose_ae.parameters(), 10)
    optimizer.step()

    ret_dict = {'loss':loss.item()}
    return ret_dict



def train_iter_vqvae(args, target_poses, pose_vqvae, optimizer):
    optimizer_g, optimizer_d = optimizer

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()

    pose_vqvae.train()
    #print(f'input poses: {target_poses.shape}, pre_seq: {pre_seq.shape}, in_audio: {in_audio.shape}')

    loss,rec_loss,quant_loss = pose_vqvae.training_step(target_poses,0,0)
    total_loss = loss
    total_loss.backward()

    optimizer_g.step()
    optimizer_d.step()

    ret_dict = {'loss':loss.item(),'recon_loss':rec_loss.item(),'quant_loss':quant_loss.item()}
    return ret_dict


def train_iter_vae(args, target_poses, pose_vqvae, optimizer):
    optimizer_g = optimizer

    optimizer_g.zero_grad()

    pose_vqvae.train()
    #print(f'input poses: {target_poses.shape}, pre_seq: {pre_seq.shape}, in_audio: {in_audio.shape}')

    loss = pose_vqvae.training_step(target_poses,0,0)
    total_loss = loss
    total_loss.backward()

    optimizer_g.step()

    ret_dict = {'loss':loss.item()}
    return ret_dict
