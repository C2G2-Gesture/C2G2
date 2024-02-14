from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib
import pprint
import sys
import time
from tqdm import tqdm
from data_loader.lmdb_data_loader import *
from model.pose_diffusion import PoseDiffusion
from parse_args_diffusion import parse_args
from train_eval.train_diffusion import train_iter_diffusion,train_iter_vqvae,train_iter_vae, train_iter_vqvae_cond
from utils.average_meter import AverageMeter
from utils.vocab_utils import build_vocab
import utils.train_utils
from model.motion_ae import *
from model.vqvae import VQModel,VAEModel, CondVQModel

matplotlib.use('Agg')  # we don't use interactive GUI
[sys.path.append(i) for i in ['.', '..']]
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


def init_model(args, _device):
    # init model
    if args.model == 'vqvae':
        print("init diffusion model")
        vqvae = VQmodel(args).to(_device)
    return vqvae


def train_epochs(args, train_data_loader, lang_model, pose_dim, weight_path=None, use_cond=True, speaker_model=None):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG'),AverageMeter('recon_loss'),AverageMeter('quant_loss')]

    # tb_path = args.name + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    # tb_writer = SummaryWriter(log_dir=str(Path(args.model_save_path).parent / 'tensorboard_runs' / tb_path))

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_model_epoch_interval = 5
    
    if mode == "vqvae":
        if use_cond:
            assert weight_path!=None
            
            vqvae = CondVQModel(in_channels=27,n_embed=1024,embed_dim=64,hidden_channels=64,num_res_blocks=2,deeper=False,pos_enc=False).to(device)

            g_factor = 0.3
            lr_g = 0.0005
            lr_d = g_factor*lr_g

            optimizer_g = torch.optim.Adam(list(vqvae.decoder.parameters()),
                                        lr=lr_g, betas=(0.5, 0.9))

            optimizer_d = torch.optim.Adam(list(vqvae.loss.discriminator.parameters()),
                                            lr=lr_d, betas=(0.5, 0.9))
            weight = torch.load(weight_path)["state_dict"]
            # vqvae_ori = VQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2,deeper=False,pos_enc=False).to(device)
            model_dict = vqvae.state_dict()
            
            pretrained_dict = {k: v for k, v in weight.items() if (k in model_dict and k[:7]!="decoder")}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            vqvae.load_state_dict(model_dict)
        else:
            vqvae = VQModel(in_channels=27,n_embed=1024,embed_dim=64,hidden_channels=64,num_res_blocks=2).to(device)

        g_factor = 0.1
        lr_g = 0.0005
        lr_d = g_factor*lr_g

        optimizer_g = torch.optim.Adam(list(vqvae.encoder.parameters())+
                                    list(vqvae.decoder.parameters())+
                                    list(vqvae.quantize.parameters())+
                                    list(vqvae.quant_conv.parameters())+
                                    list(vqvae.post_quant_conv.parameters()),
                                    lr=lr_g, betas=(0.5, 0.9))

        optimizer_d = torch.optim.Adam(list(vqvae.loss.discriminator.parameters()),
                                        lr=lr_d, betas=(0.5, 0.9))
        
    else:
        vae = VAEModel(in_channels=27,n_embed=64,embed_dim=64,hidden_channels=64,num_res_blocks=2).to(device)
        lr_g = 0.0005
        optimizer_g = torch.optim.Adam(list(vae.encoder.parameters())+
                                    list(vae.decoder.parameters())+
                                    list(vae.decoder_t.parameters())+
                                    list(vae.encoder_t.parameters())+
                                    list(vae.encoder_mu.parameters())+
                                    list(vae.encoder_log.parameters()),
                                    lr=lr_g, betas=(0.5, 0.9))
        

    # training
    global_iter = 0
    for epoch in range(args.epochs):

        # save model
        if (epoch % save_model_epoch_interval == 0 and epoch > 0) or epoch == args.epochs - 1:
            if mode == "vqvae":
                state_dict = vqvae.state_dict()
            else:
                state_dict = vae.state_dict()

            save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model, 'speaker_model': speaker_model,
                'pose_dim': pose_dim, 'state_dict': state_dict,
            }, save_name)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(tqdm(train_data_loader)):
            global_iter += 1
            _, _, _, _, target_vec, _, _, _, target_nnm_vec = data

            batch_size = target_vec.size(0)
            target_vec = target_vec.to(device)
            target_nnm_vec = target_nnm_vec.to(device)

            # train
            loss = []
            # if args.model == 'pose_diffusion':
            #     loss = train_iter_diffusion(args, in_audio, target_vec, 
            #                           diffusion_model, optimizer)
            if mode == "vqvae":
                loss = train_iter_vqvae_cond(args, target_vec, 
                                      vqvae, [optimizer_g, optimizer_d],target_nnm_vec)
            else:
                loss = train_iter_vae(args, target_vec, 
                                      vae, optimizer_g)
        

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # write to tensorboard
            # for key in loss.keys():
            #     tb_writer.add_scalar(key + '/train', loss[key], global_iter)

            # print training status
            if (iter_idx+1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.5f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    tb_writer.close()


def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    collate_fn = default_collate_fn

    # dataset
    mean_dir_vec = np.array(args.mean_dir_vec).reshape(-1, 3)
    args.batch_size = 256
    train_dataset = SpeechMotionDataset(args.train_data_path[0],
                                        n_poses=args.n_poses,
                                        subdivision_stride=args.subdivision_stride,
                                        pose_resampling_fps=args.motion_resampling_framerate,
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=args.mean_pose,
                                        remove_word_timing=(args.input_context == 'text'),
                                        
                                        
                                        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    val_dataset = SpeechMotionDataset(args.val_data_path[0],
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      speaker_model=train_dataset.speaker_model,
                                      mean_dir_vec=mean_dir_vec,
                                      mean_pose=args.mean_pose,
                                      remove_word_timing=(args.input_context == 'text')
                                      )

    test_dataset = SpeechMotionDataset(args.test_data_path[0],
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate,
                                       speaker_model=train_dataset.speaker_model,
                                       mean_dir_vec=mean_dir_vec,
                                       mean_pose=args.mean_pose)

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path[0])[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path, args.wordembed_path,
                             args.wordembed_dim)
    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train
    pose_dim = args.pose_dim
    train_epochs(args, train_loader, lang_model,
                 pose_dim=pose_dim, speaker_model=train_dataset.speaker_model, use_cond=True, weight_path = args.vqvae_weight)


if __name__ == '__main__':
    _args = parse_args()
    mode = "vqvae"
    main({'args': _args})
