import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from model.vqmodule.quantize import QuantizeEMAReset,QuantizeReset,QuantizeEMA,Quantizer

from model.vqmodule.modules import Encoder, Decoder, ConditionalDecoder
from model.vqmodule.modules_conv import Encoder_convd, Decoder_convd

import numpy as np
from model.loss.vqloss import VQL1WithDiscriminator
from torch.distributions import kl_divergence


class VQModel(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_res_blocks,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 deeper = False,
                 pos_enc = False,
                 device = "cuda:0"
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.deeper = deeper
        
        if self.deeper:
            self.encoder = Encoder(ch=hidden_channels,ch_mult=(1,2,2,4),num_res_blocks=num_res_blocks,in_channels=in_channels,z_channels=embed_dim,depth=4,pos_enc=pos_enc)
            self.decoder = Decoder(ch=hidden_channels,ch_mult=(1,2,3),out_ch = in_channels,num_res_blocks=num_res_blocks,z_channels=embed_dim,depth=2,pos_enc=pos_enc)
        else:
            self.encoder = Encoder(ch=hidden_channels,num_res_blocks=num_res_blocks,in_channels=in_channels,depth=4,z_channels=embed_dim)
            self.decoder = Decoder(ch=hidden_channels,out_ch = in_channels,num_res_blocks=num_res_blocks,depth=2,z_channels=embed_dim)
        self.loss = VQL1WithDiscriminator(30000,disc_in_channels = in_channels)
        self.quantize = QuantizeEMAReset(n_embed, embed_dim,device)
        # self.quantize = Quantizer(n_embed, embed_dim)
        self.quant_conv = torch.nn.Conv1d(embed_dim, embed_dim, 1)

        self.post_quant_conv = torch.nn.Conv1d(embed_dim, embed_dim, 1)
        
            

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = h.transpose(1,2)
        h = self.quant_conv(h).transpose(1,2)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = quant.transpose(1,2)
        quant = self.post_quant_conv(quant).transpose(1,2)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, _ = self.encode(input)
        b,t,n = quant.shape
        dec = self.decode(quant)
        dec = dec.reshape(b,t,-1)
        if return_pred_indices:
            return dec, diff
        return dec, diff

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = batch
        xrec, qloss = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss,rec_loss,quant_loss,log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,last_layer=self.get_last_layer(), split="train")

            #self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss, rec_loss, quant_loss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss.item(), x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            print(log_dict_disc)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    
class CondVQModel(VQModel):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_res_blocks,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 deeper = False,
                 pos_enc = False
                 ):
        super().__init__(in_channels,
                 hidden_channels,
                 num_res_blocks,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 deeper = False,
                 pos_enc = False)
        self.decoder = ConditionalDecoder(ch=hidden_channels,ch_mult=(1,2,3),out_ch = in_channels,num_res_blocks=num_res_blocks,z_channels=embed_dim,depth=2,pos_enc=pos_enc)

    def decode(self, quant, cond):
        quant = quant.transpose(1,2)
        quant = self.post_quant_conv(quant).transpose(1,2)
        dec = self.decoder(quant,cond)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, cond, return_pred_indices=False, enc_train = False):
        if enc_train:
            quant, diff, _ = self.encode(input)
            b,t,n = quant.shape
        else:
            with torch.no_grad():
                quant, diff, _ = self.encode(input)
                b,t,n = quant.shape
        dec = self.decode(quant,cond)
        dec = dec.reshape(b,t,-1)
        if return_pred_indices:
            return dec, diff
        return dec, diff

    def training_step(self, batch,nnm_vec, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = batch
        xrec, qloss = self(x, nnm_vec[:,0,:], return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss,rec_loss,quant_loss,log_dict_ae = self.loss(qloss, nnm_vec, xrec, optimizer_idx, self.global_step,last_layer=self.get_last_layer(), split="train")

            #self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss, rec_loss, quant_loss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss.item(), x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            print(log_dict_disc)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight


class VAEModel(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_res_blocks,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.encoder = Encoder(ch=hidden_channels,num_res_blocks=num_res_blocks,in_channels=in_channels,z_channels=embed_dim)
        self.decoder = Decoder(ch=hidden_channels,out_ch = in_channels,num_res_blocks=num_res_blocks,z_channels=embed_dim)
        self.encoder_t = torch.nn.Linear(34,34)
        self.decoder_t = torch.nn.Linear(34,34)
        self.encoder_mu = torch.nn.Linear(embed_dim,embed_dim)
        self.encoder_log = torch.nn.Linear(embed_dim,embed_dim)


        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = h.transpose(1,2)
        h = self.encoder_t(h).transpose(1,2)
        mu = self.encoder_mu(h)
        logvar = self.encoder_log(h) 
        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        
        return mu, logvar

    def decode(self, quant):
        quant = quant.transpose(1,2)
        quant = self.decoder_t(quant).transpose(1,2)
        dec = self.decoder(quant)
        
        return dec
    
    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, return_pred_indices=False):
        mu,logvar = self.encode(input)
        latent = self.reparameterize(mu,logvar)
        dec = self.decode(latent)
        return dec

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = batch
        xrec = self(x)
        L1loss = torch.nn.L1Loss()
        loss = L1loss(x,xrec).mean()
        return loss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight    
    
class VQModel_convd(VQModel):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_res_blocks,
                 n_embed,
                 embed_dim,
                 ch_mult,
                 downsample_rate,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 deeper = False,
                 pos_enc = False
                 ):
        super().__init__(in_channels,
                 hidden_channels,
                 num_res_blocks,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 deeper = False,
                 pos_enc = False)
        self.ch_mult = ch_mult
        
        self.encoder = Encoder_convd(ch=hidden_channels,ch_mult=self.ch_mult,downsample_rate=downsample_rate, num_res_blocks=num_res_blocks,in_channels=in_channels,z_channels=embed_dim,depth=4,pos_enc=pos_enc)
        self.decoder = Decoder_convd(ch=hidden_channels,ch_mult=self.ch_mult,downsample_rate=downsample_rate, out_ch = in_channels,num_res_blocks=num_res_blocks,z_channels=embed_dim,depth=2,pos_enc=pos_enc)

    def decode(self, quant):
        quant = quant.transpose(1,2)
        quant = self.post_quant_conv(quant).transpose(1,2)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False, enc_train = False):
    
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        b,t,n = dec.shape
        dec = dec.reshape(b,t,-1)
        if return_pred_indices:
            return dec, diff
        return dec, diff
    
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
# def main():
#     from torchsummary import summary
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     t = VQModel(in_channels=126,n_embed=1024,embed_dim=128,hidden_channels=128,num_res_blocks=2,deeper=False,pos_enc=False).to(device)
#     summary(t, (34, 126))
    
# main()
   
