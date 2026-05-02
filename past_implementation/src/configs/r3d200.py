from ._base import cfg
from types import SimpleNamespace

# Dataloader
cfg.batch_size= 12
cfg.num_workers= 6

# Dataset
cfg.dataset_type= "_3d"
cfg.data_dir= "./data/processed/"

# Model
cfg.model_type = "unet3d"
cfg.backbone = "r3d200"
cfg.roi_size= (64, 672, 672)
cfg.in_chans= 1
cfg.seg_classes= 1
cfg.deep_supervision= True

# Encoder
encoder_cfg= SimpleNamespace()
encoder_cfg.drop_path_rate= 0.2
encoder_cfg.use_checkpoint= True
cfg.encoder_cfg= encoder_cfg

# Decoder
decoder_cfg= SimpleNamespace()
decoder_cfg.decoder_channels= (256,)
decoder_cfg.upsample_mode= "deconv" # nontrainable | deconv | deconvgroup | pixelshuffle
cfg.decoder_cfg= decoder_cfg

# Loss
cfg.loss_type= "src.losses.SmoothBCE"
loss_cfg= SimpleNamespace()
loss_cfg.pos_weight= 256.0
loss_cfg.smooth= 1e-3
cfg.loss_cfg= loss_cfg

# Label cfg
cfg.kernel_sigma= 1.0
cfg.kernel_size= 7
cfg.kernel_type= "smooth"

# Other
cfg.logging_steps= 25
cfg.eval_epochs= 50
cfg.save_epochs= 50
cfg.epochs= 401