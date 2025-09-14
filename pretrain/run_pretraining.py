# Training in 256Hz data and 4s
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from engine_pretraining import *
from configs import *
import wandb
torch.set_float32_matmul_precision("medium")

seed_torch(7)


# init model
wandb.login(key='ca0cbcfb28dcbf7cd1b54e0a6d40d8a482fc730f')
# --- W&B: thiết lập run ---
# (tuỳ chọn) đặt entity/project/name, tags, config
wandb_logger = WandbLogger(
    project="eegpt-pretrain",
    name=f"EEGPT-{tag}-{variant}",
    log_model=True,               # tự upload checkpoint tốt nhất nếu có internet
    tags=[tag, variant]
)
# đẩy hparams lên W&B (tuỳ chọn)
wandb_logger.experiment.config.update(
    {"tag": tag, "variant": variant, **get_config(**(MODELS_CONFIGS[tag]))},
    allow_val_change=True
)

model = LitEEGPT(get_config(**(MODELS_CONFIGS[tag])), 
                 USE_LOSS_A =(variant != "A"),
                 USE_LN     =(variant != "B"),
                 USE_SKIP   =(variant != "C"))
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
# (khuyên dùng) lưu best theo val/loss nếu bạn có metric đó
ckpt_cb = pl.callbacks.ModelCheckpoint(
    monitor="val/loss", mode="min", save_top_k=3, filename="{epoch:02d}-{val_loss:.4f}"
)
callbacks = [lr_monitor, ckpt_cb]

trainer = pl.Trainer(strategy='auto', devices=devices, max_epochs=max_epochs, callbacks=callbacks,
                     logger=[
                             wandb_logger,
                             pl_loggers.TensorBoardLogger('./logs/', name=f"EEGPT_{tag}_{variant}_tb"), 
                             pl_loggers.CSVLogger('./logs/', name=f"EEGPT_{tag}_{variant}_csv")],
                     log_every_n_steps=10
                     )

# model

trainer.fit(model, train_loader, valid_loader)
wandb_logger.experiment.finish()