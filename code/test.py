from v2a_model import V2AModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        save_last=True)
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    trainer = pl.Trainer(
        gpus=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        max_epochs=4000,
        check_val_every_n_epoch=50,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )

    model = V2AModel(opt)
    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    testset = create_dataset(opt.dataset.metainfo, opt.dataset.test)

    trainer.test(model, testset, ckpt_path=checkpoint)

if __name__ == '__main__':
    main()