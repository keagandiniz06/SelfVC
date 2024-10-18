import pytorch_lightning as pl
from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
import os




@hydra_runner(config_path="/home/keagan/Documents/projects/SelfVC/models/", config_name="hifigan.yaml")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    trainer.fit(model)


if __name__ == '__main__':
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()  # noqa pylint: disable=no-value-for-parameter    