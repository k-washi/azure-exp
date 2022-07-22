import os
from dotenv import load_dotenv
load_dotenv(verbose=True)
load_dotenv(dotenv_path=".env")


from pathlib import Path
import neptune
import hydra
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger

from src.yogacls.train.datamodule import YogaDataModule
from src.yogacls.train.model import YogaPoseClassifier

SEED = 3407
seed_everything(SEED, workers=True)

state_dict_key = "state_dict"
cfg_path = Path(__file__, "..", "..", "..", "..", "..").resolve().joinpath("src/conf")

NEPTUNE_AI_API_TOKEN_KEY = "NEPTUNE_AI_API_TOKEN"

def create_neptune_logger(project_name, log_mode, log_exe_id, log_desc, tags=[], source_files=[]):
    neptune_api_token = os.environ.get(NEPTUNE_AI_API_TOKEN_KEY, None)

    if neptune_api_token is None:
        raise ValueError(f"Neptune AIのAPI Tokenを読み込めませんでした。")
    neptune_logger = NeptuneLogger(
        
        project=project_name,
        name=f"exp_{log_exe_id}",
        api_key=neptune_api_token,
        mode=log_mode,
        description=log_desc,
        #tags=tags,
        #source_files=source_files
        )
    return neptune_logger

def pick_hparam(cfg):
    return {
        "lr": cfg.ml.learning_rate,
        "batch_size": cfg.ml.batch_size,
        "epochs": cfg.ml.epochs,
        "scheduler_init_t": cfg.ml.scheduler.t_initial
    }

@hydra.main(config_path=str(cfg_path), config_name="default", version_base=None)
def experiment(cfg) -> None:
    cfg.ml.log_name = "yoga_loghtvit_tiny"
    cfg.ml.log_project_name = "k-washi/yoga-pose-classification"
    cfg.ml.log_exe_id = "00001"
    cfg.ml.log_doc = "LightViT_tinyの実験"
    cfg.ml.log_tags = ["lightvit_tiny"]
    cfg.ml.fast_dev_run = False
    
    # ハイパーパラメータ
    cfg.ml.learning_rate = 1e-3
    cfg.ml.batch_size = 32
    cfg.ml.epochs = 200
    cfg.ml.scheduler.t_initial = 50
    cfg.ml.scheduler.warm_up_t = 5
    print(cfg)
    
    logger = create_neptune_logger(
        cfg.ml.log_project_name,
        cfg.ml.log_mode,
        cfg.ml.log_exe_id,
        cfg.ml.log_doc,
        tags=cfg.ml.log_tags
    )
    
    

    logger.log_hyperparams(pick_hparam(cfg))
    logger.experiment["experiment_params"].log(cfg.ml)
    # モデル保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.ml.model_save.save_dir}/{cfg.ml.log_name}/{cfg.ml.log_exe_id}",
        filename="checkpoint-{epoch:04d}-{val_loss:.4f}",
        save_top_k=cfg.ml.model_save.top_k,
        monitor=cfg.ml.model_save.monitor,
        mode=cfg.ml.model_save.mode
    )
    
    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

    try:
        device = "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    
    dataset = YogaDataModule(cfg)
    model = YogaPoseClassifier(cfg)
    
    
    
    
    if device == "gpu":
        trainer = Trainer(
            precision=cfg.ml.mix_precision,
            accelerator=device,
            #devices=cfg.ml.gpu_devices,
            max_epochs=cfg.ml.epochs,
            accumulate_grad_batches=cfg.ml.accumulate_grad_batches,
            gradient_clip_val=cfg.ml.gradient_clip_val,
            profiler=cfg.ml.profiler,
            fast_dev_run=cfg.ml.fast_dev_run,
            logger=logger,
            callbacks=callback_list
        )
    else:
        raise NotImplementedError("cpuバージョンの訓練は実装していません。")
    

    trainer.fit(model, dataset)

    print("Fin Train")
    model_dir = f"{cfg.ml.model_save.save_dir}/{cfg.ml.log_name}/{cfg.ml.log_exe_id}"
    best_model_path = f"{model_dir}/best_model.ckpt"
    # model チェックポイントの保存
    if len(checkpoint_callback.best_model_path):
        print(f"BEST MODEL: {checkpoint_callback.best_model_path}")
        print(f"BEST SCORE: {checkpoint_callback.best_model_score}")
        _ckpt = torch.load(checkpoint_callback.best_model_path)
        model = YogaPoseClassifier(cfg)
        model.load_state_dict(_ckpt[state_dict_key])
        torch.save(model.model.state_dict(), best_model_path)
        print(f"To BEST MODEL: {best_model_path}")
        # FOR LOAD
        # _ckpt = torch.load(f"{cfg.ml.model_save.save_dir}/{cfg.ml.log_name}/{cfg.ml.version}/best_model.ckpt")
        # model.model.load_state_dict(_ckpt)
    else:
        print("best model is not exist.")
    
    _ = trainer.test(
        model=model,
        dataloaders=dataset
    )
    
    # ログに設定などを出力
    logger.log_model_summary(model=model, max_depth=-1)
if __name__ == "__main__":
    experiment()