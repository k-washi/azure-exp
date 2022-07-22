import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.yogacls.train.dataset import load_datalist, YogaPoseDataset


class YogaDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ml_cfg = cfg.ml
    
    def setup(self, stage=None):
        data_list, class_labels = load_datalist(self.ml_cfg.dataset.data_dir)
        print(f"DATA NUM: {len(data_list)}, example. {data_list[:2]}")
        self.train_list, self.eval_list = train_test_split(
                data_list,
                test_size=float(self.ml_cfg.dataset.eval_rate), shuffle=True, random_state=int(self.ml_cfg.seed)
            )
        if stage == "fit" or stage is None:
            with open(self.ml_cfg.dataset.class_ids_path, "w") as f:
                json.dump(class_labels, f, indent=2, ensure_ascii=False)
            if self.ml_cfg.debug:
                self.train_list = self.train_list[:5]
                self.eval_list = self.eval_list[:5]

        if stage == "test" or stage is None:
            if self.ml_cfg.debug:
                self.eval_list = self.eval_list[:10]
    
    def train_dataloader(self) -> DataLoader:
        train_dataset = YogaPoseDataset(
            self.cfg, self.train_list, is_aug=self.ml_cfg.dataset.use_augments
        )
        print(f"TRAIN DATASET NUM: {len(self.train_list)}")
        return DataLoader(
            train_dataset,
            batch_size=self.ml_cfg.batch_size,
            drop_last=self.ml_cfg.drop_last,
            shuffle=True,
            num_workers=self.ml_cfg.num_workers,
            pin_memory=self.ml_cfg.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        eval_dataset = YogaPoseDataset(self.cfg, self.eval_list, is_aug=False)
        print(f"VAL DATASET NUM: {len(self.eval_list)}")
        return DataLoader(
            eval_dataset,
            batch_size=self.ml_cfg.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.ml_cfg.num_workers,
            pin_memory=self.ml_cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        # ! とりあえず、テスト用のデータセットをもらうまでevalを使う
        test_dataset = YogaPoseDataset(self.cfg, self.eval_list, is_aug=False)
        print(f"TEST DATASET NUM: {len(self.eval_list)}")
        return DataLoader(
            test_dataset,
            batch_size=self.ml_cfg.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.ml_cfg.num_workers,
            pin_memory=self.ml_cfg.pin_memory,
        )