# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset
import torch.distributed as dist
from torch_geometric.data import InMemoryDataset, download_url, Data, extract_zip
import torch
import pickle
import os.path as osp
import shutil

class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()

class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()


class PDBBind(InMemoryDataset):
    url = "https://pretrain.blob.core.windows.net/datasets/pdbbind.zip"

    def __init__(self, root, set_name, cut_off, split, transform=None, pre_transform=None,
                 pre_filter=None):
        self.path = f"{root}/pdbbind/preprocess"
        assert set_name in ["refined-set-2019", "refined-set-2020"]
        assert cut_off in ["5-5-0", "5-5-5", "6-5-5"]
        assert split in ["train", "valid", "test"]
        super(PDBBind, self).__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{set_name}/{set_name}-{cut_off}_{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self):
        return self.path

    @property
    def processed_dir(self):
        return self.path

    @property
    def processed_file_names(self):
        to_return = []
        for set_name in ["refined-set-2019", "refined-set-2020"]:
            for cut_off in ["5-5-0", "5-5-5", "6-5-5"]:
                for split in ["train", "valid", "test"]:
                    to_return.append(f"{set_name}/{set_name}-{cut_off}_{split}.pt")
        return to_return

    @property
    def raw_file_names(self):
        to_return = []
        for set_name in ["refined-set-2019", "refined-set-2020"]:
            for cut_off in ["5-5-0", "5-5-5", "6-5-5"]:
                for split in ["train", "valid", "test"]:
                    to_return.append(f"{set_name}/{set_name}-{cut_off}_{split}.pkl")
        return to_return

    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            shutil.rmtree(self.raw_dir)
            path = download_url(self.url, self.root)
            extract_zip(path, self.root)
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        for set_name in ["refined-set-2019", "refined-set-2020"]:
            for cut_off in ["5-5-0", "5-5-5", "6-5-5"]:
                for split in ["train", "valid", "test"]:
                    data_path = f"{self.raw_dir}/{set_name}/{set_name}-{cut_off}_{split}.pkl"
                    with open(data_path, 'rb') as f:
                        mols = pickle.load(f)
                    torch.save(self.collate(mols),
                       osp.join(self.processed_dir, f'{set_name}/{set_name}-{cut_off}_{split}.pt'))


class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None

        train_set = None
        valid_set = None
        test_set = None

        root = "dataset"
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
        elif name == "zinc":
            inner_dataset = MyZINC(root=root)
            train_set = MyZINC(root=root, split="train")
            valid_set = MyZINC(root=root, split="val")
            test_set = MyZINC(root=root, split="test")
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=root, name=nm)
        elif name == "pdbbind":
            set_name = None
            cut_off = None
            for param in params:
                key, value = param.split("=")
                if key == "set":
                    set_name = value
                if key == "cut_off":
                    cut_off = value
            train_set = PDBBind(root, set_name=set_name, cut_off=cut_off, split="train")
            valid_set = PDBBind(root, set_name=set_name, cut_off=cut_off, split="valid")
            test_set = PDBBind(root, set_name=set_name, cut_off=cut_off, split="test")
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                )
        else:
            return (
                None
                if inner_dataset is None
                else GraphormerPYGDataset(inner_dataset, seed)
            )
