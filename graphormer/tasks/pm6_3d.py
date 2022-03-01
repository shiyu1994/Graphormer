# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from pathlib import Path

from functools import lru_cache
from dataclasses import field

import numpy as np
import torch
from fairseq.data import FairseqDataset
from fairseq.tasks import FairseqTask, register_task

from ..data.dataset import EpochShuffleDataset

import os
import json
from .is2re import pad_1d
from .graph_prediction import GraphPredictionConfig

class PM6Dataset(FairseqDataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.data = []
        self.instance_paths = []
        for dir_name in np.sort(os.listdir(self.path)):
            if dir_name.startswith("Compound"):
                for instance_name in np.sort(os.listdir(Path(self.path) / dir_name)):
                    instance_path = "{0}/{1}/{2}".format(self.path, dir_name, instance_name)
                    self.instance_paths.append(instance_path)
                    self.data.append(json.loads(next(open(Path(instance_path) / "{0}.PM6.S0.json".format(instance_name), "r"))))
        print("Finish loading datasets with {0} samples".format(len(self.data)))

    def parse_json(self, json_obj):
        json_obj = json_obj["pubchem"]["PM6"]
        atoms_obj = json_obj["atoms"]
        atoms = torch.Tensor(atoms_obj["elements"]["number"]).long()  # atom numbers
        num_atoms = atoms_obj["elements"]["atom count"]
        assert num_atoms == len(atoms)
        pos = torch.Tensor(atoms_obj["coords"]["3d"]).float().reshape(-1, 3)
        assert pos.size(0) == num_atoms
        alpha_homo = json_obj["properties"]["energy"]["alpha"]["homo"]
        alpha_lumo = json_obj["properties"]["energy"]["alpha"]["lumo"]
        alpha_gap = json_obj["properties"]["energy"]["alpha"]["gap"]
        beta_homo = json_obj["properties"]["energy"]["beta"]["homo"]
        beta_lumo = json_obj["properties"]["energy"]["beta"]["lumo"]
        beta_gap = json_obj["properties"]["energy"]["beta"]["gap"]
        mulliken = torch.Tensor(json_obj["properties"]["partial charges"]["mulliken"]).float().reshape(-1, 1).repeat([1, 3])
        return {
            "num_atoms": num_atoms,
            "atoms": atoms,
            "pos": pos,

            "alpha_homo": alpha_homo,
            "alpha_lumo": alpha_lumo,
            "alpha_gap": alpha_gap,
            "beta_homo": beta_homo,
            "beta_lumo": beta_lumo,
            "beta_gap": beta_gap,
            "mulliken": mulliken,
        }

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.parse_json(self.data[index])

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        num_atoms = torch.Tensor([sample["num_atoms"] for sample in samples]).long()
        max_num_atoms = torch.max(num_atoms)
        atoms = pad_1d([sample["atoms"] for sample in samples], multiplier=1)
        assert atoms.size(1) == max_num_atoms
        pos = pad_1d([sample["pos"] for sample in samples], multiplier=1)
        alpha_homo = torch.Tensor([sample["alpha_homo"] for sample in samples])
        alpha_lumo = torch.Tensor([sample["alpha_lumo"] for sample in samples])
        alpha_gap = torch.Tensor([sample["alpha_gap"] for sample in samples])
        beta_homo = torch.Tensor([sample["beta_homo"] for sample in samples])
        beta_lumo = torch.Tensor([sample["beta_lumo"] for sample in samples])
        beta_gap = torch.Tensor([sample["beta_gap"] for sample in samples])
        mulliken = pad_1d([sample["mulliken"] for sample in samples], multiplier=1)
        return {
            "num_atoms": num_atoms,

            "net_input": {
                "atoms": atoms,
                "tags": torch.ones_like(atoms).long(),
                "pos": pos,
                "real_mask": torch.ones_like(atoms).bool(),
            },

            "targets": {
                "alpha_homo": alpha_homo,
                "alpha_lumo": alpha_lumo,
                "relaxed_energy": alpha_gap, # alpha_gap
                "beta_homo": beta_homo,
                "beta_lumo": beta_lumo,
                "beta_gap": beta_gap,
                "deltapos": mulliken, # mulliken
            },
        }

    def num_tokens(self, index):
        return self.data[index]["pubchem"]["PM6"]["atoms"]["elements"]["atom count"]

    def size(self, index):
        return self.num_tokens(index)


@dataclass
class PM6PredictionConfig(GraphPredictionConfig):
    data_path: str = field(
        default="data",
        metadata={"help": "directory for data"}
    )

@register_task("pm6_3d", dataclass=PM6PredictionConfig)
class PM6Task3D(FairseqTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_dataset(self, split: str, **kwargs):
        dataset = PM6Dataset(Path(self.cfg.data_path))

        if split == "train":
            dataset = EpochShuffleDataset(
                dataset,
                num_samples=len(dataset),
                seed=self.cfg.seed,
            )

        self.datasets[split] = dataset

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg)

    @property
    def target_dictionary(self):
        return None
