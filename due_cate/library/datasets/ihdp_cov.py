import torch
import pyreadr
import numpy as np

from pathlib import Path

from torch.utils import data

from sklearn import preprocessing
from sklearn import model_selection

_CONTINUOUS_COVARIATES = [
    "bw",
    "b.head",
    "preterm",
    "birth.o",
    "nnhealth",
    "momage",
]

_BINARY_COVARIATES = [
    "sex",
    "twin",
    "b.marr",
    "mom.lths",
    "mom.hs",
    "mom.scoll",
    "cig",
    "first",
    "booze",
    "drugs",
    "work.dur",
    "prenatal",
    "ark",
    "ein",
    "har",
    "mia",
    "pen",
    "tex",
    "was",
]

_TREATMENT = ["treat"]


class IHDPCov(data.Dataset):
    def __init__(self, root, split, mode, seed):
        root = Path(root)
        data_path = root / "ihdp.RData"
        # Download data if necessary
        if not data_path.exists():
            r = requests.get(
                "https://github.com/vdorie/npci/raw/master/examples/ihdp_sim/data/ihdp.RData"
            )
            with open(data_path, "wb") as f:
                f.write(r.content)
        df = pyreadr.read_r(str(data_path))["ihdp"]
        # Make observational as per Hill 2011
        df = df[~((df["treat"] == 1) & (df["momwhite"] == 0))]
        df = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES + _TREATMENT]
        # Standardize continuous covariates
        df[_CONTINUOUS_COVARIATES] = preprocessing.StandardScaler().fit_transform(
            df[_CONTINUOUS_COVARIATES]
        )
        # Generate response surfaces
        rng = np.random.default_rng(seed)
        x = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES]
        t = df["treat"]
        beta_x = rng.choice(
            [0.0, 0.1, 0.2, 0.3, 0.4],
            size=(len(_CONTINUOUS_COVARIATES) + len(_BINARY_COVARIATES),),
            p=[0.6, 0.1, 0.1, 0.1, 0.1],
        )
        mu0 = np.exp((x + 0.5).dot(beta_x))
        df["mu0"] = mu0
        mu1 = (x + 0.5).dot(beta_x)
        omega = (mu1[t == 1] - mu0[t == 1]).mean(0) - 4
        mu1 -= omega
        df["mu1"] = mu1
        eps = rng.normal(size=t.shape)
        y0 = mu0 + eps
        df["y0"] = y0
        y1 = mu1 + eps
        df["y1"] = y1
        y = t * y1 + (1 - t) * y0
        df["y"] = y
        # Train test split
        df_train, df_test = model_selection.train_test_split(
            df, test_size=0.1, random_state=seed
        )
        df_train = df_train[df["b.marr"] == 1]
        self.mode = mode
        self.split = split
        # Set x, y, and t values
        self.y_mean = (
            df_train["y"].to_numpy(dtype="float32").mean(keepdims=True)
            if mode == "mu"
            else np.asarray([0.0], dtype="float32")
        )
        self.y_std = (
            df_train["y"].to_numpy(dtype="float32").std(keepdims=True)
            if mode == "mu"
            else np.asarray([1.0], dtype="float32")
        )
        covars = _CONTINUOUS_COVARIATES + _BINARY_COVARIATES
        covars.remove("b.marr")
        self.dim_input = len(covars)
        self.dim_treatment = 1
        self.dim_output = 1
        if self.split == "test":
            self.x = df_test[covars].to_numpy(dtype="float32")
            self.t = df_test["treat"].to_numpy(dtype="float32")
            self.mu0 = df_test["mu0"].to_numpy(dtype="float32")
            self.mu1 = df_test["mu1"].to_numpy(dtype="float32")
            self.y0 = df_test["y0"].to_numpy(dtype="float32")
            self.y1 = df_test["y1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = self.mu1 - self.mu0
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")
        else:
            df_train, df_valid = model_selection.train_test_split(
                df_train, test_size=0.3, random_state=seed
            )
            if split == "train":
                df = df_train
            elif split == "valid":
                df = df_valid
            else:
                raise NotImplementedError("Not a valid dataset split")
            self.x = df[covars].to_numpy(dtype="float32")
            self.t = df["treat"].to_numpy(dtype="float32")
            self.mu0 = df["mu0"].to_numpy(dtype="float32")
            self.mu1 = df["mu1"].to_numpy(dtype="float32")
            self.y0 = df["y0"].to_numpy(dtype="float32")
            self.y1 = df["y1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = df["y"].to_numpy(dtype="float32")
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = (
            torch.from_numpy(self.x[idx]).float()
            if self.mode == "pi"
            else torch.from_numpy(np.hstack([self.x[idx], self.t[idx]])).float()
        )
        targets = torch.from_numpy((self.y[idx] - self.y_mean) / self.y_std).float()
        return inputs, targets
