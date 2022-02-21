from pathlib import Path

import pandas as pd
import numpy as np

import warnings
from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from utils import (
    AtomFeaturizer,
    BondFeaturizer,
    graph_features_from_smiles,
)
from catboost import CatBoostClassifier


DATA_PATH = Path("./data")
SEED = 1234
TRAIN_SIZE = 0.9
TEST_SIZE = 1 - TRAIN_SIZE

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def seed_everything(seed=SEED):
    np.random.seed(seed)


seed_everything(SEED)

model = CatBoostClassifier(
    task_type="GPU",
    devices=[0],
    auto_class_weights="SqrtBalanced",
    iterations=3000,
    eval_metric="F1",
    text_features=["Smiles"],
    embedding_features=["atom_features", "bond_features"],
)


def train():
    print("Reading Dataset")
    df = pd.read_csv(DATA_PATH / "base/train.csv", index_col=0)
    df.Active = df.Active.astype(int)

    atom_featurizer = AtomFeaturizer(
        allowable_sets={
            "symbol": {
                "Cl",
                "S",
                "Se",
                "Zn",
                "K",
                "F",
                "Ag",
                "P",
                "N",
                "Na",
                "Mg",
                "C",
                "Br",
                "O",
                "B",
                "Si",
                "I",
                "Ca",
                "Al",
                "H",
                "Sr",
                "Li",
                "As",
            },
            "n_valence": {0, 1, 2, 3, 4, 5, 6},
            "n_hydrogens": {0, 1, 2, 3, 4},
            "hybridization": {
                "s",
                "sp",
                "sp2",
                "sp3",
            },
        }
    )
    bond_featurizer = BondFeaturizer(
        allowable_sets={
            "bond_type": {"single", "double", "triple", "aromatic"},
            "conjugated": {True, False},
        }
    )

    atom_features, bond_features = graph_features_from_smiles(
        df.Smiles, atom_featurizer, bond_featurizer
    )
    print("Atom features shape:", atom_features.shape)
    print("Bond features shape:", bond_features.shape)
    df["atom_features"] = atom_features
    df["bond_features"] = bond_features

    # x_train, x_val, y_train, y_val = train_test_split(
    #     df.drop("Active", axis=1),
    #     df["Active"],
    #     test_size=TEST_SIZE,
    #     random_state=SEED,
    # )

    # print("Train Catboost")
    # model.fit(
    #     x_train,
    #     y_train,
    #     use_best_model=True,
    #     eval_set=(x_val, y_val),
    # )
    # print("Saving model")
    # model.save_model("models/catboost_model.cbm")
    # print(f"Model saved to 'models/catboost_model.cbm'")


if __name__ == "__main__":
    train()
