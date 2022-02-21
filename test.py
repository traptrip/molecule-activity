import os
from pathlib import Path

import pandas as pd
import numpy as np

import warnings
from rdkit import RDLogger

from utils import (
    AtomFeaturizer,
    BondFeaturizer,
    graphs_from_smiles,
)


DATA_PATH = Path("./data")
SEED = 1234
BATCH_SIZE = 256
CLASS_WEIGHTS = {0: 1, 1: 26}
THRESHOLD = 0.5

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
df_train = pd.read_csv(DATA_PATH / "base/train.csv")
print("\nThreshold -", THRESHOLD)
print("\nTRAIN")
print(df_train.Active.value_counts(normalize=True))


def seed_everything(seed=SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


seed_everything(SEED)


def test():
    df = pd.read_csv(DATA_PATH / "base/test.csv").drop("Unnamed: 0", axis=1)

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
            "hybridization": {"s", "sp", "sp2", "sp3"},
        }
    )
    bond_featurizer = BondFeaturizer(
        allowable_sets={
            "bond_type": {"single", "double", "triple", "aromatic"},
            "conjugated": {True, False},
        }
    )

    x_test = graphs_from_smiles(df.Smiles, atom_featurizer, bond_featurizer)
    y_test = [0] * len(df)

    # mpnn = MPNNModel(
    #     atom_dim=x_test[0][0][0].shape[0],
    #     bond_dim=x_test[1][0][0].shape[0],
    #     batch_size=BATCH_SIZE,
    #     message_units=64,
    #     message_steps=4,
    #     num_attention_heads=8,
    #     dense_units=512,
    # )
    # mpnn.load_weights("./models/mpnn_best.h5")

    mpnn = tf.keras.models.load_model(
        "./models/mpnn_best.h5",
        custom_objects={
            "MessagePassing": MessagePassing,
            "TransformerEncoderReadout": TransformerEncoderReadout,
            "f1": f1,
            "AdamW": AdamW,
        },
    )

    test_dataset = MPNNDataset(x_test, y_test, batch_size=BATCH_SIZE)
    y_pred = [
        1 if y > THRESHOLD else 0
        for y in tf.squeeze(mpnn.predict(test_dataset), axis=1)
    ]

    df["Active"] = y_pred
    df["Active"] = df["Active"].astype(bool)
    print("\nTEST")
    print(df.Active.value_counts(normalize=True))

    df.to_csv(f"submission_mpnn_thresh_{THRESHOLD}.csv", index=False)


if __name__ == "__main__":
    test()
