import os
import time
from pathlib import Path
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import warnings
from rdkit import RDLogger
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

from scripts.mpnn.utils import (
    AtomFeaturizer,
    BondFeaturizer,
    graphs_from_smiles,
)
from scripts.mpnn.dataset import MPNNDataset
from scripts.mpnn.mpnn import (
    MPNNModel,
    MessagePassing,
    TransformerEncoderReadout,
)

DATA_PATH = Path("./data")
SEED = 1234
TRAIN_SIZE = 0.95
TEST_SIZE = 1 - TRAIN_SIZE
BATCH_SIZE = 256
N_EPOCHS = 500
CLASS_WEIGHTS = {0: 1, 1: 26}
THRESHOLD = 0.8
N_FOLD = 5


# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

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
        "hybridization": {"s", "sp", "sp2", "sp3"},  # TODO: sp3d and sp3d2
    }
)
bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)


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


def train_model_and_predict(train, test, folds, atom_featurizer, bond_featurizer):
    x = train.drop("Active", axis=1)
    y = train["Active"].astype(int)
    x_test = graphs_from_smiles(test.Smiles, atom_featurizer, bond_featurizer)
    test_dataset = MPNNDataset(x_test, [0] * len(test), batch_size=BATCH_SIZE)
    best_score = 0
    best_model = None
    best_pred = None
    models = [
        MPNNModel(
            atom_dim=x_test[0][0][0].shape[0],
            bond_dim=x_test[1][0][0].shape[0],
            batch_size=BATCH_SIZE,
            message_units=64,
            message_steps=4,
            num_attention_heads=8,
            dense_units=512,
        )
        for _ in range(N_FOLD)
    ]
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    for model in models:
        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=[f1],
        )

    for _ in range(10):
        scores = []
        prediction = np.zeros(len(test))

        for fold_n, (train_index, valid_index) in enumerate(folds.split(x, y)):
            print("\nFold", fold_n + 1, "started at", time.ctime())
            x_train, x_valid = x.loc[train_index], x.loc[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            x_train = graphs_from_smiles(
                x_train.Smiles, atom_featurizer, bond_featurizer
            )
            x_valid = graphs_from_smiles(
                x_valid.Smiles, atom_featurizer, bond_featurizer
            )

            train_dataset = MPNNDataset(x_train, y_train, batch_size=BATCH_SIZE)
            valid_dataset = MPNNDataset(x_valid, y_valid, batch_size=BATCH_SIZE)

            mpnn = models[fold_n]

            history = mpnn.fit(
                train_dataset,
                epochs=N_EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=2,
                class_weight=CLASS_WEIGHTS,
            )

            y_pred_valid = [
                1 if y > THRESHOLD else 0
                for y in tf.squeeze(mpnn.predict(valid_dataset), axis=1)
            ]

            y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1).numpy()

            scores.append(f1_score(y_valid, y_pred_valid))
            print(f"Score: {scores[-1]:.4f}.")
            prediction += y_pred

        prediction /= N_FOLD

        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_model = mpnn
            best_pred = prediction
            best_model.save_weights("models/mpnn_best_cv.h5")

        print(f"CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):.4f}.")

    return best_model, best_pred


if __name__ == "__main__":
    train = pd.read_csv(DATA_PATH / "base/train.csv", index_col=0)
    train.Active = train.Active.astype(int)
    test = pd.read_csv(DATA_PATH / "base/test.csv", index_col=0)

    kfold = StratifiedKFold(n_splits=N_FOLD, shuffle=True)

    best_model, pred = train_model_and_predict(
        train,
        test,
        kfold,
        atom_featurizer,
        bond_featurizer,
    )

    test["Active"] = pred.astype(bool)
    test.to_csv("models/submission.csv")
