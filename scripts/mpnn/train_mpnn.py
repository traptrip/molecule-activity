import os
from pathlib import Path
import datetime

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import warnings
from rdkit import RDLogger

from scripts.mpnn.utils import (
    AtomFeaturizer,
    BondFeaturizer,
    graphs_from_smiles,
)
from scripts.mpnn.dataset import MPNNDataset
from scripts.mpnn.mpnn import MPNNModel

DATA_PATH = Path("./data")
SEED = 1234
TRAIN_SIZE = 0.9
BATCH_SIZE = 128
N_EPOCHS = 100000
CLASS_WEIGHTS = {0: 1, 1: 26}

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


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


def train():
    df = pd.read_csv(DATA_PATH / "base/train.csv").drop("Unnamed: 0", axis=1)
    df.Active = df.Active.astype(int)

    atom_featurizer = AtomFeaturizer(
        allowable_sets={
            "symbol": {
                "B",
                "Br",
                "C",
                "Ca",
                "Cl",
                "F",
                "H",
                "I",
                "N",
                "Na",
                "O",
                "P",
                "S",
                "Ag",
                "Mg",
                "Se",
                "Zn",
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

    # Shuffle array of indices ranging from 0 to df.shape[0]
    permuted_indices = np.random.permutation(np.arange(df.shape[0]))

    # Train set
    train_index = permuted_indices[: int(df.shape[0] * TRAIN_SIZE)]
    x_train = graphs_from_smiles(
        df.iloc[train_index].Smiles, atom_featurizer, bond_featurizer
    )
    y_train = df.iloc[train_index].Active

    # Valid set
    valid_index = permuted_indices[int(df.shape[0] * TRAIN_SIZE) :]
    x_valid = graphs_from_smiles(
        df.iloc[valid_index].Smiles, atom_featurizer, bond_featurizer
    )
    y_valid = df.iloc[valid_index].Active

    print("Atom dim:", x_train[0][0][0].shape[0])
    print("Bond dim:", x_train[1][0][0].shape[0])
    mpnn = MPNNModel(
        atom_dim=x_train[0][0][0].shape[0],
        bond_dim=x_train[1][0][0].shape[0],
        batch_size=BATCH_SIZE,
        message_units=64,
        message_steps=4,
        num_attention_heads=8,
        dense_units=512,
    )
    # mpnn.load_weights("./models/mpnn_5k.h5")
    schedule_lr = tf.optimizers.schedules.PiecewiseConstantDecay([10000], [1e-3, 1e-4])
    schedule_wd = tf.optimizers.schedules.PiecewiseConstantDecay([10000], [1e-4, 1e-5])
    optimizer = tfa.optimizers.AdamW(
        # beta_1=0.9,
        # beta_2=0.98,
        # epsilon=1e-06,
        weight_decay=schedule_wd,
        learning_rate=schedule_lr,
    )

    mpnn.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=optimizer,  # keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[f1],
    )

    callbacks = [
        # ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=4e-5),
        ModelCheckpoint(
            filepath=os.path.join("./models/", f"mpnn_lstm_best.h5"),
            monitor="val_f1",
            save_best_only=True,
            verbose=1,
            mode="max",
            save_weights_only=True,
        ),
        TensorBoard(
            log_dir=os.path.join("logs", f"{datetime.datetime.now():%Y-%m-%d_%H-%M}")
        ),
    ]

    # keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)

    train_dataset = MPNNDataset(x_train, y_train, batch_size=BATCH_SIZE)
    valid_dataset = MPNNDataset(x_valid, y_valid, batch_size=BATCH_SIZE)

    history = mpnn.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=N_EPOCHS,
        verbose=1,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks,
    )

    mpnn.save_weights("./models/mpnn_1m.h5")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["f1"], label="train f1")
    plt.plot(history.history["val_f1"], label="valid f1")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("F1", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig("./models/history.png")


if __name__ == "__main__":
    train()
