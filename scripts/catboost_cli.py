from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

SEED = 1234
TRAIN_SIZE = 0.2

model = CatBoostClassifier(
    auto_class_weights="SqrtBalanced",
    iterations=3000,
    eval_metric="F1",
)


def train(arguments):
    print("Reading Dataset")
    df = pd.read_csv(arguments.data_path, dtype='int8') # , engine="pyarrow"
    df = df.drop_duplicates()
    
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("Active", axis=1),
        df["Active"].astype(int),
        test_size=TRAIN_SIZE,
        random_state=SEED,
    )
    print("Train Catboost")
    model.fit(
        X_train,
        y_train,
        use_best_model=True,
        eval_set=(X_test, y_test),
    )
    print("Saving model")
    model.save_model(arguments.model_filepath)
    print(f"Model saved to {str(arguments.model_filepath)}")


def test(arguments):
    print("Reading Dataset")
    df = pd.read_csv(arguments.data_path)
    print("Predicting")
    model.load_model(arguments.model_filepath)
    pred = model.predict(df)

    df = pd.read_csv(arguments.base_data_path)
    df["Active"] = pred
    df["Active"] = df["Active"].astype(bool)
    df.to_csv("submission.csv", index=False)
    print("Saved submission to submission.csv")


def setup_parser(parser):
    """Setup arguments parser for CLI"""
    subparsers = parser.add_subparsers(help="Choose command")

    train_parser = subparsers.add_parser(
        "train",
        help="train catboost",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    train_parser.set_defaults(callback=train)
    train_parser.add_argument(
        "-d", "--data_path", default=Path("./data/train.csv"), type=Path
    )
    train_parser.add_argument(
        "-m", "--model_filepath", default=Path("./models/model.cbm"), type=Path
    )

    test_parser = subparsers.add_parser(
        "test",
        help="test catboost",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    test_parser.set_defaults(callback=test)
    test_parser.add_argument(
        "-b", "--base_data_path", default=Path("./data/base/test.csv"), type=Path
    )
    test_parser.add_argument(
        "-d", "--data_path", default=Path("./data/test.csv"), type=Path
    )
    test_parser.add_argument(
        "-s", "--submit_path", default=Path("./submission.csv"), type=Path
    )
    test_parser.add_argument(
        "-m", "--model_filepath", default=Path("./models/model.cbm"), type=Path
    )


def main():
    parser = ArgumentParser(
        prog="catboost_cli",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
