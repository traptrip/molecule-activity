import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def main(src_file: Path, dir_folder: Path):
    df = pd.read_csv(src_file, index_col=0)

    for name, smiles in tqdm(enumerate(df["Smiles"]), total=len(df)):
        mol = Chem.MolFromSmiles(smiles)
        save_path = str(dir_folder / f"{name}.mol")
        print(Chem.MolToMolBlock(mol), file=open(save_path, "w+"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--src_file", type=Path, default="data/raw/train.csv")
    parser.add_argument(
        "--dir_folder", type=Path, default="data/interim/train_mol_files"
    )
    args = parser.parse_args()

    assert args.src_file.exists()
    args.dir_folder.mkdir(parents=True, exist_ok=True)

    main(args.src_file, args.dir_folder)
