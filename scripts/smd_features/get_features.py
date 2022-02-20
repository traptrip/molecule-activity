import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from smdt.molecular_descriptors import (
    getAllDescriptorsforMol,
    _topology,
    _constitutional,
    _bcut,
    _basak,
    _cats2d,
    _charge,
    _connectivity,
    _estate,
    _geary,
    _kappa,
    _moe,
    _moran,
    _moreaubroto,
)


def main(src_path: Path, dist_path: Path):
    df = pd.read_csv(src_path, index_col=0)
    AllDescriptors = list()

    for idx, row in tqdm(enumerate(df.values), total=len(df)):
        smiles, active = row
        try:
            descriptors = getAllDescriptorsforMol(Chem.MolFromSmiles(smiles))
            descriptors = [idx] + descriptors + [active]
            AllDescriptors.append(descriptors)
        except Exception as e:
            print(idx, smiles, active)

    AllDescriptors = pd.DataFrame(AllDescriptors)
    AllDescriptors = AllDescriptors.rename({0: "idx", 759: "Active"}, axis=1)
    AllDescriptors.to_csv(dist_path / src_path.name, index=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--src_path", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument(
        "--dist_path", type=Path, default=Path("data/interim/smd_features")
    )
    args = parser.parse_args()

    assert args.src_path.exists()
    args.dist_path.mkdir(
        exist_ok=True,
        parents=True,
    )
    main(args.src_path, args.dist_path)
