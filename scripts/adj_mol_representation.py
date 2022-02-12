import argparse
from collections import Counter
from pathlib import Path
from random import choice

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def update_maximum(curr_counter: Counter, max_coutnter: Counter) -> Counter:
    new_counter = Counter()
    atoms = set(list(curr_counter.keys()) + list(max_coutnter.keys()))
    for atom in atoms:
        new_counter[atom] = max(curr_counter[atom], max_coutnter[atom])
    return new_counter


def get_max_counter(df: pd.DataFrame) -> Counter:
    max_coutnter = Counter()
    for smiles in df["Smiles"].to_list():
        mol = Chem.MolFromSmiles(smiles)
        mol_seq = [item.GetSymbol() for item in mol.GetAtoms()]
        curr_counter = Counter(mol_seq)

        max_coutnter = update_maximum(curr_counter, max_coutnter)

    max_coutnter = Counter(
        {
            "I": 6,
            "Li": 3,
            "As": 4,
            "Na": 6,
            "F": 21,
            "S": 8,
            "Ca": 1,
            "Se": 1,
            "P": 4,
            "N": 50,
            "K": 1,
            "Sr": 2,
            "Zn": 1,
            "H": 6,
            "O": 75,
            "Al": 16,
            "Mg": 3,
            "B": 2,
            "C": 184,
            "Ag": 1,
            "Cl": 6,
            "Si": 4,
            "Br": 6,
        }
    )

    return max_coutnter


def get_available_numbers(atoms_list: list, max_coutnter) -> dict:
    available_numbers = dict()
    for atom in atoms_list:
        available_numbers[atom.GetSymbol()] = [
            idx for idx in range(max_coutnter[atom.GetSymbol()])
        ]

    return available_numbers


def get_match_dict(atoms_list: list, available_numbers: dict) -> dict:
    match_dict = dict()
    for idx in range(len(atoms_list)):
        atom_symb = atoms_list[idx].GetSymbol()
        atom_number = choice(available_numbers[atom_symb])
        match_dict[idx] = f"{atom_symb}_{atom_number}"
        available_numbers[atom_symb].remove(atom_number)

    return match_dict


def get_adjacency_matrix(atoms_list: list, header: list, match_dict: dict) -> list:
    adj_matrix = [[0 for _ in range(len(header))] for _ in range(len(header))]
    for atom in atoms_list:
        for bond in atom.GetBonds():
            bond_type = bond.GetBondTypeAsDouble()
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            adj_idx1 = header.index(match_dict[begin_atom.GetIdx()])
            adj_idx2 = header.index(match_dict[end_atom.GetIdx()])
            adj_matrix[adj_idx1][adj_idx2] = bond_type
            adj_matrix[adj_idx2][adj_idx1] = bond_type

    return adj_matrix


def main(src_path: Path, dir_path: Path, N: int) -> None:
    df = pd.read_csv(src_path, index_col=0)
    df["Active"] = df["Active"] if "Active" in df.columns else None

    max_coutnter = get_max_counter(df)

    header = list()
    for atom in sorted(max_coutnter):
        for idx in range(max_coutnter[atom]):
            header.append(f"{atom}_{idx}")

    new_header = list()
    for adj_idx in range(len(header)):
        for j in range(adj_idx):
            new_header.append(f"{header[adj_idx]}_{j}")
    new_header.append("Active")

    with open(str(dir_path / f"{src_path.name}"), "w") as csv:
        csv.write(",".join(new_header))
        csv.write("\n")

        for _ in tqdm(range(N)):
            for smiles, active in tqdm(df.values):
                mol = Chem.MolFromSmiles(smiles)
                atoms_list = list(mol.GetAtoms())
                available_numbers = get_available_numbers(atoms_list, max_coutnter)
                match_dict = get_match_dict(atoms_list, available_numbers)
                adj_matrix = get_adjacency_matrix(atoms_list, header, match_dict)

                data_line = list()
                for adj_idx in range(len(adj_matrix)):
                    data_line.extend(adj_matrix[adj_idx][0:adj_idx])
                data_line.append(active)

                csv.write(",".join(map(str, data_line)))
                csv.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("src_path", type=Path, help="Путь к сырому датасету")
    parser.add_argument("dir_path", type=Path, help="Путь, куда сохранить результат")
    parser.add_argument("-N", default=1, type=int, help="Во солько раз увел. датасет")
    args = parser.parse_args()

    assert args.src_path.exists()
    assert args.N > 0
    args.dir_path.mkdir(parents=True, exist_ok=True)

    main(args.src_path, args.dir_path, args.N)
