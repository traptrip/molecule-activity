import argparse
from collections import Counter
from pathlib import Path
from random import choice

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]


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

    # max_coutnter = Counter(
    #     {
    #         "I": 6,
    #         "Li": 3,
    #         "As": 4,
    #         "Na": 6,
    #         "F": 21,
    #         "S": 8,
    #         "Ca": 1,
    #         "Se": 1,
    #         "P": 4,
    #         "N": 50,
    #         "K": 1,
    #         "Sr": 2,
    #         "Zn": 1,
    #         "H": 6,
    #         "O": 75,
    #         "Al": 16,
    #         "Mg": 3,
    #         "B": 2,
    #         "C": 184,
    #         "Ag": 1,
    #         "Cl": 6,
    #         "Si": 4,
    #         "Br": 6,
    #     }
    # )

    max_coutnter = Counter(
        {
            "S": 4,
            "Cl": 6,
            "C": 40,
            "Mg": 3,
            "K": 1,
            "Si": 4,
            "N": 10,
            "Zn": 1,
            "P": 4,
            "Sr": 2,
            "As": 4,
            "Ca": 1,
            "O": 18,
            "H": 6,
            "Br": 6,
            "F": 21,
            "I": 6,
            "Ag": 1,
            "Na": 3,
            "B": 2,
            "Se": 1,
            "Li": 3,
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
        # atom_number = available_numbers[atom_symb][0]
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
            if bond_type == 1.0:
                adj_matrix[adj_idx1][adj_idx2] = int(1)
                adj_matrix[adj_idx2][adj_idx1] = int(1)
            elif bond_type == 1.5:
                adj_matrix[adj_idx1][adj_idx2] = int(2)
                adj_matrix[adj_idx2][adj_idx1] = int(2)
            elif bond_type == 2:
                adj_matrix[adj_idx1][adj_idx2] = int(3)
                adj_matrix[adj_idx2][adj_idx1] = int(3)
            elif bond_type == 3:
                adj_matrix[adj_idx1][adj_idx2] = int(4)
                adj_matrix[adj_idx2][adj_idx1] = int(4)
            else:
                raise RuntimeError(bond_type)

    return adj_matrix


def main(src_path: Path, dir_path: Path, N: int, as_train: bool) -> None:
    df = pd.read_csv(src_path, index_col=0)
    df["Active"] = df["Active"] if "Active" in df.columns else None
    
    if as_train:
        print("Processing as Train")
        large_files = (2643, 3465, 4731, 1817, 578, 3065, 2245, 117, 3501, 4903, 598, 3445, 5181, 2597, 3422, 4843, 4612, 61, 4419, 172, 1836, 3340, 1965, 79, 3311, 4779, 3896, 1996, 2252, 2973, 3978, 1919, 1498, 4458, 3023, 1226, 3983, 3408, 2719, 391, 525, 2168, 3192, 2543, 3752, 1261, 2670, 2862, 1202, 5113, 4816, 1987, 1951, 1484, 3172, 5141, 4156, 4671, 3078, 3947, 1152, 2459, 2641, 4149, 209, 2610, 1377, 4976, 4340, 3047, 3675, 5412, 200, 5184, 3447, 5154, 5190, 3796, 2306, 1078, 4264, 4349, 565, 3871, 360, 5416, 3653, 1815, 635, 3994, 408, 482, 3814, 1485, 863, 398, 3608, 3420, 3458, 732, 3469, 5248, 3100, 2929, 1773, 5083, 692, 1772, 1633, 1536, 3148, 1917, 1419, 3738, 4273, 3329, 1675, 3597, 2171, 1898, 335, 4815, 4053, 2537, 2989, 5490, 2329, 3251, 5277, 2690, 1703, 147, 679, 3942, 3345, 3131, 138, 5202, 3014, 2633, 3015, 1155, 4670, 1770, 5518, 2104, 5255, 1397, 1914, 210, 3921, 5471, 4444, 4301, 94, 3815, 1667, 1017, 1994, 1671, 3765, 1233, 954, 2460, 3225, 3179, 1865, 5132, 5091, 4060, 5373, 3619, 703, 3874, 5147, 793, 5377, 4728, 2508, 3167, 995, 1085, 116, 5484, 5040, 2916, 4690, 3724, 2557, 4207, 2266, 3805, 4429, 4245, 3376, 3153, 3907, 1718, 1004, 2269, 670, 1000, 4559, 4129, 3403, 1970, 959, 907, 5449, 4475, 4181, 3332, 1197)
    else:
        print("Processing as Test")
        large_files = (1342, 1459, 498, 1417, 1265, 1316, 1287, 16, 1440, 988, 548, 1390, 598, 1428, 767, 1189, 472, 360, 861, 1485, 482, 899, 1274, 1376, 1606, 382, 154, 1170, 512, 189, 1391, 1360, 963, 719, 1547, 198, 365, 921, 17, 1545, 732, 1325, 1361, 1064, 1465, 1347, 680, 208, 701, 391, 862, 1297, 653, 10, 34, 434, 1462, 803, 1019, 1244, 1033, 597)
    
    df = df.loc[~df.index.isin(large_files)]

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
    parser.add_argument("--as_train", type=bool, help="Обрабатывать как трейновый датасет")
    parser.add_argument("-N", default=1, type=int, help="Во солько раз увел. датасет")
    args = parser.parse_args()

    assert args.src_path.exists()
    assert args.N > 0
    args.dir_path.mkdir(parents=True, exist_ok=True)

    main(args.src_path, args.dir_path, args.N, args.as_train)
