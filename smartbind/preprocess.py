import pickle

import openbabel.pybel as pybel


"""
Function to preprocess the dictionary data from the raw data
"""
def build_rna_smol_map_from_raw_dict(raw_dict):
    seq_to_chain_ids = {}
    seq_to_ligands = {}

    for _, complexes in raw_dict.items():
        for _, sample in complexes.items():
            rna_seq = sample['rna_chain_sequence']
            seq_id = f"{sample['pdb_id']}_{sample['rna_chain_id']}"
            ligand_id = sample['ligand_id']

            if rna_seq not in seq_to_chain_ids:
                seq_to_chain_ids[rna_seq] = []
            if rna_seq not in seq_to_ligands:
                seq_to_ligands[rna_seq] = []

            seq_to_chain_ids[rna_seq].append(seq_id)
            seq_to_ligands[rna_seq].append(ligand_id)

    seq_id_smol_map = []
    for rna_seq, seq_id_list in seq_to_chain_ids.items():
        unique_seq_ids = list(dict.fromkeys(seq_id_list))
        unique_ligands = list(dict.fromkeys(seq_to_ligands[rna_seq]))
        seq_id_smol_map.append((unique_seq_ids, unique_ligands))

    return seq_id_smol_map


def build_val_test_set(fold_num=0, dict_path=str, topk_decoy=100, return_rna_smol_map=False):
    raw_dict = load_dict(dict_path)
    rna_smol_map = build_rna_smol_map_from_raw_dict(raw_dict)

    val_dict, train_dict = split_train_test(raw_dict, fold_num)

    val_rna_seq_list, val_match_smol_list, val_complex_id_list, val_match_smol_id_list, val_contact_index, \
        val_match_smol_fp_list = parse_dict(val_dict, topk_decoy=topk_decoy)
    train_rna_seq_list, train_match_smol_list, train_complex_id_list, train_match_smol_id_list, train_contact_index, \
        train_match_smol_fp_list = parse_dict(train_dict, topk_decoy=topk_decoy)

    out = (val_rna_seq_list, val_match_smol_list, val_complex_id_list, val_match_smol_id_list, val_contact_index,
           val_match_smol_fp_list, train_rna_seq_list, train_match_smol_list, train_complex_id_list,
            train_match_smol_id_list, train_contact_index, train_match_smol_fp_list)
    if return_rna_smol_map:
        return out + (rna_smol_map,)
    return out


def load_dict(file_path):
    with open(file_path, 'rb') as handle:
        loaded_dict = pickle.load(handle)
    return loaded_dict


def split_train_test(dict, fold_num):
    train_dict, test_dict = {}, {}
    for key in dict.keys():
        for key2 in dict[key].keys():
            if dict[key][key2]['train_split'][fold_num]:
                train_dict[key2] = dict[key][key2]
                train_dict[key2]['complex_id'] = key
            else:
                test_dict[key2] = dict[key][key2]
                test_dict[key2]['complex_id'] = key

    return test_dict, train_dict


def parse_dict(dict, topk_decoy=100):
    """
    Decipher the dict to return lists:
    - rna sequence
    - matched ligand
    - complex id
    - ligand id
    - contact index
    """
    rna_seq_list, match_smol_list, complex_id_list, match_smol_id_list, contact_index, match_smol_fp_list = \
        [], [], [], [], [], []
    for key in dict.keys():
        rna_seq_list.append((f'{dict[key]["pdb_id"]}_{dict[key]["rna_chain_id"]}',
                             dict[key]['rna_chain_sequence']))
        match_smol_list.append(dict[key]['downloaded_ligand_smiles'])
        match_smol_fp_list.append((dict[key]['ligand_id'], dict[key]['download_fp2'], dict[key]['decoy_smiles'][:topk_decoy]))
        complex_id_list.append(dict[key]['pdb_id'])
        match_smol_id_list.append(dict[key]['ligand_id'])
        contact_index.append(dict[key]['contact_map_1d'])

    return rna_seq_list, match_smol_list, complex_id_list, match_smol_id_list, contact_index, match_smol_fp_list


def convert_smiles_to_pf2(smiles):
    mol = pybel.readstring("smi", smiles)
    fp2_bits = mol.calcfp(fptype="FP2").bits
    fp2_bit_one_hot = [0] * 1024
    for bit in fp2_bits:
        fp2_bit_one_hot[bit] = 1
    return fp2_bit_one_hot
