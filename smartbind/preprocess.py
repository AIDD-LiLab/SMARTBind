import pickle

import openbabel.pybel as pybel


"""
Function to preprocess the dictionary data from the raw data
"""

def build_val_test_set(fold_num=0, dict_path=str):
    raw_dict = load_dict(dict_path)

    val_dict, train_dict = split_train_test(raw_dict, fold_num)

    val_rna_seq_list, val_match_smol_list, val_complex_id_list, val_match_smol_id_list, val_contact_index, \
        val_match_smol_fp_list = parse_dict(val_dict)
    train_rna_seq_list, train_match_smol_list, train_complex_id_list, train_match_smol_id_list, train_contact_index, \
        train_match_smol_fp_list = parse_dict(train_dict)

    # build match pair dict: {}
    match_pair_dict = {}
    for i in range(len(train_complex_id_list)):
        if train_complex_id_list[i] not in match_pair_dict:
            match_pair_dict[train_complex_id_list[i]] = [train_match_smol_id_list[i]]
        else:
            match_pair_dict[train_complex_id_list[i]].append(train_match_smol_id_list[i])

    return val_rna_seq_list, val_match_smol_list, val_complex_id_list, val_match_smol_id_list, val_contact_index, \
        val_match_smol_fp_list, train_rna_seq_list, train_match_smol_list, train_complex_id_list, \
        train_match_smol_id_list, train_contact_index, train_match_smol_fp_list, match_pair_dict


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


def parse_dict(dict):
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
        match_smol_fp_list.append(dict[key]['download_fp2'])
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
