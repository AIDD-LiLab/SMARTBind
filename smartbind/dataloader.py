from torch.utils.data import Dataset, DataLoader
import random


class RLDataset(Dataset):
    def __init__(self,
                 rna_sequences,
                 rna_sequences_names,
                 match_smols,
                 match_smols_names,
                 match_pair_dict=None,
                 non_binding_index_list=None,
                 decoyfinder_list=None,
                 random_decoy_num=12,
                 is_val=False,
                 replace_ratio=0.15,
                 augmentation_factor=3,
                 replace_res_type=None
                 ):
        if non_binding_index_list is None:
            non_binding_index_list = []
        if replace_res_type is None:
            replace_res_type = ['R', 'Y', 'K', 'M', 'S', 'W', 'B', 'D', 'H', 'V', 'N', '-']
        self.rna_sequences = rna_sequences
        self.match_smols = match_smols
        self.random_decoy_num = random_decoy_num
        # load decoy list from match_smol_list
        self.decoy_list = match_smols.copy()
        self.decoyfinder_list = decoyfinder_list
        self.is_val = is_val
        self.rna_sequences_names = rna_sequences_names
        self.match_smols_names = match_smols_names
        self.replace_ratio = replace_ratio
        self.augmentation_factor = augmentation_factor
        self.replace_type = replace_res_type
        self.non_binding_index_list = non_binding_index_list
        # make sure the decoy in not bind to the rna (when rna has multiple binding sites case)
        self.match_dict = match_pair_dict

    def __len__(self):
        return len(self.rna_sequences)

    def __getitem__(self, index):
        rna_sequence = self.rna_sequences[index]
        rna_sequence_name = self.rna_sequences_names[index]
        match_smol = self.match_smols[index]
        match_smol_name = self.match_smols_names[index]
        # get random decoy smiles from decoy list where number is random_decoy_num, and will not select match_smol
        if self.decoyfinder_list is not None:
            temp_decoy_list = self.decoyfinder_list[index].copy()
        else:
            temp_decoy_list = self.decoy_list.copy()

        if self.is_val is False and self.match_dict is not None:
            decoy_list_removed = []
            # remove decoy that bind to the rna from temp_decoy_list
            for key, value in self.match_dict.items():
                if key == rna_sequence_name:
                    for val in value:
                        # get index of i in self.match_smols_names
                        removes_index_list = [i for i, x in enumerate(self.match_smols_names) if x == val]
                        # remove those index from temp_decoy_list
                        for i in removes_index_list:
                            decoy_list_removed.append(self.decoy_list[i])

            # decoy_list_removed = list(set(decoy_list_removed))
            for i in decoy_list_removed:
                while i in temp_decoy_list:
                    temp_decoy_list.remove(i)

        while match_smol in temp_decoy_list:
            temp_decoy_list.remove(match_smol)

        if self.is_val:
            decoy_smols = temp_decoy_list
        else:
            decoy_smols = random.sample(temp_decoy_list, self.random_decoy_num)

        if self.is_val:
            # return contact position list in list [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
            contact_index_out = self.non_binding_index_list[index]

            return rna_sequence, match_smol, decoy_smols, rna_sequence_name, match_smol_name, contact_index_out

        # data augmentation
        rna_sequence_str = rna_sequence[1]
        contact_index = self.non_binding_index_list[index]
        try:
            if random.random() > 1 / (self.augmentation_factor + 1):
                # random replace base with other base in contact index list with ratio replace_ratio
                for i in range(len(rna_sequence_str)):
                    if contact_index[i] == 0:
                        if random.random() < self.replace_ratio:
                            rna_sequence_str = rna_sequence_str[:i] + random.choice(
                                self.replace_type) + rna_sequence_str[i + 1:]
        except Exception as e:
            print(f'Issue happened at dataloader with pairs: {rna_sequence_name} {match_smol_name}.')
            print(e)

        rna_sequence = (rna_sequence[0], rna_sequence_str)
        return rna_sequence, match_smol, decoy_smols, rna_sequence_name, match_smol_name, contact_index


class Collate:
    def __call__(self, batch):
        rna_sequences = [item[0] for item in batch]
        match_smols = [item[1] for item in batch]
        decoy_smols_list = [item[2] for item in batch]
        rna_sequences_names = [item[3] for item in batch]
        match_smols_names = [item[4] for item in batch]
        contact_index_list = [item[5] for item in batch]
        return rna_sequences, match_smols, decoy_smols_list, rna_sequences_names, match_smols_names, contact_index_list


class RLDataLoader:
    def __init__(self, dataset, batch_size=1, num_workers=4, if_shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate = Collate()
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     collate_fn=self.collate,
                                     num_workers=num_workers,
                                     shuffle=if_shuffle,
                                     drop_last=False)

    def get_batches(self):
        return iter(self.dataloader)
