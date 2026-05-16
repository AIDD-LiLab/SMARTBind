import numpy as np

def tanimoto_similarity(one_hot_str1, one_hot_str2):
    """
    Calculate the Tanimoto similarity between two one-hot strings

    Parameters
    ----------
    one_hot_str1: str
        One-hot string fingerprint of molecule 1
    one_hot_str2: str
        One-hot string fingerprint of molecule 2
    Returns
    -------
    similarity: float
        Tanimoto similarity between the two fingerprints
    """
    arr1 = np.array([int(bit) for bit in one_hot_str1])
    arr2 = np.array([int(bit) for bit in one_hot_str2])
    intersection = np.sum(arr1 & arr2)
    union = np.sum(arr1 | arr2)
    similarity = intersection / union
    return similarity


def top_decoys(decoy_lib, top_num, anchor_fp, tanimoto_threshold=0.5):
    """
    Find the top similar fingerprints in the decoy library

    Parameters
    ----------
    decoy_lib: list of str
        List of one-hot string fingerprints for decoy molecules
    top_num: int
        Number of top similar decoys to return
    anchor_fp: str
        One-hot string fingerprint of the anchor molecule
    tanimoto_threshold: float
        Tanimoto similarity threshold to filter decoys
    Returns
    -------
    top_similar_fp: list of int
    """
    similarity_dict = {}
    for i in range(len(decoy_lib)):
        similarity = tanimoto_similarity(anchor_fp, decoy_lib[i])
        similarity_dict[i] = similarity

    sorted_similarity_dict = sorted(similarity_dict, key=similarity_dict.get, reverse=True)
    # only keep those below the threshold to build a decoy set
    sorted_similarity_dict = [i for i in sorted_similarity_dict if similarity_dict[i] <= tanimoto_threshold]
    top_similar_index = sorted_similarity_dict[:top_num]
    top_similar_fp = [decoy_lib[i] for i in top_similar_index]
    return top_similar_fp
