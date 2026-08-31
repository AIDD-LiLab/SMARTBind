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


def build_fp_matrix(fp_list):
    """
    Stack a list of one-hot fingerprints (str or list of int) into a bool matrix

    Parameters
    ----------
    fp_list: list of (str or list of int)
    Returns
    -------
    matrix: np.ndarray of bool, shape (n_fp, n_bits)
    """
    return np.array([[int(bit) for bit in fp] for fp in fp_list], dtype=bool)


def top_decoy_indices(decoy_matrix, anchor_fp, top_num, tanimoto_threshold=0.5):
    """
    Vectorized version of top_decoys that returns the indices of the selected
    decoys, so that the caller can pull both the fingerprints and the SMILES.

    Same selection rule as top_decoys: sort all decoys by Tanimoto similarity to
    the anchor in descending order, drop everything above the threshold, then
    keep the first `top_num`.

    Parameters
    ----------
    decoy_matrix: np.ndarray of bool, shape (n_decoy, n_bits)
        Decoy library fingerprints, see build_fp_matrix
    anchor_fp: str or list of int
        One-hot fingerprint of the anchor molecule
    top_num: int
        Number of decoys to return
    tanimoto_threshold: float
        Only decoys with similarity <= this value are eligible
    Returns
    -------
    indices: np.ndarray of int
    """
    anchor = np.array([int(bit) for bit in anchor_fp], dtype=bool)
    intersection = decoy_matrix @ anchor.astype(np.int32)
    union = decoy_matrix.sum(axis=1, dtype=np.int32) + int(anchor.sum()) - intersection
    # union == 0 means both fingerprints are all-zero; treat as not eligible
    similarity = np.divide(intersection, union, out=np.full(len(union), np.inf),
                           where=union > 0)
    order = np.argsort(-similarity, kind='stable')
    eligible = order[similarity[order] <= tanimoto_threshold]
    return eligible[:top_num]


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
