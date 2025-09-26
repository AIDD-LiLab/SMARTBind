
def rank_eval(match_distance, decoy_distance_list):
    """
    Calculate the rank percentile of the match distance among the decoy distance list
    Larger distance means more similar
    """
    rank = 0
    for decoy_distance in decoy_distance_list:
        if match_distance <= decoy_distance:
            rank += 1
    return (len(decoy_distance_list) + 1 - rank) / (len(decoy_distance_list) + 1)
