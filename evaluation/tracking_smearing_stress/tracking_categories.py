"""Current exclusive jet-tag definitions used by the Rb tracking study."""
import numpy as np


B_THRESHOLDS = (0.396, 0.601, 0.855)
C_THRESHOLDS = (0.524, 0.605)
TAG_LABELS = ("Q", "S", "L", "C", "X", "U")
TAG_INDEX = {tag: index for index, tag in enumerate(TAG_LABELS)}
PAIR_LABELS = (
    "QQ", "SS", "LL", "CC", "XX", "QS", "LQ", "CQ", "QX",
    "LS", "CS", "SX", "CL", "LX", "CX", "QU", "SU", "LU", "CU", "UX",
)


def assign_tags(b_score, c_score):
    tags = np.full(b_score.shape, TAG_INDEX["U"], dtype=np.int64)
    tags[(b_score > B_THRESHOLDS[2]) & (b_score <= 1.0)] = TAG_INDEX["Q"]
    tags[(b_score > B_THRESHOLDS[1]) & (b_score <= B_THRESHOLDS[2])] = TAG_INDEX["S"]
    tags[(b_score <= B_THRESHOLDS[0]) & (c_score > C_THRESHOLDS[1]) & (c_score <= 1.0)] = TAG_INDEX["C"]
    tags[(b_score <= B_THRESHOLDS[0]) & (c_score <= C_THRESHOLDS[0])] = TAG_INDEX["X"]
    tags[(b_score > B_THRESHOLDS[0]) & (b_score <= B_THRESHOLDS[1])] = TAG_INDEX["L"]
    return tags


def pair_lookup():
    lookup = np.full((len(TAG_LABELS), len(TAG_LABELS)), -1, dtype=np.int64)
    pair_index = {pair: index for index, pair in enumerate(PAIR_LABELS)}
    for first, first_index in TAG_INDEX.items():
        for second, second_index in TAG_INDEX.items():
            pair = "".join(sorted((first, second)))
            if pair in pair_index:
                lookup[first_index, second_index] = pair_index[pair]
    return lookup
