from typing import List, Hashable, Dict, Tuple
import pandas as pd

def transition_matrix(states: List[Hashable]) -> pd.DataFrame:
    pairs = list(zip(states[:-1], states[1:]))
    if len(pairs) == 0:
        raise ValueError("Need at least 2 draws to compute transitions.")

    counts: Dict[Tuple[Hashable, Hashable], int] = {}
    for a, b in pairs:
        counts[(a, b)] = counts.get((a, b), 0) + 1

    from_states = sorted({a for a, _ in pairs})
    to_states = sorted({b for _, b in pairs})

    mat = pd.DataFrame(0, index=from_states, columns=to_states, dtype=int)
    for (a, b), c in counts.items():
        mat.at[a, b] = c

    # Row-normalize -> probabilities
    prob = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)
    prob.index.name = "from_state"
    return prob
