import os
import random

import numpy as np
import torch


SEED_PADRAO = 42


def set_seed(seed: int = SEED_PADRAO) -> int:
    """Fixa as seeds de random, numpy e torch para tornar as execucoes
    reproduziveis. Retorna a seed usada.

    Usado na avaliacao comparativa: combinado com decodificacao
    deterministica (greedy), garante que cada numero seja 100% reproduzivel.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
