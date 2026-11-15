import os
import random
import numpy as np


def set_seed(seed: int = 2026):
    """
    Configure tous les générateurs aléatoires du projet.
    """

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed