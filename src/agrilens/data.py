from pathlib import Path

import pandas as pd

MANIFEST_PATH = Path(
    "01_data/metadata/split_manifest_v2.csv"
)


def load_manifest(split=None):
    """
    Charge le manifest officiel du projet.

    Parameters
    ----------
    split : str | None
        train, val ou None
    """

    df = pd.read_csv(MANIFEST_PATH)

    if split is None:
        return df

    if split == "test":
        raise ValueError(
            "Le split TEST est verrouillé pendant S2 et S3."
        )

    return df[df["split"] == split].copy()