"""
extract_embeddings.py : calcule UNE FOIS les "embeddings" de toutes les images avec MobileNetV2
pre-entraine sur ImageNet (backbone GELE). Un embedding = 1280 nombres qui resument ce que le
reseau "voit" dans l'image.

Pourquoi ainsi : le backbone est gele (ses poids ne changent pas), donc sa sortie pour une image
donnee est toujours la meme. On la calcule une fois, on la range dans un fichier Parquet, puis on
entraine la derniere couche en quelques secondes, autant de fois qu'on veut.
(Suite Guide S2.9 ; "cache d'embeddings" du Guide S2, utilise ici car pas de GPU.)

Usage (a la racine du projet) :
    python scripts\\extract_embeddings.py --n 64      # test rapide sur 64 images
    python scripts\\extract_embeddings.py             # les 26 397 images
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SEED = 2026
IMAGENET_MEAN = (0.485, 0.456, 0.406)   # statistiques du pre-entrainement ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    """Lit une image a partir de son chemin relatif a la racine du projet."""

    def __init__(self, paths, size):
        from torchvision import transforms as T
        self.paths = list(paths)
        self.tf = T.Compose([
            T.Resize((size, size)),          # on redimensionne SANS recadrer : on garde toute la feuille
            T.ToTensor(),                    # image -> tenseur 3 x H x W, valeurs entre 0 et 1
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), i


def build_backbone(pretrained=True):
    """MobileNetV2 sans sa derniere couche de classification : sortie = vecteur de 1280 nombres."""
    from torchvision import models
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    backbone = torch.nn.Sequential(net.features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    backbone.eval()                          # IMPORTANT : BatchNorm en mode eval (statistiques figees)
    for p in backbone.parameters():
        p.requires_grad = False              # backbone gele
    return backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="01_data/metadata/split_manifest_v2.csv")
    ap.add_argument("--out", default="01_data/processed/embeddings_mobilenetv2_v1.parquet")
    ap.add_argument("--n", type=int, default=0, help="echantillon aleatoire pour tester (0 = toutes les images)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--no-pretrained", action="store_true", help="TEST DU CODE UNIQUEMENT : poids aleatoires")
    a = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(a.manifest)
    if "filepath_rel" not in df.columns:
        raise SystemExit("Colonne filepath_rel absente : lance d'abord scripts\\add_filepath_rel.py")
    if a.n:
        df = df.sample(n=a.n, random_state=SEED).reset_index(drop=True)
        a.out = a.out.replace(".parquet", "_test.parquet")

    backbone = build_backbone(pretrained=not a.no_pretrained)
    n_param = sum(p.numel() for p in backbone.parameters())
    print(f"MobileNetV2 {'ImageNet' if not a.no_pretrained else 'POIDS ALEATOIRES (test)'} | "
          f"{n_param:,} parametres, tous geles | mode eval = {not backbone.training}")
    print(f"{len(df)} images | taille {a.size}x{a.size} | lot {a.batch} | {a.workers} workers | "
          f"{torch.get_num_threads()} threads torch | pas de GPU : CPU")

    loader = DataLoader(ImageDataset(df["filepath_rel"], a.size), batch_size=a.batch,
                        shuffle=False, num_workers=a.workers)
    emb = np.zeros((len(df), 1280), dtype=np.float32)

    t0 = time.time()
    done = 0
    n_batches = len(loader)
    with torch.inference_mode():             # pas de calcul de gradients : plus rapide, moins de memoire
        for b, (x, idx) in enumerate(loader, start=1):
            emb[idx.numpy()] = backbone(x).numpy()
            done += len(idx)
            if b == 1 or b % 20 == 0 or b == n_batches:
                el = time.time() - t0
                eta = el / done * (len(df) - done)
                print(f"  {done}/{len(df)} images | {el/60:.1f} min ecoulees | reste ~{eta/60:.1f} min", flush=True)

    out = pd.DataFrame(emb, columns=[f"e_{i}" for i in range(1280)])
    out.insert(0, "filepath_rel", df["filepath_rel"].values)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"{len(out)} images, {emb.shape[1]} dimensions, {time.time()-t0:.0f} s, "
          f"NaN = {int(np.isnan(emb).sum())}, min={emb.min():.2f}, max={emb.max():.2f}")
    print(f"ecrit : {a.out}")


if __name__ == "__main__":      # obligatoire sous Windows pour les workers du DataLoader
    main()
