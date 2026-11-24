"""Decoupage du corpus en chunks pour le RAG v2 (P1.2).

Regles (decisions validees en P1.2) :
- une fiche est decoupee par section ``## `` ; la section « Sources » n'est
  jamais indexee, ses lignes deviennent une metadonnee de chaque chunk ;
- une section de ``max_chars`` caracteres ou moins forme un seul chunk ;
- une section plus longue est coupee entre les puces (jamais au milieu d'une
  puce), et la derniere puce de chaque partie est repetee au debut de la suivante ;
- le texte de chaque chunk commence par « Titre de la fiche : Section ».
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = (
    "id",
    "titre",
    "culture",
    "agent",
    "classe_modele",
    "label",
    "date_consultation",
)
SOURCES_TITLE = "Sources"
_BULLET = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Parametres du decoupage."""

    max_chars: int = 800
    overlap: bool = True

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError("max_chars doit valoir au moins 100")


@dataclass(frozen=True)
class Chunk:
    """Un morceau de fiche pret a etre indexe."""

    chunk_id: str
    doc_id: str
    titre: str
    section: str
    part: int
    n_parts: int
    contenu: str
    texte: str
    label: str
    classe_modele: str
    culture: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Fiche:
    """Une fiche lue depuis le corpus."""

    header: dict[str, str]
    sections: tuple[tuple[str, str], ...]
    sources: tuple[str, ...]


def _slug(texte: str) -> str:
    """Transforme un titre en identifiant ASCII (sans accents)."""
    sans_accents = unicodedata.normalize("NFKD", texte)
    ascii_only = sans_accents.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")


def parse_fiche(path: Path) -> Fiche:
    """Lit une fiche Markdown et valide son en-tete."""
    texte = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    parties = texte.split("---")
    if len(parties) < 3:
        raise ValueError(f"{path.name}: en-tete '---' introuvable")
    header: dict[str, str] = {}
    for ligne in parties[1].strip().splitlines():
        if ":" in ligne:
            cle, _, valeur = ligne.partition(":")
            header[cle.strip()] = valeur.strip()
    manquants = [c for c in REQUIRED_FIELDS if not header.get(c)]
    if manquants:
        raise ValueError(f"{path.name}: champs manquants {manquants}")
    corps = "---".join(parties[2:])
    sections: list[tuple[str, str]] = []
    sources: tuple[str, ...] = ()
    for bloc in re.split(r"^## ", corps, flags=re.MULTILINE)[1:]:
        titre, _, contenu = bloc.partition("\n")
        titre = titre.strip()
        contenu = contenu.strip()
        if titre.startswith(SOURCES_TITLE):
            sources = tuple(
                _BULLET.sub("", ln).strip()
                for ln in contenu.splitlines()
                if _BULLET.match(ln)
            )
        else:
            sections.append((titre, contenu))
    return Fiche(header=header, sections=tuple(sections), sources=sources)


def _blocks(contenu: str) -> list[str]:
    """Separe un texte en blocs : une puce (avec ses lignes de suite) ou un paragraphe."""
    blocs: list[list[str]] = []
    nouveau = True
    for ligne in contenu.splitlines():
        if not ligne.strip():
            nouveau = True
            continue
        if nouveau or _BULLET.match(ligne):
            blocs.append([ligne.rstrip()])
        else:
            blocs[-1].append(ligne.rstrip())
        nouveau = False
    return ["\n".join(b) for b in blocs]


def _cut_oversized(bloc: str, max_chars: int) -> list[str]:
    """Coupe un bloc trop long par phrases, puis par mots en dernier recours."""
    if len(bloc) <= max_chars:
        return [bloc]
    morceaux: list[str] = []
    courant = ""
    for phrase in _SENTENCE_END.split(bloc):
        while len(phrase) > max_chars:
            coupe = phrase.rfind(" ", 0, max_chars)
            coupe = coupe if coupe > 0 else max_chars
            if courant:
                morceaux.append(courant)
                courant = ""
            morceaux.append(phrase[:coupe].strip())
            phrase = phrase[coupe:].strip()
        candidat = f"{courant} {phrase}".strip() if courant else phrase
        if len(candidat) > max_chars:
            morceaux.append(courant)
            courant = phrase
        else:
            courant = candidat
    if courant:
        morceaux.append(courant)
    return morceaux


def split_section(contenu: str, config: ChunkingConfig) -> list[str]:
    """Decoupe le contenu d'une section en parties de ``max_chars`` caracteres au plus."""
    if len(contenu) <= config.max_chars:
        return [contenu]
    blocs: list[str] = []
    for bloc in _blocks(contenu):
        blocs.extend(_cut_oversized(bloc, config.max_chars))
    parties: list[list[str]] = []
    courant: list[str] = []
    for bloc in blocs:
        taille = len("\n".join(courant + [bloc]))
        if courant and taille > config.max_chars:
            parties.append(courant)
            reprise = courant[-1]
            garde = config.overlap and len(reprise) + 1 + len(bloc) <= config.max_chars
            courant = [reprise] if garde else []
        courant.append(bloc)
    if courant:
        parties.append(courant)
    return ["\n".join(p) for p in parties]


def chunk_fiche(path: Path, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Produit les chunks d'une fiche."""
    cfg = config or ChunkingConfig()
    fiche = parse_fiche(path)
    h = fiche.header
    chunks: list[Chunk] = []
    for section, contenu in fiche.sections:
        parties = split_section(contenu, cfg)
        for i, partie in enumerate(parties, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{h['id']}::{_slug(section)}::{i}",
                    doc_id=h["id"],
                    titre=h["titre"],
                    section=section,
                    part=i,
                    n_parts=len(parties),
                    contenu=partie,
                    texte=f"{h['titre']} : {section}\n{partie}",
                    label=h["label"],
                    classe_modele=h["classe_modele"],
                    culture=h["culture"],
                    sources=fiche.sources,
                )
            )
    return chunks


def chunk_corpus(corpus_dir: Path, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Produit les chunks de toutes les fiches ``*.md`` du dossier, en ordre stable."""
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_fiche(path, config))
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("identifiants de chunks en double")
    logger.info("%d chunks produits depuis %s", len(chunks), corpus_dir)
    return chunks
