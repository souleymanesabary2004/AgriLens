"""
main.py - E3 : API REST du modele AgriLens (FastAPI).

Endpoints :
  GET  /health   -> le service est-il vivant, le modele est-il charge, quel seuil d'abstention
  POST /predict  -> recoit une photo, renvoie classe, confiance et decision (accepte / a verifier)
  GET  /fiche    -> E4 : fiche sourcee de la classe predite (ou message fixe pour une classe saine)
  POST /ask      -> E4 : question en francais, renvoie des passages SOURCES du corpus (ou un refus)
  /api/v1/...    -> E5 : catalogue de donnees et lineage (voir src/agrilens/catalog_api.py)

Lancer depuis la racine du projet :
    uvicorn main:app --app-dir 08_api --port 8000
Puis ouvrir http://127.0.0.1:8000/docs pour tester dans le navigateur.
"""
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_BYTES = 10 * 1024 * 1024            # 10 Mo
MAX_PIXELS = 50_000_000                 # protege contre les images "bombes" (petit fichier, enorme image)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


# ---------- Contrat de reponse (Pydantic) : forme exacte de ce que l'API renvoie ----------
class TopItem(BaseModel):
    label: str
    probability: float


class Prediction(BaseModel):
    class_id: int
    label: str
    display_fr: str
    confidence: float
    abstained: bool
    decision: str                       # "accepte" ou "a_verifier_par_un_humain"
    threshold: float
    top: List[TopItem]


class AskIn(BaseModel):
    question: str
    top_k: int = 3


class Passage(BaseModel):
    doc_id: str
    titre: str
    section: str
    score: float
    texte: str
    sources: List[dict]


class AskOut(BaseModel):
    answered: bool
    reason: str
    message: str
    best_score: float
    results: List[Passage]


class FicheOut(BaseModel):
    kind: str                           # "fiche", "healthy" ou "none"
    doc_id: str | None = None
    titre: str | None = None
    sections: dict
    sources: List[dict]
    message: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    threshold: float | None = None


def load_real_predictor():
    from agrilens.inference import Predictor
    return Predictor(
        model_path=ROOT / "models/head_mobilenetv2_v1.pt",
        threshold_path=ROOT / "05_deep_learning/results/abstention_threshold.json",
        manifest_path=ROOT / "01_data/metadata/split_manifest_v2.csv",
    )


def load_real_retriever():
    """E4 : charge le corpus de fiches et le seuil de refus (None si le corpus est absent)."""
    from agrilens.rag import Retriever, load_threshold
    corpus = ROOT / "07_rag_genai/corpus"
    if not corpus.exists():
        return None
    return Retriever(corpus, threshold=load_threshold(ROOT / "07_rag_genai/rag_threshold.json"))


def create_app(predictor=None, retriever="auto", catalog_state="auto") -> FastAPI:
    """Construit l'API. `predictor` / `retriever` / `catalog_state` permettent aux tests d'injecter des faux."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Au demarrage : le modele est charge UNE fois et reste en memoire.
        app.state.predictor = predictor if predictor is not None else load_real_predictor()
        app.state.retriever = load_real_retriever() if retriever == "auto" else retriever
        # E5 : catalogue et lineage (None si 11_catalog/catalog.json n'a pas encore ete genere)
        if catalog_state == "auto":
            from agrilens.catalog_api import load_state
            app.state.catalog_state = load_state(ROOT / "11_catalog/catalog.json", ROOT / "11_catalog/lineage.json")
        else:
            app.state.catalog_state = catalog_state
        yield

    app = FastAPI(title="AgriLens API", version="0.1.0",
                  description="Classification de maladies de cultures (17 classes) avec abstention. "
                              "Prototype local : ne pas utiliser pour une decision agronomique reelle.",
                  lifespan=lifespan)

    @app.get("/health", response_model=Health)
    def health(request: Request):
        p = getattr(request.app.state, "predictor", None)
        return Health(status="ok" if p is not None else "model_not_loaded", model_loaded=p is not None,
                      threshold=getattr(p, "threshold", None))

    @app.post("/predict", response_model=Prediction)
    async def predict(request: Request, file: UploadFile = File(...)):
        p = getattr(request.app.state, "predictor", None)
        if p is None:
            raise HTTPException(503, "Modele non charge.")
        # Barriere 1 : type declare par le client (peut mentir, donc pas suffisant)
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(415, f"Type non supporte ({file.content_type}). Attendu : JPEG, PNG ou WebP.")
        # Barriere 2 : taille (on lit au plus MAX_BYTES + 1 octet)
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"Fichier trop volumineux (maximum {MAX_BYTES // (1024 * 1024)} Mo).")
        # Barriere 3 : le contenu est-il vraiment une image lisible ?
        try:
            img = Image.open(io.BytesIO(data))
            if img.width * img.height > MAX_PIXELS:
                raise HTTPException(413, "Image trop grande (nombre de pixels excessif).")
            img.load()
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(400, "Fichier illisible : ce n'est pas une image valide.")
        # Le calcul est lourd : on le sort de la boucle principale pour ne pas bloquer les autres requetes.
        return await run_in_threadpool(p.predict, img)

    @app.get("/fiche", response_model=FicheOut)
    def fiche(request: Request, label: str):
        r = getattr(request.app.state, "retriever", None)
        if r is None:
            raise HTTPException(503, "Corpus non charge.")
        if not label.strip() or len(label) > 200:
            raise HTTPException(422, "Parametre label invalide.")
        return r.fiche_for_label(label)

    @app.post("/ask", response_model=AskOut)
    async def ask(request: Request, body: AskIn):
        r = getattr(request.app.state, "retriever", None)
        if r is None:
            raise HTTPException(503, "Corpus non charge.")
        if len(body.question) > 500:
            raise HTTPException(422, "Question trop longue (500 caracteres maximum).")
        return await run_in_threadpool(r.search, body.question, max(1, min(body.top_k, 5)))

    # E5 : routes /api/v1/catalog et /api/v1/lineage
    from agrilens.catalog_api import router as catalog_router
    app.include_router(catalog_router)

    return app


app = create_app()