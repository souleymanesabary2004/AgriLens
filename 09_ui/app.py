"""
app.py - E3 : interface Streamlit d'AgriLens.

La page ne calcule rien : elle envoie la photo a l'API (POST /predict) et affiche la reponse.
L'API doit donc tourner dans un autre terminal.

Lancer depuis la racine du projet :
    streamlit run 09_ui/app.py
"""
import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("AGRILENS_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AgriLens", page_icon="🌿")
st.title("AgriLens")
st.caption("Reconnaissance de maladies de cultures (tomate, maïs, cacao) - 17 classes. "
           "Prototype : ne remplace pas un agronome.")


def api_health():
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


def render_result(out):
    """Affiche la reponse de l'API : message vert (accepte) ou orange (a verifier par un humain)."""
    conf = out["confidence"]
    if out["abstained"]:
        st.warning(f"**À vérifier par un humain.** Le modèle n'est pas assez sûr "
                   f"(confiance {conf:.0%}, seuil {out['threshold']:.0%}).")
        st.write(f"Simple hypothèse, à ne pas considérer comme un diagnostic : *{out['display_fr']}*")
    else:
        st.success(f"**{out['display_fr']}**  \nConfiance : {conf:.1%}")
    st.progress(min(max(conf, 0.0), 1.0), text=f"Confiance : {conf:.1%}")
    top = pd.DataFrame(out["top"]).set_index("label")
    st.write("Les trois classes les plus probables :")
    st.bar_chart(top["probability"])


# ---------- barre latérale : état du service ----------
with st.sidebar:
    st.subheader("État du service")
    health = api_health()
    if health and health.get("model_loaded"):
        st.success("API active, modèle chargé")
        st.write(f"Seuil d'abstention : **{health['threshold']:.2f}**")
    elif health:
        st.warning("API active mais modèle non chargé")
    else:
        st.error("API injoignable")
        st.write(f"Adresse essayée : `{API_URL}`")
        st.write("Lance l'API dans un autre terminal :")
        st.code("uvicorn main:app --app-dir 08_api --port 8000")
    st.divider()
    st.write("Si la confiance du modèle est sous le seuil, l'image est renvoyée à un humain "
             "au lieu d'être classée.")

# ---------- zone principale ----------
def render_fiche(f):
    """E4 : fiche sourcee de la maladie reconnue (ou message fixe pour une plante saine)."""
    if f["kind"] == "healthy":
        st.info(f["message"])
    elif f["kind"] == "fiche":
        with st.expander(f"Fiche : {f['titre']}", expanded=True):
            for name, text in f["sections"].items():
                st.markdown(f"**{name}**")
                st.markdown(text)
            st.markdown("**Sources :** " + " · ".join(f"[{x['titre']}]({x['url']})" for x in f["sources"]))
            st.caption(f["message"])
    else:
        st.caption(f["message"])


def analyse(uploaded):
    """Appelle l'API une fois par photo (resultat garde en memoire tant que la photo ne change pas)."""
    key = hash(uploaded.getvalue())
    if st.session_state.get("photo_key") != key:
        st.session_state["photo_key"] = key
        st.session_state["photo_out"] = None
        st.session_state["photo_fiche"] = None
        st.session_state["photo_err"] = None
        try:
            with st.spinner("Analyse en cours..."):
                resp = httpx.post(f"{API_URL}/predict", timeout=60,
                                  files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
                if resp.status_code != 200:
                    detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
                    st.session_state["photo_err"] = f"L'API a refusé la requête ({resp.status_code}) : {detail}"
                else:
                    out = resp.json()
                    st.session_state["photo_out"] = out
                    if not out["abstained"]:        # pas de fiche automatique si le modele doute
                        fr = httpx.get(f"{API_URL}/fiche", params={"label": out["label"]}, timeout=10)
                        if fr.status_code == 200:
                            st.session_state["photo_fiche"] = fr.json()
        except httpx.HTTPError:
            st.session_state["photo_err"] = "Impossible de joindre l'API. Vérifie qu'elle tourne (voir la barre latérale)."


uploaded = st.file_uploader("Dépose une photo de feuille ou de cabosse", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    st.image(uploaded, caption=uploaded.name)
    analyse(uploaded)
    if st.session_state.get("photo_err"):
        st.error(st.session_state["photo_err"])
    elif st.session_state.get("photo_out"):
        render_result(st.session_state["photo_out"])
        if st.session_state.get("photo_fiche"):
            render_fiche(st.session_state["photo_fiche"])
        elif st.session_state["photo_out"]["abstained"]:
            st.caption("Photo renvoyée à un humain : pas de fiche affichée automatiquement. "
                       "Tu peux poser une question plus bas.")
else:
    st.session_state.pop("photo_key", None)
    st.info("Dépose une image pour commencer.")

# ---------- E4 : questions sur le corpus de fiches (recherche + citations, pas de texte invente) ----------
st.divider()
st.subheader("Poser une question sur une maladie")
st.caption("Je renvoie des passages de fiches sourcées. Si je n'ai rien de fiable, je le dis.")
question = st.text_input("Ta question (en français)", placeholder="Ex. : feuilles de tomate enroulées et jaunes avec des mouches blanches")
if st.button("Chercher") and question.strip():
    try:
        resp = httpx.post(f"{API_URL}/ask", json={"question": question, "top_k": 3}, timeout=30)
    except httpx.HTTPError:
        st.error("Impossible de joindre l'API.")
    else:
        if resp.status_code != 200:
            st.error(f"L'API a refusé la requête ({resp.status_code}).")
        else:
            out = resp.json()
            if not out["answered"]:
                st.warning(out["message"])
            else:
                for p in out["results"]:
                    with st.expander(f"{p['titre']} — {p['section']} (score {p['score']:.2f})", expanded=(p is out["results"][0])):
                        st.markdown(p["texte"])
                        st.markdown("**Sources :** " + " · ".join(f"[{s['titre']}]({s['url']})" for s in p["sources"]))
                st.caption(out["message"])
