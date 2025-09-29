# app_chat_rag_fr.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, time, requests
from typing import List, Dict
from enum import Enum

import numpy as np
import gradio as gr
from pymilvus import connections, Collection
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder

# ─────────────────────────────────────────────────────────────────────
# CONFIG (utilise idéalement des variables d'environnement)
# ─────────────────────────────────────────────────────────────────────
COLLECTION   = os.getenv("MILVUS_COLLECTION", "scientific_Maxime_assistant")

# Milvus / Zilliz (⚠️ mets ces deux valeurs en variables d'env dans la vraie vie)
URI          = os.getenv("ZILLIZ_URI", "https://in03-be569fdfbf79a63.serverless.gcp-us-west1.cloud.zilliz.com")
TOKEN        = os.getenv("ZILLIZ_TOKEN", "57e31c439ceb9b518959da6fca76cea446e8242cf47913e0febdaa6c66c945c882e68158b04486dd4f833e21a7789592158b7ef3")

# Embeddings / Rerank
EMB_MODEL    = os.getenv("EMB_MODEL_NAME", "BAAI/bge-m3")
USE_FP16     = os.getenv("USE_FP16", "true").lower() in {"1","true","yes","y"}
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() in {"1","true","yes","y"}
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM (Qwen via Ollama)
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen2.5:14b-instruct")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

# Port/host UI
#SERVER_NAME  = os.getenv("SERVER_NAME", "0.0.0.0")
#SERVER_PORT  = int(os.getenv("SERVER_PORT", "7860"))

# Port/host UI
# Par défaut : localhost et port choisi automatiquement
SERVER_NAME  = os.getenv("GRADIO_SERVER_NAME", os.getenv("SERVER_NAME", "127.0.0.1"))
_env_port    = os.getenv("GRADIO_SERVER_PORT", os.getenv("SERVER_PORT", "")).strip()
SERVER_PORT  = int(_env_port) if _env_port.isdigit() else None


# Sécurité minimale
if not URI or not TOKEN:
    raise SystemExit("❌ ZILLIZ_URI / ZILLIZ_TOKEN manquants. Ex: setx ZILLIZ_URI \"...\" && setx ZILLIZ_TOKEN \"...\"")

# ─────────────────────────────────────────────────────────────────────
# DÉTECTION TYPE DE QUESTION
# ─────────────────────────────────────────────────────────────────────
class QType(Enum):
    METHODS     = "méthodes"
    MATERIALS   = "matériaux"
    SOFTWARE    = "logiciels"
    COMPARISON  = "comparaison"
    DATA        = "données"
    LIMITS      = "limites"
    GENERAL     = "général"

def detect_qtype(q: str) -> QType:
    t = q.lower()
    if any(w in t for w in ["méthode","algorithme","algorithm","procédure","protocole"]): return QType.METHODS
    if any(w in t for w in ["matériau","propriété mécanique","module de young","hyperélast","visco"]): return QType.MATERIALS
    if any(w in t for w in ["logiciel","software","matlab","opensim","abaqus","ansys","comsol"]): return QType.SOFTWARE
    if any(w in t for w in ["compare","compar","point fort","point faible","avantage","désavantage"]): return QType.COMPARISON
    if any(w in t for w in ["donnée","in-vivo","in vivo","in-vitro","in vitro","expérimental","dataset"]): return QType.DATA
    if any(w in t for w in ["limitation","contrainte","problème","difficulté"]): return QType.LIMITS
    return QType.GENERAL

# ─────────────────────────────────────────────────────────────────────
# PROMPTS (FRANÇAIS FORCÉ)
# ─────────────────────────────────────────────────────────────────────
#PROMPTS = {
#    QType.METHODS: (
#        "Réponds en FRANÇAIS UNIQUEMENT.\n"
#        "Tu es un expert méthodologique français. "
#        "À partir des PASSAGES ci-dessous, liste chaque méthode/algorithme/protocole EXACT (nom complet), "
#        "les paramètres d'entrée/sortie, le logiciel (et version si présent), l'objectif précis et formules/équations si mentionnées.\n"
#        "Cite chaque item ainsi: (Auteur principal et al., Année — DOI — Section). "
#        "N'invente rien. Si un élément manque, écris « non indiqué ».\n\n"
#        "Question: {q}\n\nPASSAGES:\n{psg}\n"
#    ),
#    QType.MATERIALS: (
#        "Réponds en FRANÇAIS UNIQUEMENT.\n"
#        "Tu es un expert en caractérisation. À partir des PASSAGES, détaille pour chaque matériau: nom/composition, "
#        "propriétés mesurées, méthode de caractérisation, géométrie des échantillons, conditions expérimentales. "
#        "Citations: (Auteur et al., Année — DOI — Section). N'ajoute rien de non présent.\n\n"
#        "Question: {q}\n\nPASSAGES:\n{psg}\n"
#    ),
#    QType.SOFTWARE: (
#        "Réponds en FRANÇAIS UNIQUEMENT.\n"
#        "Tu es un expert logiciels scientifiques. À partir des PASSAGES, liste chaque logiciel (nom + version si présent), "
#        "fonction utilisée (simulation/optimisation/traitement), modules, paramètres clés et validation. "
#        "Citations: (Auteur et al., Année — DOI — Section). N'invente pas.\n\n"
#        "Question: {q}\n\nPASSAGES:\n{psg}\n"
#    ),
#    QType.COMPARISON: (
#        "Réponds en FRANÇAIS UNIQUEMENT.\n"
#        "Analyse COMPARATIVE stricte à partir des PASSAGES. Pour chaque article: points forts/innovations, "
#        "limitations explicites, différences méthodologiques, qualité/provenance des données, conclusion. "
#        "Citations: (Auteur et al., Année — DOI — Section). N'invente rien.\n\n"
#        "Question: {q}\n\nPASSAGES:\n{psg}\n"
#    ),
#    QType.DATA: (
#        "Réponds en FRANÇAIS UNIQUEMENT.\n"
#        "À partir des PASSAGES, décris précisément le type de données (in vivo/in vitro, capteurs, essais), "
#        "quantités, sujets, protocoles et prétraitements. Cite (Auteur et al., Année — DOI — Section). "
#        "N'invente rien.\n\nQuestion: {q}\n\nPASSAGES:\n{psg}\n"
#    ),
#    QType.LIMITS: (
#        "Réponds en FRANÇAIS UNIQUEMENT.\n"
#        "Liste UNIQUEMENT les limites/contraintes explicitement mentionnées dans les PASSAGES, "
#        "avec leur impact. Cite (Auteur et al., Année — DOI — Section). N'invente rien.\n\n"
#        "Question: {q}\n\nPASSAGES:\n{psg}\n"
#    ),
#    QType.GENERAL: (
#        "Réponds en FRANÇAIS UNIQUEMENT.\n"
#        "Réponds de façon technique et précise STRICTEMENT à partir des PASSAGES. "
#        "Cite (Auteur et al., Année — DOI — Section). N'invente rien.\n\n"
#        "Question: {q}\n\nPASSAGES:\n{psg}\n"
#    ),
#}
#
#SUMMARY_PROMPT = (
#    "Réponds en FRANÇAIS UNIQUEMENT.\n"
#    "Résume cet ARTICLE UNIQUEMENT à partir des PASSAGES fournis. "
#    "Structure: Contexte, Méthodes (noms exacts + logiciels), Données/échantillons, Résultats quantitatifs, Limites, Conclusion. "
#    "Pour chaque phrase importante, ajoute (Auteur et al., Année — DOI — Section #[i]). "
#    "Si une info manque, écris « non indiqué ».\n\n"
#    "Article: {title} ({year}) — DOI: {doi}\n\nPASSAGES numérotés:\n{psg}\n"
#)




# ─────────────────────────────────────────────────────────────────────
# PROMPTS (ENGLISH • EXPLICIT METHODS)
# ─────────────────────────────────────────────────────────────────────
PROMPTS = {
    QType.METHODS: (
        "Answer in ENGLISH ONLY.\n"
        "Be EXPLICIT and PROCEDURAL. From the PASSAGES, extract for EACH method/algorithm/protocol:\n"
        "• Exact name of the method/algorithm/protocol\n"
        "• Inputs & outputs (units, ranges if present)\n"
        "• Experimental setup (specimens, preparation, fixtures, environment)\n"
        "• Loading/control type (force/strain/displacement), waveform, amplitude, frequency, #cycles\n"
        "• Software & version (e.g., MATLAB, FEBio) and specific modules/functions\n"
        "• Optimization/identification method (e.g., Levenberg–Marquardt), objective function (e.g., NRMSE), constraints\n"
        "• Constitutive model form(s) and key equations/parameters\n"
        "• Validation procedure(s) and metrics\n"
        "• Any sensitivity analyses (what varied, by how much, impact)\n"
        "If an item is missing, write “not reported”. Cite each item like: (First-author et al., Year — DOI — Section).\n\n"
        "Question: {q}\n\nPASSAGES:\n{psg}\n"
    ),
    QType.MATERIALS: (
        "Answer in ENGLISH ONLY.\n"
        "Report explicitly for EACH material: name/composition; specimen geometry & preparation; test devices; environment; measured properties; data reduction/equations; software; uncertainties.\n"
        "Cite (First-author et al., Year — DOI — Section). Do not invent.\n\n"
        "Question: {q}\n\nPASSAGES:\n{psg}\n"
    ),
    QType.SOFTWARE: (
        "Answer in ENGLISH ONLY.\n"
        "List EACH software/tool EXACTLY as named (and version if present). For each, specify role (simulation/optimization/pre-processing), modules/settings, inputs/outputs, and how results were validated.\n"
        "Cite (First-author et al., Year — DOI — Section). Do not invent.\n\n"
        "Question: {q}\n\nPASSAGES:\n{psg}\n"
    ),
    QType.COMPARISON: (
        "Answer in ENGLISH ONLY.\n"
        "Do a STRICT COMPARISON by article: innovations; explicit limitations; methodological differences; data provenance/quality; validation approach; key parameters. Cite each point.\n\n"
        "Question: {q}\n\nPASSAGES:\n{psg}\n"
    ),
    QType.DATA: (
        "Answer in ENGLISH ONLY.\n"
        "Describe data precisely: in vivo/in vitro; sample size; specimen type; instrumentation; control mode; signals recorded; sampling rate; preprocessing.\n"
        "Cite (First-author et al., Year — DOI — Section). No additions.\n\n"
        "Question: {q}\n\nPASSAGES:\n{psg}\n"
    ),
    QType.LIMITS: (
        "Answer in ENGLISH ONLY.\n"
        "List ONLY explicit limitations/constraints and their impact. Cite each item.\n\n"
        "Question: {q}\n\nPASSAGES:\n{psg}\n"
    ),
    QType.GENERAL: (
        "Answer in ENGLISH ONLY.\n"
        "Answer technically and precisely using ONLY the PASSAGES. Prefer specific numbers, equations, software, and procedures over generalities. Cite each claim.\n\n"
        "Question: {q}\n\nPASSAGES:\n{psg}\n"
    ),
}

SUMMARY_PROMPT = (
    "Answer in ENGLISH ONLY.\n"
    "Produce a METHODS-CENTRIC structured summary from PASSAGES ONLY.\n"
    "Sections: Context; Specimens & Preparation; Test Apparatus & Environment; Loading Protocols; Signals & Data Reduction; Constitutive Model (equations/parameters); Software & Optimization (names, versions, algorithms, objective); Validation; Sensitivity/Robustness; Key Quantitative Results; Limitations; Conclusion.\n"
    "Cite important sentences like (First-author et al., Year — DOI — Section #[i]). If info is missing, write “not reported”.\n\n"
    "Article: {title} ({year}) — DOI: {doi}\n\nPASSAGES:\n{psg}\n"
)


# ─────────────────────────────────────────────────────────────────────
# OUTILS
# ─────────────────────────────────────────────────────────────────────
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", re.I)

def _primary_author(p: Dict) -> str:
    if p.get("first_author"): return p["first_author"]
    s = (p.get("authors_str") or "").split(";")[0].strip()
    return s or "Auteur"

def _fmt_passages(passages: List[Dict], max_chars=1000) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        title = p.get("title") or "—"
        year  = p.get("year")  or "—"
        doi   = p.get("doi")   or "—"
        sect  = p.get("section_path") or p.get("section_type","")
        auth  = _primary_author(p)
        txt   = (p.get("chunk") or "").replace("\n"," ").strip()
        if len(txt) > max_chars: txt = txt[:max_chars].rstrip() + "…"
        head  = f"[#{i}] {auth} et al., {year} — {title} | DOI: {doi} | Section: {sect}"
        blocks.append(head + "\n" + txt)
    return "\n\n".join(blocks)

def _sources_footer(passages: List[Dict]) -> str:
    # liste dédupliquée (auteur, année, titre, DOI, section)
    seen, lines = set(), []
    for i, p in enumerate(passages, 1):
        key = f"{p.get('doi','')}|{p.get('title','')}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [#{i}] { _primary_author(p) } et al., {p.get('year','—')}. {p.get('title','—')}. DOI: {p.get('doi','—')}. Section: {p.get('section_path','')}")
    return "\n".join(lines) if lines else "—"

# ─────────────────────────────────────────────────────────────────────
# RETRIEVER (Milvus + BGE-M3 + CrossEncoder reranker)
# ─────────────────────────────────────────────────────────────────────
class ScientificRetriever:
    def __init__(self):
        connections.connect(alias="default", uri=URI, token=TOKEN, secure=True)
        self.col = Collection(COLLECTION)
        try: self.col.load()
        except Exception: pass

        self.emb = BGEM3FlagModel(EMB_MODEL, use_fp16=USE_FP16)   # BGE-M3 => vecteurs normalisés
        self.reranker = CrossEncoder(RERANK_MODEL) if USE_RERANKER else None
        self.search_params = {"metric_type": "IP", "params": {"search_list_size": 100}}

    def _encode(self, q: str) -> np.ndarray:
        out = self.emb.encode([q], batch_size=1)   # -> dict{"dense_vecs": np.ndarray}
        vec = out["dense_vecs"]
        return vec.astype(np.float32) if vec.dtype != np.float32 else vec

    def search(self, query: str, k: int = 18) -> List[Dict]:
        qv = self._encode(query)
        res = self.col.search(
            data=qv,
            anns_field="embedding",
            param=self.search_params,
            limit=k,
            output_fields=["chunk_id","title","year","section_path","section_type",
                           "chunk","doi","first_author","authors_str","source_file",
                           "chunk_index","token_count"]
        )
        items = []
        for h in res[0]:
            e = h.entity
            items.append({
                "chunk_id": e.get("chunk_id"),
                "title": e.get("title"),
                "year": e.get("year"),
                "section_path": e.get("section_path"),
                "section_type": e.get("section_type"),
                "chunk": e.get("chunk"),
                "doi": e.get("doi"),
                "first_author": e.get("first_author"),
                "authors_str": e.get("authors_str"),
                "source_file": e.get("source_file"),
                "chunk_index": e.get("chunk_index"),
                "token_count": e.get("token_count"),
                "_score": float(h.distance) if hasattr(h,"distance") else None
            })
        if self.reranker and items:
            pairs  = [(query, it["chunk"]) for it in items]
            scores = self.reranker.predict(pairs)
            order  = np.argsort(-scores)
            items  = [items[i] for i in order]
        return items

    def by_doi(self, doi: str, limit: int = 200) -> List[Dict]:
        rows = self.col.query(
            expr=f'doi == "{doi}"',
            output_fields=["chunk_id","title","year","section_path","section_type",
                           "chunk","doi","first_author","authors_str","source_file",
                           "chunk_index","token_count"],
            limit=limit
        )
        rows.sort(key=lambda r: r.get("chunk_index", 0))
        return rows

# ─────────────────────────────────────────────────────────────────────
# LLM (Qwen via Ollama) — FR forcé via system prompt
# ─────────────────────────────────────────────────────────────────────
#class QwenLLM:
#    def __init__(self, model=LLM_MODEL, base=OLLAMA_URL):
#        self.model = model
#        self.base  = base
#        self.system_fr = (
#            "Tu es un expert scientifique francophone. "
#            "Réponds EXCLUSIVEMENT en français, sauf pour les titres d’articles, noms propres, variables, équations ou citations textuelles. "
#            "Ne traduis pas les DOI ni les références. "
#            "Si le contenu source est en anglais, synthétise et explique en français."
#        )

    # ── Language / System prompt ──
class QwenLLM:
    def __init__(self, model=LLM_MODEL, base=OLLAMA_URL):
        self.model = model
        self.base = base
        self.system_fr = (
                "You are a scientific methods auditor. "
                "Write in ENGLISH ONLY. Be concrete and procedural. "
                "Always name software/tools (and versions if present), algorithms, objective functions, hyperparameters, equations, and validation steps. "
                "Do NOT add facts not present in the provided passages."
            )

    def ask(self, prompt: str, timeout=180) -> str:
        r = requests.post(
            f"{self.base}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "system": self.system_fr,
                "options": {"temperature": 0.3, "repeat_penalty": 1.1}
            },
            timeout=timeout
        )
        r.raise_for_status()
        return (r.json() or {}).get("response","").strip()

# ─────────────────────────────────────────────────────────────────────
# ASSISTANT (avec filet de sécurité FR)
# ─────────────────────────────────────────────────────────────────────
EN_HINTS = re.compile(r"\b(the|and|with|this|these|those|is|are|were|was|model|results|method|dataset|study|we|our|their)\b", re.I)

class Assistant:
    def __init__(self):
        self.ret = ScientificRetriever()
        self.llm = QwenLLM()

    def _ensure_french(self, text: str) -> str:
        tokens = text.split()
        window = tokens[:200] if len(tokens) > 200 else tokens
        if not window:
            return text
        en_ratio = sum(1 for t in window if EN_HINTS.search(t)) / float(len(window))
        if en_ratio > 0.15:  # output probablement en anglais
            fix_prompt = (
                "Translate the following text into ENGLISH, keeping all numbers, equations, units, "
                "and references/DOI exactly as written. Do not add any information.\n\n"
                f"TEXT:\n{text}"
            )
            return self.llm.ask(fix_prompt)
        return text

    def answer(self, question: str) -> str:
        # résumé ciblé si la question contient un DOI
        doi_in_q = DOI_RE.search(question)
        if doi_in_q and any(w in question.lower() for w in ["résume","resume","summary","synthèse","synthese"]):
            doi = doi_in_q.group(0)
            chunks = self.ret.by_doi(doi)
            if not chunks:
                return f"Aucun article trouvé pour DOI {doi}."
            art = chunks[0]
            prompt = SUMMARY_PROMPT.format(
                title=art.get("title","—"), year=art.get("year","—"), doi=doi,
                psg=_fmt_passages(chunks, max_chars=900)
            )
            summary = self.llm.ask(prompt)
            summary = self._ensure_french(summary)
            return summary + "\n\nSources:\n" + _sources_footer(chunks)

        # sinon QA
        qtype = detect_qtype(question)
        passages = self.ret.search(question, k=18)
        if not passages:
            return "Aucun passage pertinent trouvé dans la base."
        prompt = PROMPTS[qtype].format(q=question, psg=_fmt_passages(passages))
        ans = self.llm.ask(prompt)
        ans = self._ensure_french(ans)
        return ans + "\n\nSources:\n" + _sources_footer(passages)

assistant = Assistant()

# ─────────────────────────────────────────────────────────────────────
# UI GRADIO (style ChatGPT)
# ─────────────────────────────────────────────────────────────────────
INTRO = (
    "Assistant RAG scientifique (BGE-M3 + Milvus + Qwen).\n"
    "- Pose une question précise (méthodes, logiciels, données, limites, comparaison, etc.).\n"
    "- Pour résumer un article : « résume DOI:10.xxxx/yyy ».\n"
    "- Les réponses citent toujours (Auteur, Année — DOI — Section) et sont en français."
)

def chat_fn(message, history):
    try:
        return assistant.answer(message)
    except Exception as e:
        return f"❌ Erreur: {e}"

if __name__ == "__main__":
    launch_kwargs = dict(server_name=SERVER_NAME, share=False)
    # On ne force le port que s'il est explicitement défini
    if SERVER_PORT is not None:
        launch_kwargs["server_port"] = SERVER_PORT

    gr.ChatInterface(
        fn=chat_fn,
        title="Maxime • Assistant scientifique RAG (FR)",
        description=INTRO,
        theme="soft",
        type= "messages",
        examples=[
            "Quelles méthodes d'identification des paramètres musculaires sont citées ?",
            "Quels logiciels et versions sont utilisés pour la simulation ?",
            "résume DOI:10.1016/j.jmbbm.2023.105968",
        ],
    ).launch(**launch_kwargs)


#.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=False)