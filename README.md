# Lokale Modelle selbst trainieren – Notebooks

Notebooks zum Talk *„Lokale Modelle selbst trainieren“* (Johann Hartmann, Mayflower). Drei echte Probleme, drei
selbst trainierte Modelle, alles mit öffentlichen Daten. Jedes Notebook läuft eigenständig in Google Colab:
öffnen, GPU-Laufzeit wählen, *Alle ausführen*.

| | Notebook | Was trainiert wird | Fertiges Modell |
|---|---|---|---|
| 01 | [![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayflower/local-training-talk-notebooks/blob/main/01_colbert_produkte_trainieren_colab.ipynb) **Produktsuche mit ColBERT** | `mLateOn` auf Amazon ESCI, Full-Fine-Tuning, Evaluation gegen BM25, Dense und Basismodell | [mlateon-esci-product-search](https://huggingface.co/johannhartmann/mlateon-esci-product-search) |
| 02 | [![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayflower/local-training-talk-notebooks/blob/main/02_decision_crawler_trainieren_colab.ipynb) **Crawler mit Decision Model** | Decider-2B + LoRA wählt die nächste Browseraktion aus dem Accessibility-Baum; Live-Crawl mit Playwright | [decider-2b-a11y-crawler-lora](https://huggingface.co/johannhartmann/decider-2b-a11y-crawler-lora) |
| 03 | [![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayflower/local-training-talk-notebooks/blob/main/03_toolcall_guard_trainieren_colab.ipynb) **Tool-Call-Guard gegen HITL-Müdigkeit** | Decider-2B + LoRA entscheidet CONTINUE / ASK / BLOCK für Tool-Calls von Agenten | [decider-2b-toolcall-guard-lora](https://huggingface.co/johannhartmann/decider-2b-toolcall-guard-lora) |

**Laufzeit:** T4 (kostenlos) funktioniert, L4 oder A100 sind deutlich schneller. Die Notebooks wählen ihr Profil
nach GPU, RAM und Platte selbst. Training in Colab-Größe dauert je nach GPU etwa eine halbe bis wenige Stunden.

**Secrets** (Schlüssel-Symbol links in Colab), alle optional:
- `HF_TOKEN` (write) – eigenes Modell auf den Hugging Face Hub hochladen
- `ANTHROPIC_API_KEY` – nur für die abgeschalteten Schritte, die Trainingsdaten oder Urteile neu erzeugen (kostenpflichtig, mit Budgetgrenze)

**Fortsetzen:** Mit Google Drive als Speicher lassen sich abgebrochene Läufe fortsetzen (`USE_GOOGLE_DRIVE` bzw. die Drive-Option im Setup).

## Datensätze

- [toolcall-guard-v1](https://huggingface.co/datasets/johannhartmann/toolcall-guard-v1) – 5.259 Tool-Call-Entscheidungen, erzeugt mit distilabel + Claude aus AgentDojo und ToolEmu (MIT)
- [esci-llm-judged-top10](https://huggingface.co/datasets/johannhartmann/esci-llm-judged-top10) – LLM-Urteile für die von ESCI nie gelabelten Top-10-Treffer (Apache-2.0)
- [nnetnav-live-objective-filter](https://huggingface.co/datasets/johannhartmann/nnetnav-live-objective-filter) – Qualitätsfilter für NNetNav-Aufgaben (Apache-2.0)

Details zu Profilen, getesteten Läufen und Messwerten: [DETAILS.md](DETAILS.md).

## Lizenz

Notebooks: Apache-2.0. Basismodelle, Daten und veröffentlichte Modelle: siehe die jeweiligen Hugging-Face-Karten.
