# Colab-Fassungen der Notebooks

Eigenständige Varianten für Google Colab: nur die `.ipynb` hochladen, Laufzeit mit GPU wählen, *Alle ausführen*.

## 01 · Der eigene Produkt-Retriever (`01_colbert_produkte_trainieren_colab.ipynb`)

Full-Fine-Tuning von `lightonai/mLateOn-unsupervised` auf Amazon ESCI mit Sentence Transformers 6.1.0
(`MultiVectorEncoder`, `CachedMultiVectorMultipleNegativesRankingLoss`, Skala 30, annotierte + geminte Hard
Negatives), exakte MaxSim-Evaluation gegen BM25, Basismodell und das veröffentlichte Vollmodell
[`johannhartmann/mlateon-esci-product-search`](https://huggingface.co/johannhartmann/mlateon-esci-product-search),
LLM-vervollständigte Metriken, Vorher/Nachher-Suche, Export.

**Eigenständig:** Nur die `.ipynb` hochladen. Pakete kommen von PyPI (gepinnt: `sentence-transformers[train]==6.1.0`,
`transformers==5.17.0`, `bm25s==0.3.11`, `PyStemmer==3.1.0`; Torch/NumPy/ipywidgets von Colab bleiben), ESCI von
GitHub, das Auswertungsmodul `esci_judge.py`, die LLM-Urteile und die Top-10-Listen des Vollkatalog-Laufs aus dem
öffentlichen Datensatz [`johannhartmann/esci-llm-judged-top10`](https://huggingface.co/datasets/johannhartmann/esci-llm-judged-top10)
(Revision gepinnt). Sind betroffene Bibliotheken schon importiert, startet die Installationszelle die Laufzeit neu.

**Profile** (`PROFILE="auto"` wählt nach GPU-Speicher, RAM und Platte; überschreibbar, ebenso `PRECISION`):

| Profil | Training | Katalog / Test-Queries | Auto-Wahl |
|---|---|---|---|
| `t4` | 3.000 Queries, ~5.600 Tripel, 44 Schritte | 100.000 Produkte / 1.000 | Standard |
| `l4` | 6.000 Queries | 150.000 / 1.000 | ≥22 GB GPU, ≥24 GB RAM |
| `a100` | alle ~20k Queries, ~68k Tripel (Rezept des Vollmodells) | 482.105 / 2.000 | ≥38 GB GPU, ≥45 GB RAM, ≥100 GB Platte |
| `full` | wie `a100` | 482.105 / 8.955 | nur manuell |

Der Katalog enthält alle beurteilten Produkte der verwendeten Queries plus Zufallsprodukte. Zahlen auf dem
Teilkatalog sind untereinander vergleichbar, nicht 1:1 mit der Vollkatalog-Tabelle. Die ersten 1.000 Test-Queries
sind die Stichprobe des Judge-Datensatzes; Abschnitt 9b rechnet zusätzlich die Vollkatalog-Referenz aus dem Hub nach
(identisch mit `results.json`). Neue LLM-Urteile nur mit `REGENERATE_JUDGMENTS=True`, Colab-Secret
`ANTHROPIC_API_KEY` und harter Grenze `JUDGE_BUDGET_USD`.

**Fortsetzen:** `USE_GOOGLE_DRIVE=True` legt Tripel, Checkpoints (~7 GB), Modell und Ergebnisse auf Drive. Nach
einem Abbruch erneut *Alle ausführen*: Training setzt am letzten Checkpoint fort, ein fertiges Modell wird nicht neu
trainiert, Indizes werden per Fingerprint wiederverwendet (lokal, gehen mit der Laufzeit verloren).
**Export:** `HUB_REPO_ID` + Colab-Secret `HF_TOKEN` (Schreibrecht) → `push_to_hub`; `COPY_TO_DRIVE=True` → Drive.

### Getesteter Lauf

Kein echtes Colab: frische Python-3.12-Umgebung mit Colab-ähnlichem Stack (torch 2.8.0, numpy 2.0.2, pandas 2.2.2,
pyarrow 18.1, transformers 4.57, sentence-transformers 5.1, datasets 4.0, ipywidgets 7.7.1), eigener Kernel,
Ausführung mit nbclient inklusive Installationszelle. Profil `t4` erzwungen, **FP16**, VRAM per
`set_per_process_memory_fraction` auf 15 GiB gedeckelt, RTX A6000 geteilt mit anderen Jobs (Laufzeiten daher
Obergrenzen für A6000-Klasse; eine echte T4 ist langsamer).

1.000 Test-Queries, exakt über 100.000 Produkte (Teilkatalog):

| Verfahren | Recall@10 | Recall@100 | nDCG@10 | MRR@10 |
|---|---|---|---|---|
| BM25 | 0,407 | 0,724 | 0,431 | 0,576 |
| mLateOn-unsupervised (Basis) | 0,487 | 0,791 | 0,517 | 0,662 |
| mlateon-esci-product-search (veröffentlicht, Profil `full`) | 0,535 | 0,841 | 0,572 | 0,712 |
| **Eigenes Training (Profil `t4`, 9 min)** | **0,533** | **0,835** | **0,561** | **0,693** |

Eigenes Training vs. Basis: Recall@100 +0,044 (95-%-KI +0,035 bis +0,053), nDCG@10 +0,044 (+0,035 bis +0,052).
Dev-nDCG@10 0,643 → 0,678. LLM-vervollständigt (dieselben 1.000 Queries): nDCG@10 Basis 0,545 → eigenes 0,586,
veröffentlicht 0,598; 25 % der eigenen Top-10 liegen außerhalb des gelabelten Pools.

| Phase (Profil `t4`, FP16, geteilte A6000) | Minuten |
|---|---|
| Installation (Pakete im pip-Cache) / ESCI-Download (1,2 GB) | 0,1 / 0,6 |
| Index Basismodell (100k Produkte, 3,7 GiB) | 15,3 |
| Hard-Negative-Mining / Dev-Evaluation vorher | 0,9 / 1,1 |
| Training (44 Schritte, 2 Dev-Evaluationen) | 8,7 |
| BM25 / Suche Basismodell | 0,3 / 0,3 |
| Index + Suche veröffentlichtes / eigenes Modell | 12,7 / 16,6 |
| Gesamt | ~57 |

Spitzen: GPU 7,9 GiB (PyTorch, Training; nvidia-smi 9,1 GiB), RAM 6,2 GiB anonym (RSS 17,7 GiB inkl. per
`memmap` gelesener Index-Seiten, vom Kernel freigebbar), Platte 20,5 GiB. Profile `l4`, `a100`, `full` sind nicht
ausgeführt worden. Geprüft wurden außerdem: Fortsetzen nach Abbruch (Index und Tripel wiederverwendet), erneuter
Lauf mit fertigem Modell (Training übersprungen, alle Indizes wiederverwendet, 2,6 min, identische Metriken), Colab-Guards ohne `google.colab` und mit gefälschtem Modul
(fehlendes Secret, Drive-Mount), Export ohne Token, Neuberechnungspfad mit Dummy-Key und Budget 0 (distilabel
installiert, BudgetGuard stoppt vor jedem API-Aufruf, $0,00).

## 02 · Decision-Crawler trainieren (`02_decision_crawler_trainieren_colab.ipynb`)

LoRA auf `Mapika/decider-2b` (v10, Commit `d61c1c1`) als System-1-Crawler auf `stanfordnlp/nnetnav-live` (Commit `7f69dca`):
BrowserGym-v0.13.3-Beobachtung (Markierungsskripte von GitHub, SHA-256 geprüft), Kandidatenaktionen, Texteingabe als zweite
Entscheidung, Split nach Website, Auftragsfilter aus [`johannhartmann/nnetnav-live-objective-filter`](https://huggingface.co/datasets/johannhartmann/nnetnav-live-objective-filter)
(Commit `80eefb0`), echtes Training in Colab-Größe, Held-out-Evaluation für Basis / eigenes / veröffentlichter Adapter
[`johannhartmann/decider-2b-a11y-crawler-lora`](https://huggingface.co/johannhartmann/decider-2b-a11y-crawler-lora) (v2, Commit
`39421be`), 16 Live-Aufgaben auf Scraping-Übungsseiten mit Datenextraktion, Korrekturen + Weitertraining, Upload.

**Eigenständig:** Nur die `.ipynb` hochladen. `objective_filter.py` kommt aus dem Modell-Repo (`training/02_data/`, Commit
`39421be`, SHA-256 = aktuelle `02_data/objective_filter.py` inklusive der Korrektur für distilabels Rich-Tracebacks).
`decider-ai==1.5.0` mit `--no-deps` (Colabs numpy 2 bleibt), dazu `transformers==5.17.0`, `peft==0.21.0`,
`flash-linear-attention==0.5.2`, `playwright==1.57.0`, `distilabel==1.5.3`; Chromium per `playwright install --with-deps`
(ohne apt nur Chromium). `train.jsonl` (3,2 GB) wird einmal geladen und zeilenweise gelesen; Metadaten, Held-out-Auswahl und
die gezogenen Trainingsentscheidungen landen im Laufverzeichnis, ein fortgesetzter Lauf braucht die Datei nicht mehr
(`NNETNAV_ON_DRIVE=True` cacht sie zusätzlich auf Drive).

**Profile** (`PROFILE="auto"` nach GPU-Speicher; überschreibbar, ebenso `PRECISION`):

| Profil | Training (Aktion + Text) | Seiten bis | Held-out ausgewertet | Auto-Wahl |
|---|---|---|---|---|
| `t4` | 2.000 + 200, max. 2 je Aufgabe | 8.192 Tokens | erste 300 von 600 (+400 Texte) | Standard |
| `l4` | 4.000 + 400, max. 3 je Aufgabe | 12.288 | 600 (+400) | ≥ 20 GiB GPU |
| `a100` | 8.000 + 800 | 16.384 | 600 (+400) | ≥ 38 GiB GPU |
| `full` | alle 26.969 + 2.613 (Rezept von v2, 29 h auf einer A6000) | 24.576 | 600 (+400) | nur manuell |

Die Held-out-Auswahl ist unabhängig vom Profil die des vollen Laufs (Fingerabdruck `ddb7f70a…/2d09f61d…` wird gegen
`decision_contract.json` von v2 geprüft), ebenso die 160 Validierungsaufgaben. Vor dem Training steht eine grobe
Zeitschätzung je GPU-Klasse (Annahme, nicht gemessen), nach den ersten Schritten die gemessene Restzeit. Das Basismodell
crawlt live nur mit `LIVE_BASE` (auf `t4` standardmäßig aus; Referenz des vollen Laufs: 6/16).

**GPU:** L4/A100 in BF16, T4 (sm75, per Compute Capability erkannt) in FP16: Gewichte FP16, LoRA FP32, Loss-Scaling. Ohne
Flash-SDPA (T4) rechnet PyTorch Grouped-Query-Attention nur im quadratischen Referenzkernel (bei 24k Tokens ~9 GiB je
Schicht); das Notebook expandiert dann die KV-Köpfe, damit der speichersparende Kernel greift. Alle Sequenzen werden auf
512 Tokens aufgerundet (kausal, ohne Maske, Slots unverändert): Jede neue Länge kostet sonst ~17 MB Hauptspeicher für
formspezifische Kernel (gemessen; ohne Rundung wuchs der Kernel im Testlauf auf 10,1 GiB, mit 7,1 GiB).
Präzisionsprobe gegen FP32 auf 40 Held-out-Entscheidungen (Seiten ab 2,4k Tokens, Median 14,2k): gleiche argmax-Entscheidung FP16
100 % (Basis) / 97,5 % (v2), FP16 mit T4-Kernelpfad 100 % / 100 %, BF16 97,5 % / 100 %; max. |Δp| FP16 ≤ 0,0055, BF16 ≤ 0,029;
kein NaN. Abschnitt 9 vergleicht zusätzlich in jeder Laufzeit Basis und v2 je Entscheidung mit dem BF16-Referenzlauf.

**Kosten:** `ALLOW_PAID_API=False`. Der Auftragsfilter kommt vom Hub; Neuerzeugung nur mit `ALLOW_PAID_API=True`,
`RUN_FILTER=True`, Secret `ANTHROPIC_API_KEY` und Budgetgrenzen, bezahlte Antworten vorher aus dem Hub-Cache, danach Upload
nach `FILTER_SYNC_REPO` nur mit Schreibrecht.

**Drive/Fortsetzen/Export:** `USE_DRIVE=True` legt alles unter `MyDrive/a11y-training` ab (Checkpoints ~200 MB alle 25
Schritte, maximal zwei). Das Laufverzeichnis ergibt sich aus Profil, Präzision und Rezept; erneut *Alle ausführen* setzt
das Training fort und lädt fertige Auswertungen und Crawls. Mit Secret `HF_TOKEN` (Schreibrecht) lädt Abschnitt 12 Adapter,
`continued/`, `decision_contract.json`, `inference.py`, `training/` (Filtermodul, Rezept, Logs, Auswertungen) und eine
Modellkarte nach `<user>/decider-2b-a11y-crawler-lora-colab` (privat).

### Getesteter Lauf

Kein echtes Colab: frische Python-3.12-Umgebung (torch 2.8.0, numpy 2.0.2, pandas 2.2.2, transformers 4.57, datasets 4.0,
ipywidgets 7.7.1), eigener Kernel, nbclient inklusive Installationszelle, leeres Arbeits-, HF-Cache- und Playwright-Verzeichnis,
ohne Tokens. Profil `t4` erzwungen, **FP16**, dazu wie auf einer T4 Flash-SDPA abgeschaltet (Math-SDPA ebenfalls, damit ein
quadratischer Rückfall als Fehler auffiele), VRAM auf 14 GiB gedeckelt, RTX A6000 geteilt mit anderen Jobs. Chromium-Bibliotheken
kamen lokal per `LD_LIBRARY_PATH` (NixOS, kein apt), der `--with-deps`-Zweig lief deshalb nicht.

Held-out-Sites, erste 300 Aktions- und alle 400 Texteingabe-Entscheidungen; live 16 Prüfaufgaben:

| System | Held-out | Eingabetext | Live: Ziel | Live: Daten korrekt |
|---|---|---|---|---|
| Decider-2B Basis | 8,7 % | 22,3 % | – (Referenz 6/16) | – |
| v2, veröffentlicht (BF16-Referenzlauf: 44,3 % / 75,8 %) | 44,3 % | 75,8 % | 13/16 | 12/16 |
| **eigenes, Profil `t4` (2.000 + 200, 60 min)** | **38,3 %** | **64,3 %** | **12/16** | **12/16** |
| eigenes + Korrektur | Teilmenge 100: 39 → 35 % | – | 13/16 | 12/16 |

Eigenes vs. Basis: 100 nur eigenes richtig, 11 nur Basis. Eigenes vs. v2: 17 zu 35 (Vorzeichentest p ≈ 0,013). Je Aktion
eigenes / v2: click 14,6 / 19,4 %, type 50,0 / 64,8 %, stop 80,0 / 70,8 %, scroll 7,7 / 15,4 %, go_back 11,1 / 33,3 %.
Validierung 43,3 → 45,0 % (Schritt 200 gewählt). Präzision: Basis und v2 in FP16 treffen auf 100 % / 99,3 % der 300
Entscheidungen dasselbe Ergebnis wie der BF16-Referenzlauf, gleiche Genauigkeit, |Δp| Median 0,0002 / 0,0016.
Korrekturschleife: 31 geprüfte Zustände, 6 über 8.192 Tokens ausgelassen (Länderliste, ~37k Tokens), 3 Korrekturen,
39 Schritte; Korrekturaufgaben 8/10 → 9/10, Held-out-Teilmenge (100) sinkt von 39 auf 35 %.

| Phase (Profil `t4`, FP16, T4-Kernelpfad, geteilte A6000) | Minuten |
|---|---|
| Installation (pip-Cache warm) / Download `train.jsonl` + Scan | 0,4 / 1,9 |
| Held-out-Auswahl, Validierung, Trainingsstichprobe (Tokenisierung) | 4,3 |
| Training (275 Schritte, 9,5 M Tokens, 3 Validierungen) | 59,8 |
| Held-out: drei Bedingungen | 18,8 |
| Live: v2 + eigenes / Korrekturen sammeln | 5,0 / 3,9 |
| Weitertraining / Auswertung danach | 9,1 / 7,1 |
| Gesamt | 111,5 |

Spitzen: GPU 6,4 GiB (PyTorch, Training; nvidia-smi 7,7 GiB), RAM 7,1 GiB anonym, Platte 8,2 GiB (davon 6,6 GiB HF-Cache).
Auf einer echten T4 dauert das Training ein Mehrfaches (Schätzung im Notebook ~3,8 h). Ein früherer Lauf derselben Fassung ohne
Tokenbudget im Weitertraining lief dort in einen OOM (37k-Token-Seite mit Gradienten, >14 GiB); nach der Korrektur setzte
erneutes *Alle ausführen* im selben Verzeichnis nach 16 min fort (alles bis dahin aus dem Cache, ohne `train.jsonl`). Geprüft
außerdem: Colab-Guards ohne `google.colab` und mit gefälschtem Modul (Secret fehlt/vorhanden, Drive-Mount abgelehnt), Upload
ohne Token, Neuerzeugung des Filters mit Dummy-Key und Budget 0 (8 Aufgaben aus dem Hub-Cache, 8/8 wie veröffentlicht,
$0,00, Kostenbuch ohne neue Einträge). Nicht getestet: echte T4 (Triton-Kernel von flash-linear-attention auf sm75,
Laufzeit), `--with-deps` in Colab, Profile `l4`/`a100`/`full`, Upload mit echtem Token, Drive-Durchsatz.

## 03 · Tool-Call-Guard trainieren (`03_toolcall_guard_trainieren_colab.ipynb`)

LoRA auf `Mapika/decider-2b` (v11, Commit `533964d`) als Guard (CONTINUE/ASK/BLOCK) auf
[`johannhartmann/toolcall-guard-v1`](https://huggingface.co/datasets/johannhartmann/toolcall-guard-v1): Baselines,
Teacher-Referenz aus dem Cache, echtes Training (Colab-Standard **1 Epoche**, 197 Schritte), Reload, Auswertung auf
`test_seen_tools`/`test_unseen_tools` und ASSEBench (menschliche Labels, zur Laufzeit gebaut), der veröffentlichte
Adapter [`johannhartmann/decider-2b-toolcall-guard-lora`](https://huggingface.co/johannhartmann/decider-2b-toolcall-guard-lora)
(2 Epochen) als zusätzliches System, Schwelle, `ToolCallGuard` mit AgentDojo-Demo, Review-Widget, Weitertraining,
Upload des eigenen Adapters.

**Eigenständig:** `guard_data.py` kommt aus dem Adapter-Repo (`training/guard_data.py`, Revision `75d1221`,
SHA-256 geprüft); die spätere Sicherheitskorrektur (Rich-Tracebacks von distilabel ohne `show_locals`) wird im
Notebook eingefügt und das Ergebnis gegen die SHA-256 der aktuellen `03_data/guard_data.py` geprüft.
`decider-ai==1.5.0` wird mit `--no-deps` installiert (sein Pin `numpy<2` wird bewusst übergangen, `decider.prompt`
und `decider.model` nutzen kein numpy), dazu die echten Laufzeitabhängigkeiten; Colabs numpy 2 bleibt, kein Neustart.
`agentdojo==0.1.35` und `distilabel==1.5.3` installieren sauber auf Colabs Stack.

**GPU:** L4/A100 in BF16. Eine T4 hat kein natives BF16 (`torch.cuda.is_bf16_supported()` meldet dort wegen
Emulation trotzdem True, geprüft wird die Compute Capability); `PRECISION='auto'` schaltet dann auf FP16 (Gewichte
FP16, LoRA FP32, Loss-Scaling). Gegen FP32 auf 300 ungesehenen Testfällen: gleiche argmax-Entscheidung FP16
99,3 % (Basis) / 100 % (veröffentlichter Adapter), BF16 98,3 % / 100 %; kein NaN. Im FP16-Notebooklauf erreicht der
veröffentlichte Adapter exakt die BF16-Accuracy (0,802 / 0,777).

**Kosten:** `ALLOW_PAID_API=False`. Die Sonnet-Teacher-Antworten kommen aus `raw/teacher_eval_cache.sqlite` im
Datensatz-Repo; 4 der 1.032 ASSEBench-Fälle fehlen dort (im Originallauf `refusal`, nicht gecacht) und zählen wie
im Original fail-safe als ASK. Bezahlte Schritte nur mit `ALLOW_PAID_API=True` + Secret `ANTHROPIC_API_KEY`, mit
denselben Budgetgrenzen und Hub-Sync (Upload nur mit Schreibrecht, sonst lokal mit Hinweis).

**Drive/Fortsetzen/Export:** `USE_DRIVE=True` legt Checkpoints (alle 50 Schritte), Adapter und Reviews nach
`MyDrive/toolcall-training`; `RESUME_RUN=<Ordner>` setzt ein abgebrochenes Training fort. Mit Secret `HF_TOKEN`
(Schreibrecht) lädt Abschnitt 14 Adapter, `continued/`, `guard_contract.json`, `inference.py`, `training/guard_data.py`
und eine Modellkarte nach `<user>/decider-2b-toolcall-guard-lora-colab` (privat).

### Getesteter Lauf

Kein echtes Colab: frische Python-3.12-Umgebung (torch 2.8.0+cu126, numpy 2.0.2, pandas 2.2.2, transformers 4.57,
datasets 4.0, ipywidgets 7.7.1), eigener Kernel, nbclient inklusive Installationszelle, leeres Arbeits- und
HF-Cache-Verzeichnis, ohne Tokens. RTX A6000, geteilt mit anderen Jobs (Zeiten sind Obergrenzen für diese Klasse).

| System | Acc. seen / unseen | Macro-F1 seen / unseen | ASSEBench binär |
|---|---|---|---|
| Decider-2B Basis | 0,476 / 0,532 | 0,409 / 0,468 | 0,608 |
| **eigenes LoRA, 1 Epoche (BF16)** | **0,768 / 0,767** | **0,676 / 0,679** | **0,629** |
| veröffentlicht, 2 Epochen | 0,802 / 0,777 | 0,712 / 0,694 | 0,650 |
| eigenes + 7 Reviews | 0,780 / 0,744 | 0,674 / 0,627 | 0,690 |
| Claude Sonnet 5 (Cache) | – | – | 0,789 |

Schwelle auf `val`: CONTINUE ab p ≥ 0,7 → falsches CONTINUE 3,8 % / 4,9 %. Eigenes und veröffentlichtes Modell
entscheiden auf 91 % / 92 % der Testfälle gleich. AgentDojo-Demo: Angreifer-`send_money` wird geblockt.

Laufzeit BF16: Installation 0,2 min (pip-Cache), bis Baselines + Teacher 7,6 min, Training 28,9 min (Peak 5,7 GiB),
Gesamt 66,7 min. Zweiter Lauf mit `PRECISION='fp16'` und auf 40 Schritte begrenztem Training (nur im Test): alle
Zellen fehlerfrei, Loss-Verlauf wie in BF16, 38 min gesamt. Kostenbuch in beiden Läufen leer ($0,00). Colab-Guards
geprüft ohne `google.colab` und mit gefälschtem Modul (Drive-Mount abgelehnt → lokale Ausgaben, Secret gelesen und
nicht ausgegeben). Nicht getestet: echte T4 (Triton-Kernel von flash-linear-attention auf sm75), L4/A100-Laufzeiten,
Upload mit echtem Token, `RESUME_RUN` über Sitzungen, Widget-Darstellung im Colab-Frontend.
