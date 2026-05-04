# Slide-Deck Generation Prompt — Exoplanet ML Classifier

> Paste the prompt below verbatim into Claude / GPT / Gemini / a slide-AI
> tool such as **Tome**, **Beautiful.ai**, **Gamma**, or **SlidesGPT**.
> The output should be ready to drop into PowerPoint or Google Slides
> for the 12-minute MLEA_M sustentation on May 15, 2026.

---

## 📝 Prompt (copy from here, including the line above the rule)

```text
ROLE
You are an expert technical-presentation designer building a polished slide deck for
a graduate-level Machine-Learning course defence. The audience is two professors
(Mario Julián Cañón Ayala and Hector Javier Hortua Orjuela) and 30 master-level
students. Tone: professional, scientifically rigorous, visually clean, no emojis on
the actual slide bodies (a single emoji is allowed per slide title for warmth).

PROJECT BRIEF
- Title: "Exoplanet ML Classifier — Tabular vetting of NASA Kepler Objects of Interest"
- Course: MLEA_M — Machine Learning, Maestría en Ciencia de Datos
- Institution: Escuela Colombiana de Ingeniería Julio Garavito (ECI), 2026-1
- Authors: Andersson David Sánchez Méndez and Cristian Santiago Pedraza Rodríguez
- Heritage: continues the ECI Centauri team's "Exoplanet Hunter AI" prototype, which
  was a Global Finalist at the NASA Space Apps Challenge 2025
  (Frontend: github.com/JAPV-X2612/ECI-Centauri-Frontend,
   Backend:  github.com/JAPV-X2612/ECI-Centauri-Backend,
   CNN:      github.com/Ch0comilo/astronet-cnn-v3)
- Time budget: 12 minutes. Plan ~16-18 slides.
- Submission rubric (50 pts total, NEVER mention the points but cover every category):
   1. Problem & motivation (4 pts)
   2. Data (4 pts)
   3. EDA (8 pts)
   4. Models (14 pts) — at least 5 implemented
   5. Comparison metrics (6 pts)
   6. Hyper-parameters (4 pts)
   7. Application (4 pts)
   8. Code (3 pts)
   9. Presentation (3 pts)
- Final result: Tuned XGBoost wins with F1=0.860, ROC-AUC=0.936, MCC=0.721 on the
  held-out 20% test split (RANDOM_SEED = 42).

DESIGN RULES
- Aspect ratio: 16:9.
- Theme: dark navy primary (#0B3D91 — NASA navy) and orange accent (#FB8C00).
  Body text on a near-white (#FAFAFA) background. Use a single accent colour per
  slide, never both at full saturation.
- Typography: a sans-serif (e.g. Inter, Helvetica, Calibri). Title 32-36pt, body
  18-22pt, monospace 16-18pt for code.
- Charts: render via matplotlib with sns.set_theme(style='whitegrid'). NO 3D charts.
  NO clip art. Use the user's saved figures in `reports/figures/` whenever
  applicable.
- Equations: render with LaTeX. Always wrap them in display style.
- One idea per slide. If the slide needs more than 5 bullets, split into two slides.
- Every figure must have a one-sentence caption that explains the take-away, NOT
  the obvious description.

REQUIRED SLIDE SEQUENCE
Slide 1 — Title & Authors
   - Title, Course name, Authors, Professors, Date (May 15, 2026), Institution,
     ECI logo + NASA Space Apps "Global Finalist 2025" badge.

Slide 2 — Why this matters (Problem & Motivation)
   - Kepler observed ~150,000 stars and produced ~10,000 KOIs.
   - Many KOIs are not real planets (eclipsing binaries, noise).
   - Manual vetting is slow → automate triage.
   - Decision supported: "is this candidate worth a follow-up observation?"
   - Type of problem: SUPERVISED BINARY CLASSIFICATION.

Slide 3 — Dataset card
   - Source: NASA Exoplanet Archive Cumulative KOI Table (public domain).
   - 9,564 rows × 49 raw cols → 13 features after leakage removal.
   - Class balance: 4,847 FALSE POSITIVE / 4,717 CANDIDATE (≈51/49 %).
   - Show one table, one mini bar chart of the class balance.

Slide 4 — Data leakage trap
   - List the 9 columns we drop (koi_disposition, koi_score, koi_fpflag_*, IDs).
   - One short sentence each: WHY each is leakage.
   - Take-away: keeping them gives 99% F1 that means nothing.

Slide 5 — EDA highlights (1/2)
   - Embed `target_distribution.png` and `correlation_heatmap.png` side by side.
   - One-line caption: balanced classes, koi_period strongly skewed,
     koi_teq ↔ koi_insol almost perfectly correlated.

Slide 6 — EDA highlights (2/2) — PCA & KMeans
   - Embed `pca_scatter.png` (or `pca_explained_variance.png` + `kmeans_pca.png`).
   - Take-away: classes are partially linearly separable in 2D, but a non-linear
     model should win.

Slide 7 — Preprocessing pipeline
   - Mermaid-style diagram showing: raw CSV → drop_leakage → impute → scale →
     SMOTE check → stratified 72/8/20 split.
   - Note: imputer + scaler fit on TRAIN ONLY (inside sklearn Pipeline).

Slide 8 — Mathematical foundations
   - Three boxed equations:
     • Logistic regression log-odds:
       log(P/(1-P)) = β₀ + βᵀx
     • Cross-entropy loss:
       L = -1/N Σ [yᵢ log p̂ᵢ + (1-yᵢ) log(1-p̂ᵢ)]
     • SMOTE synthetic sample:
       x_new = xᵢ + λ(x_nn − xᵢ),  λ ~ U(0,1)

Slide 9 — Five classifiers (model zoo)
   - Single table with rows = algorithm, cols = "What it tests", "Strength",
     "Weakness".
   - Rows: Logistic Regression, k-NN, Random Forest, XGBoost, MLP.
   - One sentence below table: each row picks a different inductive bias on
     purpose so the comparison is honest.

Slide 10 — Architecture in one picture
   - Show the project's main mermaid pipeline (load → preprocess → 5 models →
     GridSearchCV → evaluate → save .joblib).
   - Mention: 6 reusable src/ modules, 63 unit tests, deterministic seed 42.

Slide 11 — Hyper-parameter tuning
   - GridSearchCV with 5-fold CV, scoring = F1.
   - Tuned only the two most expressive non-linear models (XGBoost, MLP).
   - Best XGBoost: n_estimators=200, max_depth=7, lr=0.05, subsample=0.8.
   - Best MLP: hidden=(128,64), ReLU, alpha=1e-4, adaptive LR.

Slide 12 — Metric choice (Why F1 + MCC, not Accuracy)
   - Two boxes:
     - F1 = 2·P·R / (P+R)  — balanced for moderate imbalance.
     - MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))  — robust under
       imbalance (Chicco & Jurman 2020).
   - Explain: missed planet > wasted follow-up → recall matters.

Slide 13 — Results: leaderboard
   - Final test-set table (winner row highlighted in NASA orange):
     Model              | Accuracy | F1     | ROC-AUC | MCC
     XGBoost (tuned)    | 0.8604   | 0.8598 | 0.9361  | 0.7210
     Random Forest      | 0.8495   | 0.8489 | 0.9271  | 0.6991
     MLP (tuned)        | 0.8202   | 0.8189 | 0.9006  | 0.6404
     Logistic Regression| 0.7784   | 0.7932 | 0.8341  | 0.5660
     k-NN               | 0.7862   | 0.7916 | 0.8568  | 0.5747

Slide 14 — Visual comparison
   - Side-by-side: `overlay_roc_curves.png` and `overlay_pr_curves.png`.
   - One-line caption: XGBoost dominates the convex hull on both curves.

Slide 15 — Confusion matrices & threshold tuning
   - Show `confusion_matrix_grid.png`.
   - Mini table:
     Threshold        | F1    | Precision | Recall
     0.50 (default)   | 0.860 | 0.851     | 0.869
     0.43 (Youden's J)| 0.862 | 0.829     | 0.899
   - Take-away: lowering the threshold ≈ +3 pp recall at small precision cost.

Slide 16 — Feature importance & interpretability
   - Show `feature_importance_xgboost.png` (top 15).
   - Highlight: koi_model_snr, koi_prad, koi_depth dominate — physically sensible.
   - Note that Logistic Regression coefficients are also available (single line).

Slide 17 — Application & impact
   - Three small icons + one-line each:
     • NASA Exoplanet Archive operators → faster triage.
     • Astronomers → higher-purity follow-up list, less wasted telescope time.
     • Education → open, fully reproducible baseline.
   - Honest limitation: tabular features only, no light-curve morphology.

Slide 18 — Conclusions & next steps
   - 4 bullets:
     1. Tuned XGBoost is the winner (F1=0.860, ROC-AUC=0.936, MCC=0.721).
     2. Removing label leakage was essential.
     3. Threshold tuning gives a free recall boost.
     4. Future: combine with light-curve CNN, try TabPFN, calibrate probabilities.
   - Add: code at github.com/AnderssonProgramming/exoplanet-ml-classifier (Apache 2.0).

Slide 19 (optional, only if time allows) — Q&A / Thank you
   - "Thank you" centred. Author emails. Acknowledgements (NASA, ECI, Profs).
   - NASA Space Apps Global Finalist 2025 badge bottom right.

DELIVERABLES YOU MUST PRODUCE
1. The full slide deck content as bullet-by-bullet markdown for every slide
   (so it can be pasted into PowerPoint Outline view).
2. A speaker-notes block under each slide (40–60 words, conversational, with the
   actual sentences I should say). Spanish OR English — match the user's choice
   below.
3. A timing plan in seconds adding up to ≤ 720 s (12 min).
4. ONE design-system summary block at the end: colour palette HEX,
   recommended fonts, spacing tokens (Apple HIG style: 4/8/16/24/32 px).

OUTPUT FORMAT
Markdown with H2 headings per slide, body bullets, then a `> Speaker notes:`
block-quote. After the last slide output a "Design system" section.

LANGUAGE: ENGLISH.
```

---

## 🛠 Recommended workflow

1. Copy the prompt above and feed it to your favourite slide AI.
2. Copy the resulting markdown into **PowerPoint → New Slide → Insert From Outline**
   (or the Google Slides equivalent).
3. Drop the saved figures from `../figures/` (relative to this folder) onto the
   matching slides.
4. Tighten the speaker notes to your speaking style.
5. Time-check: do **two full dry runs** to make sure you stay under 12 minutes.

## 📦 Where to find the assets the prompt references

| Asset | Path |
| --- | --- |
| ROC overlay | `reports/figures/overlay_roc_curves.png` |
| PR overlay | `reports/figures/overlay_pr_curves.png` |
| Confusion matrix grid | `reports/figures/confusion_matrix_grid.png` |
| XGBoost feature importance | `reports/figures/feature_importance_xgboost.png` |
| Random Forest feature importance | `reports/figures/feature_importance_random_forest.png` |
| LR coefficients | `reports/figures/feature_importance_logistic_regression.png` |
| Target distribution | `reports/figures/target_distribution.png` |
| Correlation heatmap | `reports/figures/correlation_heatmap.png` |
| Feature distributions by class | `reports/figures/feature_distributions_by_class.png` |
| Histograms | `reports/figures/histograms_numeric_features.png` |
| Missing-value heatmap | `reports/figures/missing_values_heatmap.png` |
| PCA scatter | `reports/figures/pca_scatter.png` |
| PCA explained variance | `reports/figures/pca_explained_variance.png` |
| KMeans on PCA | `reports/figures/kmeans_pca.png` |

## ✅ Final pre-flight checklist (the day before)

- [ ] Both authors agree on who delivers which slides.
- [ ] Two clean dry runs ≤ 12 minutes each.
- [ ] Notebooks 01–04 re-executed once on the presenter's laptop.
- [ ] `pytest tests/ -v` shows 63 passed, 0 failed.
- [ ] `models/best_xgboost.joblib` exists and `joblib.load` succeeds.
- [ ] Backup PDF export of the deck on a USB drive.
