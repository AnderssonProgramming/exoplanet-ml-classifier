# IEEE Paper — `main.tex`

Self-contained LaTeX source for the conference paper that documents
the Exoplanet ML Classifier project. It uses the `IEEEtran`
document class and pulls every figure from `../figures/`.

## Compile

### Local TeX Live / MiKTeX

```bash
cd reports/paper
pdflatex main.tex
bibtex   main          # only needed if you switch to a .bib file
pdflatex main.tex
pdflatex main.tex
```

The bibliography is currently inline (`thebibliography` environment),
so the `bibtex` step can be skipped — two `pdflatex` runs are enough.

### Overleaf

1. Create a new project, choose **Upload Project**.
2. Upload `main.tex` and the entire `reports/figures/` directory,
   keeping the same relative path (`paper/main.tex` and
   `figures/*.png`).
3. Set the compiler to **pdfLaTeX**.
4. Click **Recompile**.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | Source. |
| `../figures/*.png` | Every figure referenced via `\graphicspath`. |

## Notes

* The paper uses the standard IEEEtran conference layout (no abstract
  page, two-column body, `IEEEtran` BibTeX style).
* Every numerical result quoted in the paper is sourced from the
  notebooks committed in `../../notebooks/` and reproduces with
  `RANDOM_SEED = 42`.
