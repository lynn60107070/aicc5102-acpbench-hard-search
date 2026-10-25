# Paper

NeurIPS 2026 template, `preprint` option. **At most 6 pages of main content** (the brief's limit, not
the template's 9). References and appendices don't count.

## Who writes what

| File | Owner | Due |
|---|---|---|
| `sections/00_abstract.tex`, `01_introduction.tex`, `04_method.tex`, `07_discussion.tex`, `09_conclusion.tex`, `B_prompts.tex` | Lynn (L6) | CP5, Fri 30 Oct |
| `sections/02_related_work.tex`, `03_reproduction.tex`, `A_fidelity_log.tex`, `refs.bib` | Faheema (F6, F7) | refs CP3, text CP5 |
| `sections/05_experimental_setup.tex`, `06_results.tex`, `08_limitations.tex`, `tables/`, `figures/` | Aisha (A8) | CP5 |
| `sections/C_contributions.tex`, `D_ai_use.tex` | Lynn (L7), input from all | CP6, Sun 1 Nov |

Each section file starts with a comment listing what to cover and the numbers to use. Delete the
comment's `\todo{...}` line when you write the section. Before submitting, there must be **no
`\todo` left** anywhere.

## Rules
- **Tables and figures come from scripts** (`experiments/analyze.py` → `paper/tables/*.tex`,
  `paper/figures/*.pdf`), never typed by hand. Re-run the script and the paper updates.
- Cite with `\citep{key}` (in brackets) or `\citet{key}` (as a noun); keys are in `refs.bib`.
- Numbers in the text must match `results/final/`. If in doubt, check the `summary.json`.

## How we write: GitHub is the source, Overleaf compiles
**Main Overleaf project (Lynn's):** https://www.overleaf.com/project/6adb43bef2e6e9267b35af13

1. Edit **only your own section files** in `paper/sections/` (and Faheema: `refs.bib`), on your
   branch, then push and open a PR, the same as for code.
2. Each paper PR is compile-checked (LaTeX errors, broken `\cite`/`\ref`, page count) before merging.
3. After merging, Lynn copies the changed files into the Overleaf project, which produces the PDF.
4. Optional, to see your section compile while writing: upload a ZIP of this `paper/` folder to a
   free personal Overleaf project (**New Project → Upload Project**). Use it only for previewing;
   always copy your final text back into the repo.

LaTeX tips: escape `%`, `&`, `_`, `#` as `\%`, `\&`, `\_`, `\#`; cite with `\citep{key}` using keys
from `refs.bib`; refer to sections with `Section~\ref{sec:...}`.

To compile locally instead: `latexmk -pdf main.tex` (needs TeX Live or MiKTeX).
