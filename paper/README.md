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

## Compiling (Overleaf)
No LaTeX install needed.
1. Lynn creates an Overleaf project: **New Project → Upload Project**, with a ZIP of this `paper/`
   folder, then **Share** with Faheema and Aisha (edit access).
2. Everyone writes **only their own files** in Overleaf. Compiler: pdfLaTeX (the default).
3. At each writing checkpoint (CP5, CP6) Lynn downloads the Overleaf source
   (**Menu → Download → Source**) and commits it here, because the LaTeX source is a deliverable.

To compile locally instead: `latexmk -pdf main.tex` (needs TeX Live or MiKTeX).
