# Finance Intelligence Platform

**Author / Written by: Anthony De Ruiter**

Adaptive Portfolio Intelligence System (APIS) — a data-driven finance capstone applying *Python for Data Analysis* (Wes McKinney) and *Python for Finance* (Yves Hilpisch) to Canadian energy equities (TSX-listed stocks). The project demonstrates configuration, portfolio and risk analytics, stochastic modeling, derivatives pricing, and position sizing in a single reproducible report with code and outputs.

---

## Live site

**https://anthonyddrr.github.io/finance-intelligence-platform/**

The site serves a landing page and a full report (markdown, code, and executed outputs) suitable for portfolios and recruiters.

---

## Steps to view and use

1. **View online:** Open the live site above and click **View full report with code and outputs** to read the report in your browser.
2. **Run locally:** Clone the repo, install dependencies (`pip install -r requirements.txt`), then open `full_analysis.ipynb` in Jupyter to run or edit the notebook.
3. **Rebuild the report:** From the repo root, run `python scripts/build_notebook.py` to regenerate the notebook from the build script, then `jupyter nbconvert --execute --to html full_analysis.ipynb --output report.html` to produce an updated HTML report.
4. **Deploy:** Pushes to `main` trigger GitHub Actions to execute the notebook and deploy the site to GitHub Pages.

---

## Repository structure

| Item | Purpose |
|------|--------|
| `index.html` | Landing page (overview and link to report) |
| `report.html` | Full report with code and outputs (built from the notebook) |
| `full_analysis.ipynb` | Jupyter notebook source for the report |
| `scripts/build_notebook.py` | Script that generates the notebook structure and narrative |
| `requirements.txt` | Python dependencies for running and building the report |
| `.nojekyll` | Instructs GitHub Pages to serve static files as-is |
| `.github/workflows/` | Workflow to build the report and deploy to GitHub Pages |

---

## About the project

- **Scope:** 18 sections mapping to Python for Data Analysis and Python for Finance (configuration, OOP, NumPy, pandas, risk analytics, BSM options, GBM, Kelly criterion, and coverage map).
- **Custom classes:** Portfolio, RiskEngine, BSMOption, GeometricBrownianMotion.
- **Domain:** Canadian energy equities (e.g. CNQ, Suncor, Cenovus); all code is original work applied to this domain.
- **Tech:** Python 3, NumPy, pandas, SciPy, Matplotlib, yfinance.

---

**Author / Written by: Anthony De Ruiter**

---

**Note:** The short description under the repository name on GitHub is set in **Settings → General → Description**; use a professional one-line summary there.
