# Nearest-neighbor tree in dimension 1 — interactive illustration

A small [Streamlit](https://streamlit.io) app that illustrates the results of the work note
*"The nearest-neighbor tree in dimension 1 — exact law of the subtree of the second vertex and
a δ³ tail"* (`arbre_ppv_dimension1.pdf`).

## The result in one line

Points are dropped uniformly, one by one, on the circle `T = R/Z`; each new point is linked to the
nearest point already present (the seeds `x₁` red and `x₂` blue, and every later point inherits its
parent's color). Let `W` be the limiting fraction of vertices that descend from the second point.
Then

```
W = a·B₁ + (1 − a)·B₂,     a ~ U(0,1),   B₁, B₂ ~ Beta(2,2),   all independent,
```

so that `E W = 1/2`, `Var W = 1/30` and

```
P(W < δ) = 6·δ³ + O(δ⁴ log(1/δ))     (δ → 0).
```

The Beta(2,2) law of the boundary point of a bicolored interval is Preater's Theorem 4 (2009);
the circle decomposition and the tail bound are short consequences.

## Tabs

| Tab | What it shows |
| --- | --- |
| 🌳 **Model & tree** | One realization on the circle (red/blue vertices, nearest-neighbor edges), `W_n` and the arc length `a = (x₂ − x₁) mod 1`. |
| 🅱️ **Beta(2,2)** | One bicolored-interval realization (Lemma 4) and the density `6b(1−b)` / CDF `3y²−2y³` against a sample. |
| 📊 **Law of W** | Histogram of `W_n` from real tree simulations overlaid with the exact law `a·B₁ + (1−a)·B₂`, plus `E W` and `Var W`. |
| 📉 **δ³ tail** | `P(W<δ)/δ³` by quadrature of the exact law against simulated points, with the `→ 6` limit and a table reproducing the note. |
| 🔭 **Conjecture γ_d** | `γ_d = 1 + 2/d`, with `d = 1` proved, `d = 2, 3` measured, and the `d → ∞` uniform-attachment limit. |

All computations are cached; sliders in the sidebar control the sample size, tree size and random seed.

## Run

```bash
# with the bundled virtualenv
.venv/bin/streamlit run app.py

# or, from scratch
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Then open <http://localhost:8501>.

## Files

- `app.py` — the Streamlit application.
- `requirements.txt` — `streamlit`, `numpy`, `matplotlib`.
- `arbre_ppv_dimension1.pdf` — the source note.

## Notes

- The exact tail is computed by quadrature using the representation
  `P(W<δ) = 2δ ∫₀^{1/(2δ)} E[G(y_α)] dα` with `y_α = δ(1−αB₁)₊/(1−δα)`; the naive direct
  double integral loses accuracy in the boundary layer `a → 1`.
- The tree simulation is `O(m·n²)`; large `n`/number of trees take a few seconds on first load,
  then the results are cached.
