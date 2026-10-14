"""
Interactive illustration of the note:
    "The nearest-neighbor tree in dimension 1"
    Exact law of the subtree of the second vertex and a delta^3 tail

Run with:  streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------------
# General settings
# ----------------------------------------------------------------------------
RED = "#e4572e"
BLUE = "#2e86e4"
INK = "#2b2d42"
GRID = "#d9dce3"

st.set_page_config(page_title="NN tree in dimension 1", page_icon="🌳", layout="wide")

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 11,
        "axes.edgecolor": "#8a8f9c",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ----------------------------------------------------------------------------
# Computations (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simulate_trees(n, m, seed):
    """
    Simulate m trees with n vertices on the circle T = R/Z.
    x_1 is red, x_2 is blue; every vertex takes the color of its nearest
    neighbor already present. Returns the blue fraction W_n = |D_n(2)|/n.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((m, n))
    color = np.empty((m, n), dtype=np.int8)
    color[:, 0] = 0  # red  : the root
    color[:, 1] = 1  # blue : vertex 2 and its descendants
    for j in range(2, n):
        xj = X[:, j][:, None]                    # (m, 1)
        prev = X[:, :j]                          # (m, j)
        d = np.abs(prev - xj)
        d = np.minimum(d, 1.0 - d)               # circular distance
        parent = np.argmin(d, axis=1)            # nearest neighbor
        color[:, j] = np.take_along_axis(color[:, :j], parent[:, None], axis=1)[:, 0]
    return color.mean(axis=1)


@st.cache_data(show_spinner=False)
def simulate_one_tree(n, seed):
    """One full realization (positions, colors, parents) for the drawing."""
    rng = np.random.default_rng(seed)
    X = rng.random(n)
    color = np.empty(n, dtype=np.int8)
    parent = -np.ones(n, dtype=int)
    color[0], color[1] = 0, 1
    for j in range(2, n):
        d = np.abs(X[:j] - X[j])
        d = np.minimum(d, 1.0 - d)
        p = int(np.argmin(d))
        parent[j] = p
        color[j] = color[p]
    return X, color, parent


@st.cache_data(show_spinner=False)
def sample_exact(m, seed):
    """Sample of the exact law W = a B1 + (1-a) B2."""
    rng = np.random.default_rng(seed)
    a = rng.random(m)
    b1 = rng.beta(2, 2, m)
    b2 = rng.beta(2, 2, m)
    return a * b1 + (1.0 - a) * b2


@st.cache_data(show_spinner=False)
def exact_tail(delta, Na=2500, Nb=1600):
    """
    P(W < delta) by quadrature, using the exact representation of the note:
       P(W < d) = 2 d * int_0^{1/(2d)} E[G(y_alpha)] d alpha,
       y_alpha = d (1 - alpha B1)_+ / (1 - d alpha),  G = F_{Beta(2,2)}.
    """
    b = np.linspace(0.0, 1.0, Nb)
    fb = 6.0 * b * (1.0 - b)
    db = b[1] - b[0]
    alpha = np.linspace(0.0, 0.5 / delta, Na)
    da = alpha[1] - alpha[0]
    den = np.maximum(1.0 - delta * alpha[:, None], 1e-12)
    y = delta * np.clip(1.0 - alpha[:, None] * b[None, :], 0.0, None) / den
    y = np.clip(y, 0.0, 1.0)
    G = 3.0 * y ** 2 - 2.0 * y ** 3
    Eg = (G * fb[None, :]).sum(axis=1) * db
    return 2.0 * delta * Eg.sum() * da


@st.cache_data(show_spinner=False)
def exact_tail_curve(deltas, Na=2500, Nb=1600):
    return np.array([exact_tail(float(d), Na, Nb) for d in deltas])


@st.cache_data(show_spinner=False)
def beta22_samples(n, seed):
    rng = np.random.default_rng(seed)
    return rng.beta(2, 2, n)


@st.cache_data(show_spinner=False)
def interval_realization(npoints, seed):
    """
    One realization of the bicolored interval (Lemma 4): left endpoint red,
    right endpoint blue; every point falls uniformly and takes the color of the
    nearest point already present. Returns positions, colors and the history of
    the bicolored intervals J_k.
    """
    rng = np.random.default_rng(seed)
    pos = [0.0, 1.0]
    col = [0, 1]
    J = [(0.0, 1.0)]
    for _ in range(npoints):
        u = float(rng.random())
        k = int(np.argmin([abs(u - p) for p in pos]))
        c = col[k]
        i = 0
        while i < len(pos) and pos[i] < u:
            i += 1
        pos.insert(i, u)
        col.insert(i, c)
        # new bicolored interval: between the last red and the first blue
        last_red = max((idx for idx, cc in enumerate(col) if cc == 0), default=0)
        first_blue = min((idx for idx, cc in enumerate(col) if cc == 1), default=len(pos) - 1)
        if first_blue > last_red:
            J.append((pos[last_red], pos[first_blue]))
    return np.array(pos), np.array(col), J


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🌳 The nearest-neighbor tree in dimension 1")
st.caption("Work note — interactive illustration of the results (unrefereed draft, 11 October 2026)")

c1, c2, c3 = st.columns(3)
c1.latex(r"W \;=\; a\,B_1 + (1-a)\,B_2")
c2.latex(r"a\sim\mathcal U(0,1),\quad B_i\sim\mathrm{Beta}(2,2)")
c3.latex(r"\mathbb P(W<\delta) = 6\,\delta^3 + O\!\big(\delta^4\log\tfrac1\delta\big)")
st.caption(
    "Points are dropped uniformly, one by one, on the circle $\\mathbb T=\\mathbb R/\\mathbb Z$, "
    "each linked to the nearest of the points already present. $W$ is the limiting fraction of "
    "vertices that descend from the **second** point."
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Parameters")
    n_viz = st.slider("Vertices for the tree drawing", 20, 300, 120, step=10)
    n_trees = st.slider("Vertices per tree (simulation)", 128, 2048, 1024, step=128)
    m_trees = st.slider("Number of simulated trees", 100, 4000, 600, step=100)
    seed = st.number_input("Random seed", value=20261011, step=1)
    st.divider()
    st.caption("Theory: $\\mathbb E W=\\tfrac12$, $\\operatorname{Var}W=\\tfrac1{30}$.")

rng_draw = int(seed)

tab_model, tab_beta, tab_law, tab_tail, tab_conj = st.tabs(
    ["🌳 Model & tree", "🅱️ Beta(2,2)", "📊 Law of W", "📉 δ³ tail", "🔭 Conjecture γ_d"]
)

# ============================================================================
# Tab 1 — Model & tree
# ============================================================================
with tab_model:
    st.subheader("One realization on the circle")
    X, color, parent = simulate_one_tree(n_viz, rng_draw)
    Wn = color.mean()

    x1 = X[0]
    x2 = X[1]
    a = (x2 - x1) % 1.0  # length of arc I1 (x1 -> x2, counterclockwise)

    col1, col2 = st.columns([1.15, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color=GRID, lw=1.5, zorder=0)
        # parent -> child edges
        for j in range(2, n_viz):
            p = parent[j]
            if p < 0:
                continue
            t1, t2 = 2 * np.pi * X[p], 2 * np.pi * X[j]
            ax.plot(
                [np.cos(t1), np.cos(t2)],
                [np.sin(t1), np.sin(t2)],
                color=BLUE if color[j] == 1 else RED,
                alpha=0.25, lw=0.6, zorder=1,
            )
        ang = 2 * np.pi * X
        cols = np.where(color == 1, BLUE, RED)
        ax.scatter(np.cos(ang), np.sin(ang), c=cols, s=26, zorder=3, edgecolor="white", linewidth=0.4)
        # seeds
        ax.scatter([np.cos(2 * np.pi * x1)], [np.sin(2 * np.pi * x1)], s=170, marker="*",
                   c=RED, edgecolor="white", linewidth=0.8, zorder=4, label="x₁ (root, red)")
        ax.scatter([np.cos(2 * np.pi * x2)], [np.sin(2 * np.pi * x2)], s=170, marker="*",
                   c=BLUE, edgecolor="white", linewidth=0.8, zorder=4, label="x₂ (blue)")
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"n = {n_viz} vertices   ·   W_n = {Wn:.3f}")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        st.metric("W_n (this tree)", f"{Wn:.3f}")
        st.metric("Arc length a = (x₂ − x₁) mod 1", f"{a:.3f}")
        st.markdown(
            "The two seeds cut the circle into two arcs $I_1$ (length $a$) and $I_2$ "
            "(length $1-a$). Each arc evolves **independently** as the bicolored interval "
            "of Lemma 4, with its red endpoint at $x_1$ and blue endpoint at $x_2$."
        )
        st.markdown(
            "A vertex is blue (hence a descendant of $x_2$) if and only if its position falls "
            "inside the **blue part** of one of the two arcs. Hence, in the limit:"
        )
        st.latex(r"W = a\,B_1 + (1-a)\,B_2")
        st.caption("Each blue fraction $B_k$ is a Beta(2,2), independent of $a$ and of the other arc.")
        st.info(
            "Tip: click 🔄 (Rerun) in the menu or change the seed to draw a new tree.",
            icon="🎲",
        )

# ============================================================================
# Tab 2 — Beta(2,2)
# ============================================================================
with tab_beta:
    st.subheader("The boundary point of a bicolored interval follows a Beta(2,2)")
    st.markdown(
        "**Lemma 4.** In an interval with red/blue endpoints, the points met from left to "
        "right are first red, then blue: the unique bicolored interval $J_k$ shrinks to a "
        "point $Y$. **Proposition 5 (Preater).** $Y/\\ell \\sim \\mathrm{Beta}(2,2)$, with density "
        "$6b(1-b)$ and distribution function $G(y)=3y^2-2y^3$."
    )

    cA, cB = st.columns(2)
    with cA:
        npts = st.slider("Points dropped in the interval", 5, 400, 120, step=5)
        pos, col, J = interval_realization(npts, rng_draw + 1)
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        ax.scatter(pos, np.zeros_like(pos), c=np.where(col == 1, BLUE, RED),
                   s=22, zorder=3, edgecolor="white", linewidth=0.4)
        ax.scatter([0, 1], [0, 0], marker="*", s=200, c=[RED, BLUE], zorder=4,
                   edgecolor="white", linewidth=0.8)
        # last red / first blue
        lr = np.max(np.where(col == 0)[0]) if (col == 0).any() else 0
        fb = np.min(np.where(col == 1)[0]) if (col == 1).any() else len(pos) - 1
        Y = pos[lr] if fb > lr else pos[lr]
        ax.axvspan(pos[lr], pos[fb] if fb > lr else pos[lr], color="#f2c14e", alpha=0.35,
                   label="final bicolored interval")
        ax.axvline(pos[lr], color=INK, ls=":", lw=1)
        ax.set_yticks([])
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(f"{npts} points — boundary Y ≈ {pos[lr]:.3f}")
        ax.legend(loc="upper center", frameon=False, fontsize=9)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("One realization: red to the left of Y, blue to the right (law 3y²−2y³).")

    with cB:
        N = st.slider("Beta(2,2) samples", 1000, 200000, 40000, step=1000)
        b = beta22_samples(N, rng_draw)
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        ax.hist(b, bins=60, density=True, color=BLUE, alpha=0.28, label="sample")
        x = np.linspace(0, 1, 300)
        ax.plot(x, 6 * x * (1 - x), color=BLUE, lw=2.2, label=r"$6b(1-b)$")
        ax.plot(x, 3 * x ** 2 - 2 * x ** 3, color=RED, lw=2.2, label=r"$G(y)=3y^2-2y^3$")
        ax.set_xlabel("b")
        ax.set_ylim(0, 2.1)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("Density and distribution function")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

# ============================================================================
# Tab 3 — Law of W
# ============================================================================
with tab_law:
    st.subheader("Density of W: simulated tree versus exact law")
    with st.spinner(f"Simulating {m_trees} trees with n = {n_trees}…"):
        Wn_sim = simulate_trees(n_trees, m_trees, rng_draw)
    W_exact = sample_exact(400000, rng_draw + 7)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.hist(Wn_sim, bins=50, density=True, color=BLUE, alpha=0.30,
            label=f"simulated tree (n = {n_trees}, {m_trees:,} trees)".replace(",", " "))
    ax.hist(W_exact, bins=180, density=True, histtype="step", color=INK, lw=1.6,
            label="exact law  a B₁ + (1−a) B₂")
    ax.set_xlabel("w")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_xlim(0, 1)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("E W_n (simulated)", f"{Wn_sim.mean():.4f}", delta=f"{Wn_sim.mean()-0.5:+.4f} vs 1/2")
    c2.metric("Var W_n (simulated)", f"{Wn_sim.var():.5f}", delta=f"{Wn_sim.var()-1/30:+.5f} vs 1/30")
    c3.metric("Sample size", f"{m_trees:,}".replace(",", " "))
    st.caption(
        "The theoretical variance is 1/30 = 0.03333; the visible gap comes from the binomial "
        "fluctuation at finite n (as in the note: E Wₙ = 0.4993, Var Wₙ = 0.03353)."
    )
    st.latex(r"\mathbb E W=\tfrac12,\qquad \operatorname{Var}W=\tfrac1{30}")

# ============================================================================
# Tab 4 — delta^3 tail
# ============================================================================
with tab_tail:
    st.subheader("Tail: P(W < δ) ∼ 6 δ³")
    dmin = st.select_slider(
        "Smallest δ", options=[0.02, 0.01, 0.005, 0.003], value=0.005
    )
    deltas = np.logspace(np.log10(0.5), np.log10(dmin), 30)
    with st.spinner("Quadrature of the exact law…"):
        tail = exact_tail_curve(deltas)

    # simulated points on the tree
    Wn_sim = simulate_trees(n_trees, m_trees, rng_draw)
    d_probe = np.array([0.1, 0.05, 0.025])
    sim_ratio = np.array([(Wn_sim < d).mean() / d ** 3 for d in d_probe])
    ex_ratio = np.array([exact_tail(float(d)) / d ** 3 for d in d_probe])

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(deltas, tail / deltas ** 3, color=INK, lw=2.4,
            label="exact law  a B₁ + (1−a) B₂,  Bᵢ ~ Beta(2,2)")
    ax.plot(d_probe, sim_ratio, "o", color=BLUE, ms=8, label=f"simulated tree (n = {n_trees})")
    ax.axhline(6.0, color=RED, ls="--", lw=1.6, label="limit 6")
    ax.set_xscale("log")
    ax.set_xlabel("δ")
    ax.set_ylabel(r"$\mathrm{P}(W<\delta)\,/\,\delta^3$")
    ax.legend(frameon=False)
    ax.set_title("Slow convergence in δ log(1/δ), consistent with the error term")
    ax.invert_xaxis()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.markdown("**Comparison with the table in the note** (exact law by quadrature):")
    tbl = {"δ": ["0.1", "0.05", "0.025", "0.0125", "0.005"]}
    tbl["P(W<δ)/δ³ — exact law"] = [f"{exact_tail(d)/d**3:.2f}" for d in [0.1, 0.05, 0.025, 0.0125, 0.005]]
    st.table(tbl)
    st.caption("The note reports 6.64 / 6.56 / 6.40 / 6.26 / 6.13. The convergence to 6 is slow, in δ log(1/δ).")

# ============================================================================
# Tab 5 — Conjecture gamma_d
# ============================================================================
with tab_conj:
    st.subheader("Conjecture in dimension d ≥ 2: γ_d = 1 + 2/d")
    st.markdown(
        "On the torus $\\mathbb T^d$, simulations suggest $\\mathbb P(W<\\delta)\\asymp\\delta^{\\gamma_d}$ "
        "with $\\gamma_d = 1 + 2/d$. The factor **1** comes from the $\\Theta(\\delta)$ probability that the two "
        "seeds are at distance $\\delta^{1/d}$; the exponent **2/d** from a confinement of the blue region of "
        "exponent $\\kappa_d=2$ independent of the dimension. As $d\\to\\infty$ one recovers the exponent 1 of "
        "uniform attachment."
    )
    d = np.arange(1, 21)
    gamma = 1 + 2.0 / d
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.plot(d, gamma, "-o", color=INK, ms=5, label=r"conjecture $\gamma_d = 1 + 2/d$")
    ax.plot([1], [3.0], "*", color=RED, ms=20, label="d = 1: proved (Theorem 2, exponent 3)")
    ax.plot([2], [2.0], "s", color=BLUE, ms=9, label="d = 2: measured local exponent ≈ 2.0")
    ax.plot([3], [1.7], "D", color="#7dbb3a", ms=9, label="d = 3: measured local exponent ≈ 1.7")
    ax.axhline(1.0, color=RED, ls="--", lw=1.4, label="limit d→∞: uniform attachment (exponent 1)")
    ax.set_xlabel("dimension d")
    ax.set_ylabel(r"exponent $\gamma_d$")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("The case d = 1 is the only proved one")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.info(
        "Consequence for Jordan centrality: in dimension 1, the confidence sets for the root "
        "have size $K_\\Psi(\\varepsilon)\\asymp\\varepsilon^{-1/3}$ (up to logarithms), to be compared "
        "with $1/\\varepsilon$ for uniform attachment.",
        icon="🧭",
    )

st.divider()
st.caption(
    "Based on the note \"The nearest-neighbor tree in dimension 1\" and J. Preater, "
    "*A species of voter model driven by immigration*, Statist. Probab. Lett. 79 (2009)."
)
