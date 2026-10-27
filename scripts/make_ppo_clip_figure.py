"""Plot the PPO clipped surrogate objective L^CLIP as a function of the probability ratio r.

    python scripts/make_ppo_clip_figure.py

writes docs/img/ppo_clip_objective.png (uses the same clip_eps = 0.2 as PPO_v2.py).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLIP_EPS = 0.2
OUT = Path(__file__).resolve().parent.parent / "docs" / "img" / "ppo_clip_objective.png"


def clipped_objective(ratio, advantage, eps=CLIP_EPS):
    return np.minimum(ratio * advantage, np.clip(ratio, 1 - eps, 1 + eps) * advantage)


def main():
    r = np.linspace(0.4, 1.6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)

    for ax, advantage, title in [
        (axes[0], +1.0, r"$A_t > 0$: action was better than expected"),
        (axes[1], -1.0, r"$A_t < 0$: action was worse than expected"),
    ]:
        ax.plot(r, r * advantage, "--", color="grey", label=r"unclipped $r_t A_t$")
        ax.plot(r, clipped_objective(r, advantage), color="crimson", linewidth=2.5, label=r"$\min(r_t A_t,\ \mathrm{clip}(r_t)A_t)$")
        ax.axvline(1 - CLIP_EPS, color="black", linewidth=0.6, linestyle=":")
        ax.axvline(1 + CLIP_EPS, color="black", linewidth=0.6, linestyle=":")
        ax.axvline(1, color="black", linewidth=0.6)
        ax.set_xlabel(r"$r_t = \pi_\theta(a_t|s_t)\,/\,\pi_{\theta_\mathrm{old}}(a_t|s_t)$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left" if advantage > 0 else "upper right")

    axes[0].set_ylabel(r"$L^{\mathrm{CLIP}}$ (per sample)")
    axes[0].annotate("gradient is zero\nbeyond $1+\\epsilon$", xy=(1.4, 1.2), xytext=(1.04, 0.62), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].annotate("gradient is zero\nbelow $1-\\epsilon$", xy=(0.6, -0.8), xytext=(0.9, -1.35), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.8))
    fig.suptitle(rf"PPO clipped objective, $\epsilon = {CLIP_EPS}$", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
