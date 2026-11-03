"""Analysis re-exports."""
from pcadiv.analysis.plots import (
    plot_delta_vs_reference,
    plot_diversity_vs_budget,
    plot_spectrum,
)
from pcadiv.analysis.spectral import (
    classify_signal_noise,
    explained_variance_at_k,
    marchenko_pastur_bounds,
    participation_ratio,
    pool_eigenvalues,
    spectral_summary,
)

__all__ = [
    "classify_signal_noise",
    "explained_variance_at_k",
    "marchenko_pastur_bounds",
    "participation_ratio",
    "plot_delta_vs_reference",
    "plot_diversity_vs_budget",
    "plot_spectrum",
    "pool_eigenvalues",
    "spectral_summary",
]