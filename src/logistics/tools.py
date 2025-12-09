from __future__ import annotations
import numpy as np
from scipy.sparse import csr_array, eye_array, diags_array, issparse
from scipy.sparse import kron as spkron
import matplotlib.pyplot as plt


def mkron(A, *args):
    """Sequential Kronecker product of multiple matrices.

    mkron(A) = A
    mkron(A, B) = kron(A, B)
    mkron(A, B, C) = kron(A, kron(B, C))
    etc."""
    for B in args:
        if issparse(A) or issparse(B):
            A = spkron(A, B)  # type: ignore
        else:
            A = np.kron(A, B)
    return A


def a_operator(i: int, n: int, d: int = 3) -> csr_array:
    """generator of annihilation operators in the initial state (standard pauli matrices times the identity)"""
    return csr_array(
        mkron(
            eye_array(d**i),
            diags_array(np.sqrt(np.arange(1, d)), offsets=1),
            eye_array(d ** (n - i - 1)),
        )
    )


def n_operator(i: int, n: int, d: int = 3) -> csr_array:
    """generator of annihilation operators in the initial state (standard pauli matrices times the identity)"""
    return csr_array(
        mkron(
            eye_array(d**i),
            diags_array(np.arange(1, d), offsets=1),
            eye_array(d ** (n - i - 1)),
        )
    )


def vector_basis(s: str, d: int = 3) -> np.ndarray:
    match s:
        case "1":
            v = [0, 1]
        case "0":
            v = [1, 0]
        case "+":
            v = [1 / np.sqrt(2), 1 / np.sqrt(2)]
        case "-":
            v = [1 / np.sqrt(2), -1 / np.sqrt(2)]
        case _:
            raise Exception(f"Unknown quantum state label: {s}")
    return np.asarray(v + [0] * (d - len(v)), dtype=np.float64)


def vector_state(chain: str, N: int, d: int = 2):
    if len(chain) == 1 and chain in ["1", "0", "+", "-"]:
        chain = chain * N
    vectors = [vector_basis(s, d) for s in chain]
    state = vectors[0]
    for vector in vectors[1:]:
        state = np.kron(state, vector)
    return state


def set_plot_style() -> None:
    """
    Configure a clean, publication-quality Matplotlib style.
    - Uses serif fonts (like Times New Roman or DejaVu Serif)
    - Increases font and line sizes for readability
    - Applies a balanced color palette
    """
    plt.rcParams.update(
        {
            # ---- Fonts ----
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            # ---- Axes and Lines ----
            "axes.linewidth": 1.2,
            "axes.labelpad": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            # ---- Figure ----
            "figure.figsize": (8, 5),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            # ---- Colors ----
            "axes.prop_cycle": plt.cycler(  # type: ignore
                color=[
                    "#1f77b4",  # blue
                    "#ff7f0e",  # orange
                    "#2ca02c",  # green
                    "#d62728",  # red
                    "#9467bd",  # purple
                    "#8c564b",  # brown
                    "#e377c2",  # pink
                    "#7f7f7f",  # gray
                    "#bcbd22",  # lime
                    "#17becf",  # cyan
                ]
            ),
            # ---- Legend ----
            "legend.frameon": False,
            "legend.loc": "best",
        }
    )


__all__ = ["mkron", "a_operator", "n_operator", "set_plot_style"]


from numpy.polynomial import Polynomial

def DDE_analytical(gamma,phi,tau,t):
    
    alpha = 1j*phi/tau + 0.5*gamma 
    result =  np.exp(-alpha*t)*np.ones(len(t),dtype=complex)
    poli = Polynomial([1])
    N = int(t[-1]/tau)
    
    for n in range(1,N+1):
        dummie = poli.integ() 
        result += np.exp(-alpha*t)*np.exp(n*alpha*tau)*dummie(-gamma*(t-n*tau))*np.heaviside(t-n*tau,1)
        poli += dummie 

        
    return np.abs(result)**2