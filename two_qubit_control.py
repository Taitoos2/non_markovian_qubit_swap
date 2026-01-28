from __future__ import annotations
from typing import Any, Callable
import numpy as np
from scipy.sparse import csr_array, diags_array, block_array
from scipy.sparse.linalg import expm_multiply
from collections.abc import Sequence
from enum import Enum

type RealVector = np.ndarray[tuple[int], np.dtype[np.floating]]
type ComplexVector = np.ndarray[tuple[int], np.dtype[np.complexfloating]] | RealVector
type RealMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type ComplexMatrix = (
    np.ndarray[tuple[int, int], np.dtype[np.complexfloating]] | RealMatrix
)


class Waveguide(Enum):
    Ring = 0
    ChiralRing = 1
    Cable = 2


class EmittersInWaveguide:
    """
    Description of a setup with 'N' emitters, placed at `positions[0]` to `positions[N-1]`
    with uniform couplings to a set of uniformly spaced frequency modes
        w[n+1] - w[n] = 2 * pi * L / c
    The momenta can take positive values if the cable is "open", or both positive and
    negative values otherwise.
    """

    c: float
    L: float
    n_emitters: int
    positions: RealVector
    gamma: RealVector
    Delta: RealVector
    n_modes: int
    gk: ComplexMatrix | Callable[[float], ComplexMatrix]
    gk_base: ComplexMatrix
    wk: RealVector
    k: RealVector
    setup: Waveguide

    def __init__(
        self,
        positions: Sequence[float] = [0.0],
        gamma: float | Sequence[float] = 0.1,
        Delta: float | Sequence[float] | None = None,
        n_modes: int = 100,
        c: float = 1.0,
        L: float = 1.0,
        setup: Waveguide = Waveguide.Cable,
        g_time_modulation: Callable[[float], np.ndarray] | None = None,
    ):
        """Create the setup description and associated information.

        Arguments
        ---------
        positions: Sequence[float]
            Positions of emitters in the cable or ring
        gamma: float | Sequence[float]
            Spontaneous emission rate of emitters (default = 0.01)
        Delta: float | Sequence[float] | None
            Qubit frequencies. Values are defined in terms of the Free
            Spectral Range (default = n_modes / 2)
        L: float
            Total cable length (default = 1.0)
        c: float
            Speed of light (default = 1.0)
        n_modes: int
            Number of photon modes to keep (default = 100)
        setup: Waveguide (default = Waveguide.Cable)
            Type of cable where the photons live

        Computed
        --------
        FSR: float
            Free Spectral Range (separation between modes in frequency space)
        n_emitters: int
            Number of emitters (len(positions))
        wk, k: ndarray[tuple[int], dtype[floating]]
            Mode frequencies, photon momenta
        gk: ndarray[tuple[int, int], dtype[complexfloating]]
            Coupling to different modes
        """
        assert isinstance(n_modes, int) and (1 <= n_modes)
        assert isinstance(gamma, (float, int)) and 0 <= gamma
        assert isinstance(c, (float, int)) and 0 < c
        assert isinstance(L, (float, int)) and 0 < L
        self.c = c
        self.L = L
        self.n_emitters = n_emitters = len(positions)
        match setup:
            case Waveguide.Ring | Waveguide.ChiralRing:
                tau = L / c
                wavelength = L
            case Waveguide.Cable:
                tau = 2 * L / c
                wavelength = 2 * L
            case _:
                raise ValueError(f"Invalid waveguide type: {setup}")
        self.FSR = 2 * np.pi / tau

        self.positions = np.asarray(positions)
        self.n_emitters = len(positions)
        if isinstance(gamma, (float, int)):
            self.gamma = gamma * np.ones(n_emitters)
        else:
            self.gamma = np.asarray(gamma)
            assert len(gamma) == len(positions)
        if Delta is None:
            Delta = n_modes // 2
        if isinstance(Delta, (float, int)):
            self.Delta = Delta * np.ones(n_emitters) * self.FSR
        else:
            self.Delta = np.asarray(Delta) * self.FSR
            assert len(Delta) == len(positions)

        """
        We select `n_modes` around the average qubit resonance. The mode selection
        depends on the type of environment:

        - For a ring, we have frequencies spaced by 2𝜋/τ, where τ=L/c is the time
          for a photon to travel around the ring. Momenta are equispaced with
          separation 2𝜋/L and thus ω=c*k

        - For a cable, the allowed momenta are k_n = 𝜋/L, and the associated
          frequencies have a smaller spacing 𝜋c/L, which still can be written as
          2𝜋/τ, with τ=2L/c the round trip for a photon.

        The modes cannot lead to negative frequencies. Hence, if Delta is small
        we put more modes above than below Delta.
        """
        resonant_mode = round(np.mean(self.Delta) / self.FSR)
        min_mode = max(1, resonant_mode - n_modes // 2)
        modes = min_mode + np.arange(n_modes)
        self.wk = self.FSR * modes
        self.k = (2 * np.pi / wavelength) * modes
        if setup not in (Waveguide.Ring, Waveguide.Cable):
            self.wk = np.concatenate((self.wk, self.wk))
            self.k = np.concatenate((-self.k, self.k))
        self.n_modes = len(self.k)

        """
        At this stage, we have the right spectrum, but frequencies can be large,
        which confuses the computer's integrators (specially MPS). To speed up
        simulation it helps to move to a rotating frame with Delta."""
        self.frequency_shift = frequency_shift = resonant_mode * self.FSR
        self.Delta, self.unshifted_Delta = self.Delta - frequency_shift, self.Delta
        self.wk, self.unshifted_wk = self.wk - frequency_shift, self.wk

        """
        The coupling strength g_k can be deduced from the spontaneous emission rate
        and the normalization of the modes in the waveguide.
        """
        self.setup = setup
        x = self.positions.reshape(-1, 1)
        match setup:
            case Waveguide.Ring | Waveguide.ChiralRing:
                normalization = L
                self.gk_base = np.sqrt(
                    self.gamma.reshape(-1, 1) * c / normalization
                ) * np.exp(-1j * self.k * x)
            case Waveguide.Cable:
                normalization = L / 2
                self.gk_base = (
                    0.5
                    * np.sqrt(self.gamma.reshape(-1, 1) * c / normalization)
                    * np.cos(self.k * x)
                )

        # Setup time-dependent coupling (MINIMAL CHANGE)
        if g_time_modulation is None:
            self.gk = self.gk_base
        else:
            self.gk = lambda t: self.gk_base * g_time_modulation(t)

    def __str__(self) -> str:
        plural = "s" if self.n_emitters > 1 else ""
        return (
            f"Waveguide with {self.n_emitters} emitter{plural}:\n"
            + f" Decay rate gamma={self.gamma}\n"
            + f" Positions = {self.positions}\n"
            + f" Waveguide delay = {self.L / self.c}\n"
            + f" # modes = {self.n_modes}"
        )


class EmittersInWaveguideWW(EmittersInWaveguide):
    dimension: int

    def __init__(
        self,
        positions: Sequence[float] = [0.0],
        gamma: float | Sequence[float] = 0.1,
        Delta: float | Sequence[float] | None = None,
        n_modes: int = 100,
        c: float = 1.0,
        L: float = 1.0,
        setup: Waveguide = Waveguide.Cable,
        g_time_modulation: Callable[[float], np.ndarray] | None = None,
    ):
        """Create the setup description and associated information.

        Arguments
        ---------
        positions: Sequence[float]
            Positions of emitters in the cable or ring
        gamma: float | Sequence[float]
            Spontaneous emission rate of emitters (default = 0.01)
        Delta: float | Sequence[float] | None
            Qubit frequencies. Default value (default = None) implies looking
            for a frequency that allows reproducing spontaneous emission.
        L: float
            Total cable length (default = 1.0)
        c: float
            Speed of light (default = 1.0)
        n_modes: int
            Number of photon modes to keep (default = 100)
        setup: Waveguide
            Type of cable (default = Waveguide.Cable)

        Computed
        --------
        FSR: float
            Free Spectral Range (separation between modes in frequency space)
        n_emitters: int
            Number of emitters (len(positions))
        wk, k: ndarray[tuple[int], dtype[floating]]
            Mode frequencies, photon momenta
        gk: ndarray[tuple[int, int], dtype[complexfloating]]
            Coupling to different modes
        """
        super().__init__(
            positions, gamma, Delta, n_modes, c, L, setup, g_time_modulation
        )
        self.dimension = self.n_emitters + self.n_modes

    def initial_state(self, emitter: int = 0) -> RealVector:
        """Quantum state in which only one emitter has been excited."""
        assert 0 <= emitter < self.n_emitters
        output = np.zeros(self.dimension)
        output[emitter] = 1.0
        return output

    def Hamiltonian(self, t: float = 0.0) -> csr_array:
        """Construct the time-dependent Hamiltonian as a sparse matrix."""
        A00 = diags_array(
            self.Delta, offsets=0, shape=(self.n_emitters, self.n_emitters)
        )
        A11 = diags_array(self.wk, offsets=0, shape=(self.n_modes, self.n_modes))

        # Get time-dependent coupling (MINIMAL CHANGE)
        gk_t = self.gk(t) if callable(self.gk) else self.gk

        A01 = csr_array(gk_t)
        A10 = A01.T.conjugate()

        H = block_array([[A00, A01], [A10, A11]], format="csr")
        return H

    def evolve(
        self,
        T: float,
        n_steps: int = 101,
        psi0: ComplexVector | None = None,
        callback: Callable[[float, ComplexVector], Any] | None = None,
    ) -> tuple[RealVector, np.ndarray]:
        def default_callback(t: float, psi: ComplexVector) -> np.ndarray:
            return np.abs(psi[: self.n_emitters]) ** 2

        callback = callback or default_callback
        psi = np.asarray(psi0) if psi0 is not None else self.initial_state(0)
        assert len(psi) == self.dimension

        times = np.linspace(0, T, n_steps, dtype=np.float64)
        dt = times[1] - times[0] if n_steps > 1 else 0.0
        results = []

        for i, t in enumerate(times):
            if i > 0:
                H_t = self.Hamiltonian(t)
                idtH = H_t * (-1j * dt)
                psi = expm_multiply(idtH, psi)
            results.append(callback(t, psi))

        return times, np.asarray(results)
