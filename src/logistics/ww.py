from __future__ import annotations
from typing import Any, Callable
import numpy as np
from scipy.sparse import csr_array, diags_array, block_array
from scipy.sparse.linalg import expm_multiply
from collections.abc import Sequence

type RealVector = np.ndarray[tuple[int], np.dtype[np.floating]]
type ComplexVector = np.ndarray[tuple[int], np.dtype[np.complexfloating]] | RealVector
type RealMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type ComplexMatrix = (
    np.ndarray[tuple[int, int], np.dtype[np.complexfloating]] | RealMatrix
)


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
    driving: RealVector | None
    driving_frequency: float | int
    n_modes: int
    gk: ComplexMatrix
    wk: RealVector
    k: RealVector

    def __init__(
        self,
        positions: Sequence[float] = [0.0],
        gamma: float | Sequence[float] = 0.1,
        Delta: float | Sequence[float] | None = None,
        driving: float | int | Sequence[float | int] | None = None,
        driving_frequency: float | int | None = None,
        n_modes: int = 100,
        c: float = 1.0,
        L: float = 1.0,
        PBC: bool = False,
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
        driving: float | int | Sequence[float|int] | None
            Driving strength acting on the first qubit (if a number) or on
            all qubits (if a sequence of numbers). (default=None, no driving)
        driving_frequency: float | int
            Frequency of the driving (default=0.0).
        L: float
            Total cable length (default = 1.0)
        c: float
            Speed of light (default = 1.0)
        n_modes: int
            Number of photon modes to keep (default = 100)
        PBC: bool
            Whether the cable is closed (default = False)

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
        tau = (L / c) if PBC else (2 * L / c)
        self.FSR = 2 * np.pi / tau  # I think this is wrong, I think 

        self.positions = np.asarray(positions)
        self.n_emitters = len(positions)
        if Delta is None:
            n = min(round(self.FSR / gamma), n_modes // 2) + 1
            Delta = n * self.FSR
        if isinstance(gamma, (float, int)):
            self.gamma = gamma * np.ones(n_emitters)
        else:
            self.gamma = np.asarray(gamma)
            assert len(gamma) == len(positions)
        if isinstance(Delta, (float, int)):
            self.Delta = Delta * np.ones(n_emitters)
        else:
            self.Delta = np.asarray(Delta)
            assert len(Delta) == len(positions)

        """
        We select `n_modes` around the average qubit resonance. The mode selection
        depends on the type of environment:

        - For a ring, we have frequencies spaced by 2𝜋/τ, where τ=L/c is the time
          for a photon to travel around the ring. Momenta are equispaced with
          separation 2𝜋/L and thus ω=c*k

        - For a cable, the allowed momenta are k_n = n𝜋/L, and the associated
          frequencies have a smaller spacing 𝜋c/L, which still can be written as
          2𝜋/τ, with τ=2L/c the round trip for a photon.
        """
        n_reson = round(np.mean(self.Delta) / self.FSR)
        self.wk = self.FSR * (np.arange(-n_modes // 2, n_modes // 2) + n_reson + 0.0)
        self.k = self.wk / c
        if PBC:
            self.wk = np.concatenate((self.wk, self.wk))
            self.k = np.concatenate((-self.k, self.k))
        self.n_modes = len(self.k)

        """
        The coupling strength g_k can be deduced from the spontaneous emission rate
        and the normalization of the modes in the waveguide.
        """
        normalization = 2 * L
        self.gk = np.sqrt(gamma * c / normalization) * np.exp(
            -1j * self.k * self.positions.reshape(-1, 1)
        )

        if driving is not None:
            if isinstance(driving, (float, int)):
                driving = [driving] + [0.0] * (self.n_emitters - 1)
            self.driving = np.asarray(driving)
        else:
            self.driving = None
            driving_frequency = 0.0
        if driving_frequency is None:
            driving_frequency = float(np.min(self.Delta))
        self.driving_frequency = driving_frequency

    def __str__(self) -> str:
        plural = "s" if self.n_emitters > 1 else ""
        return (
            f"Waveguide with {self.n_emitters} emitter{plural}:\n"
            + f" Decay rate gamma={self.gamma}\n"
            + f" Positions = {self.positions}\n"
            + f" Waveguide delay = {self.L / self.c}\n"
            + f" # modes = {self.n_modes}\n"
            + f" driving = {self.driving}\n"
            + f" at frequency = {self.driving_frequency}\n"
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
        PBC: bool = False,
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
        PBC: bool
            Whether the cable is closed (default = False)

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
        super().__init__(positions, gamma, Delta, n_modes, c, L, PBC)
        self.dimension = self.n_emitters + self.n_modes

    def initial_state(self, emitter: int = 0) -> RealVector:
        """Quantum state in which only one emitter has been excited."""
        assert 0 <= emitter < self.n_emitters
        output = np.zeros(self.dimension)
        output[emitter] = 1.0
        return output

    def Hamiltonian(self) -> csr_array:
        A00 = diags_array(self.Delta, offsets=0)
        A11 = diags_array(self.wk, offsets=0)
        A01 = csr_array(self.gk)
        A10 = A01.T.conjugate()
        return block_array([[A00, A01], [A10, A11]], format="csr")

    def evolve(
        self,
        T: float,
        n_steps: int = 101,
        psi0: ComplexVector | None = None,
        callback: Callable[[float, ComplexVector], Any] | None = None,
    ) -> tuple[RealVector, np.ndarray]:
        def default_callback(t: float, psi: ComplexVector) -> np.ndarray:
            return np.abs(psi[: self.n_emitters]) ** 2

        if callback is None:
            callback = default_callback
        if psi0 is None:
            psi = self.initial_state(0)
        else:
            psi = np.asarray(psi0)

        times = np.linspace(0, T, n_steps)
        idtH = self.Hamiltonian() * (-1j * (times[1] - times[0]))
        output: list[np.ndarray] = []
        for i, t in enumerate(times):
            if i:
                psi = expm_multiply(idtH, psi)
            output.append(callback(t, psi))  # type: ignore
        return times, np.asarray(output)
