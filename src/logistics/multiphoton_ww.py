from __future__ import annotations
from typing import Any, Callable
import numpy as np
from scipy.sparse import csr_array, eye_array
from scipy.sparse.linalg import expm_multiply
from collections.abc import Sequence
from .ww import EmittersInWaveguide
from .bosons import (
    Basis,
    State,
    construct_basis,
    diagonals_with_energies,
    annihilation_operator,
    move_excitation_operator,
    number_operator,
    annihilation_operator,
)

type RealVector = np.ndarray[tuple[int], np.dtype[np.floating]]
type ComplexVector = np.ndarray[tuple[int], np.dtype[np.complexfloating]] | RealVector
type RealMatrix = np.ndarray[tuple[int, int], np.dtype[np.floating]]
type ComplexMatrix = (
    np.ndarray[tuple[int, int], np.dtype[np.complexfloating]] | RealMatrix
)


class EmittersInWaveguideMultiphotonWW(EmittersInWaveguide):
    """
    Description of a setup with 'N' emitters, placed at `positions[0]` to `positions[N-1]`
    with uniform couplings to a set of uniformly spaced frequency modes
        w[n+1] - w[n] = 2 * pi * L / c
    The momenta can take positive values if the cable is "open", or both positive and
    negative values otherwise.
    """

    n_excitations: list[int]
    basis: Basis
    dimension: int
    U: float

    def __init__(
        self,
        positions: Sequence[float] = [0.0],
        gamma: float | Sequence[float] = 0.1,
        Delta: float | Sequence[float] | None = None,
        driving: float | int | Sequence[float | int] | None = None,
        driving_frequency: float | int | None = None,
        n_modes: int = 100,
        n_excitations: int | list[int] = 1,
        c: float = 1.0,
        L: float = 1.0,
        PBC: bool = False,
        U: float = -1,
    ):
        """Create the setup description and associated information.

        Note about driving: each emitter can be affected by drivings of different strength, but not different frquency. This is
        because the coherent driving leads to a time dependent Hamiltonian, which can only be dealt with in the frame rotating with the
        frequency of the driving. For this reason, when driving freq is provided, every other frequency is shifted by this freq.

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
        U: float
            If U < 0, then the emitters are qubits, otherwise they are
            linear or nonlinear oscillators.

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
            positions, gamma, Delta, driving, driving_frequency, n_modes, c, L, PBC
        )
        self.U = U

        if isinstance(n_excitations, int):
            self.n_excitations = [n_excitations]
        else:
            self.n_excitations = n_excitations
        if self.U < 0:
            self.basis = construct_basis(self.n_emitters, self.n_modes, n_excitations)
        else:
            self.basis = construct_basis(
                0, self.n_emitters + self.n_modes, n_excitations
            )
        self.dimension = len(self.basis)

    def initial_state(self, state: str) -> RealVector:
        """Quantum state determining which qubits are excited."""
        if len(state) == 1:
            state *= self.n_emitters
        assert state.count("1") in self.n_excitations
        assert len(state) == self.n_emitters
        output = np.zeros(self.dimension)
        a_state: State = tuple(
            ndx for ndx in range(self.n_emitters) if state[ndx] == "1"
        )
        output[self.basis[a_state]] = 1.0
        return output
    


    def Hamiltonian(self) -> csr_array:
        H = diagonals_with_energies(
            self.basis,
            np.concatenate((self.Delta, self.wk)) - self.driving_frequency,
        )
        if self.U >= 0:
            for i in range(self.n_emitters):
                ni = number_operator(self.basis, mode=i)
                H += (self.U / 2) * (ni @ ni - ni)
        if self.driving is not None:
            for i, W_i in enumerate(self.driving):
                a = annihilation_operator(i, self.basis)
                H += 0.5 * W_i * (a + a.T)

        for i in range(self.n_emitters):
            for k in range(self.n_modes):
                operator: csr_array = self.gk[i, k] * move_excitation_operator(
                    self.basis, i, k + self.n_emitters
                )
                H += operator + operator.T.conjugate()
        return csr_array(H)

    def emitter_operator(self, operator: str, emitter: int) -> csr_array:
        match operator:
            case "s-":
                assert self.U < 0
                return annihilation_operator(emitter, self.basis)
            case "s+":
                assert self.U < 0
                return csr_array(annihilation_operator(emitter, self.basis).T)
            case "a":
                assert self.U >= 0
                return annihilation_operator(emitter, self.basis)
            case "ad":
                assert self.U >= 0
                return csr_array(annihilation_operator(emitter, self.basis).T)
            case "n":
                return number_operator(self.basis, emitter)
            case "sx":
                return self.emitter_operator("s-", emitter) + self.emitter_operator(
                    "s+", emitter
                )
            case "sz":
                return 2 * number_operator(self.basis, emitter) - eye_array(
                    self.dimension
                )
            case _:
                raise Exception(f"Unknown bosonic operator: '{operator}'")

    def total_excitation_operator(self):
        op = csr_array((self.dimension,self.dimension))
        for i in range(self.n_emitters):
            op+= 0.5*(eye_array(self.dimension)+self.emitter_operator('sz',i))
        for j in range(1,self.n_modes+1):
            a = annihilation_operator(j,self.basis)
            op += a.T@a
        return op
    
    def evolve(
        self,
        T: float,
        n_steps: int = 101,
        initial_state: ComplexVector | str | None = None,
        callback: Callable[[float, ComplexVector], Any] | None = None,
    ) -> tuple[RealVector, np.ndarray]:
        def default_callback(t: float, psi: ComplexVector) -> np.ndarray:
            return np.asarray([np.vdot(psi, n @ psi).real for n in emitter_op])

        if callback is None:
            emitter_op = [
                number_operator(self.basis, i) for i in range(self.n_emitters)
            ]
            callback = default_callback

        if initial_state is None:
            initial_state = "1" * max(self.n_excitations) + "0" * (
                self.n_emitters - max(self.n_excitations)
            )
        if isinstance(initial_state, str):
            psi = self.initial_state(initial_state)
        else:
            psi = np.asarray(initial_state)

        times = np.linspace(0, T, n_steps)
        idtH = self.Hamiltonian() * (-1j * (times[1] - times[0]))

        output: list[ComplexVector] = []
        for i, t in enumerate(times):
            if i:
                psi = expm_multiply(idtH, psi)
            output.append(callback(t, psi))  # type: ignore
        return times, np.asarray(output)
