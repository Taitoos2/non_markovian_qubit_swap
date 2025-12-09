from __future__ import annotations
import itertools
from typing import Iterable
import numpy as np
from numpy.typing import NDArray, ArrayLike
from scipy.sparse import csr_array, diags_array

"""State: A sorted tuple of integers, denoted which modes are occupied with excitations"""
State = tuple[int, ...]

"""Basis: A dictionary mapping states to positions in a vector state"""
Basis = dict[State, int]


def construct_basis(qubits: int, bosons: int, excitations: int | list[int]) -> Basis:
    """
    Construct the basis for a given number of 'qubits' and 'bosons' with a fixed
    number of 'excitations'.

    Creates the basis of all possible occupied modes for the given number of
    `excitations`. Each state is represented by a sorted tuple of the modes that
    are occupied. Modes 0 up to (qubits-1) are hard-core boson modes and thus
    can only appear once. All other modes are ordinary bosons and may host 0 up
    to `excitations`.

    Args:
        qubits (int): Number of qubits or hard-core boson modes >= 0
        bosons (int): Number of bosonic modes >= 0
        excitations (int): Number of excitations >= 0

    Returns:
        basis: Map from configurations to an index in the Hilbert space basis
    """
    if isinstance(excitations, list):
        return concatenate_basis(qubits, bosons, excitations)

    assert isinstance(excitations, int) and 0 <= excitations

    def make_bosonic_states(n_modes: int, excitations: int):
        return itertools.combinations_with_replacement(np.arange(n_modes), excitations)

    def select_hardcore_boson_states(qubits: int, states: Iterable) -> Iterable:
        return itertools.filterfalse(lambda x: unphysical_state(x, qubits), states)

    return {
        v: i
        for i, v in enumerate(
            select_hardcore_boson_states(
                qubits, make_bosonic_states(qubits + bosons, excitations)
            )
        )
    }


def unphysical_state(configuration: State, qubits: int) -> bool:
    """Given a sorted list of occupied modes, check whether a qubit mode
    appears more than once.

    Args:
        configuration (State): Sorted tuple with the occupied modes
        qubits (int): Number of hard-core boson modes >= 0

    Returns:
        bool: False if state is physical, True if it is not.
    """
    last = -1
    for mode in configuration:
        if mode >= qubits:
            return False
        if last == mode:
            return True
        last = mode
    return False


def number_operator(basis: Basis, mode: int) -> csr_array:
    """Create the number operator for the given 'mode' in this 'basis'.

    Args:
        basis (Basis): basis of bosonic states
        mode (int): mode index (>= 0)

    Returns:
        csr_matrix: diagonal matrix representing the occupation of 'mode'
    """
    L = len(basis)
    rows = np.arange(L)
    occupation = np.zeros((L,))
    for state, ndx in basis.items():
        occupation[ndx] = state.count(mode)
    return csr_array((occupation, (rows, rows)), shape=(L, L))  # type: ignore


def mode_occupations(basis: Basis, wavefunction: ArrayLike) -> NDArray[np.double]:
    """Compute the average of the modes occupations for all modes.

    Assume 'basis' is the bosonic basis for 'N' modes and that 'wavefunction'
    is a 1D vector for a state in this basis. Then 'mode_occupation' will
    return a vector of size 'N' with the average of the occupation number
    operators for each mode.

    If 'wavefunction' is an N-dimensional array, we assume that the first index
    is associated to the physical dimension of the basis and the same task
    is performed for all values of the 2 to N indices.

    Args:
        basis (Basis): basis of bosonic states
        wavefunction (ArrayLike): 1D wavefunction, or N-D collection of them
    Returns:
        occupations (NDArray): 1D vector of occupation numbers, or N-D collection
        of the computations for different wavefunctions.
    """
    wavefunction = np.asarray(wavefunction)
    num_modes = max(max(state) for state in basis) + 1
    probability = np.abs(wavefunction.reshape(len(basis), -1)) ** 2
    output = np.zeros((num_modes, probability.shape[1]))
    for state, ndx in basis.items():
        for mode in state:
            output[mode, :] += probability[ndx, :]
    return output.reshape(num_modes, *wavefunction.shape[1:])


def move_excitation_operator(
    basis: Basis, origin_mode: int, destination_mode: int
) -> csr_array:
    """
    Create a sparse matrix representation of an operator that moves an
    excitation from mode 'origin' to mode 'destination'.

    Args:
        origin (int): Index of the origin mode
        destination (int): Index of the destination mode
        basis (dict): Collection of physical states (see: construct_basis)

    Returns:
        Operator (csr_matrix): Matrix representation of the quantum operator
    """

    row: list[int] = []
    column: list[int] = []
    coefficient: list[np.floating] = []

    for state in basis:
        origin_occupation = state.count(origin_mode)
        if origin_occupation:
            ndx = state.index(origin_mode)
            transformed_state = tuple(
                sorted(state[:ndx] + state[ndx + 1 :] + (destination_mode,))
            )
            if transformed_state in basis:
                destination_occupation = transformed_state.count(destination_mode)
                row.append(basis[transformed_state])
                column.append(basis[state])
                coefficient.append(np.sqrt(origin_occupation * destination_occupation))

    return csr_array((coefficient, (row, column)), shape=(len(basis), len(basis)))  # type: ignore


def diagonals_with_energies(basis: Basis, frequencies: np.ndarray) -> csr_array:
    """Matrix of eigenenergies of occupied modes.

    Args:
        basis (dict): Collection of physical states (see: construct_basis)
        frequencies (np.ndarray): Energy of each qubit or photon mode (in this order)

    Returns:
        Operator (csr_matrix): Diagonal matrices with the energy of each basis state.
    """
    energy = np.empty(len(basis))  # initialize the energy coresponding to each vector.

    for occupation, pos in basis.items():
        energy[pos] = np.sum(frequencies[list(occupation)])

    return diags_array(energy, offsets=0, format="csr")


def concatenate_basis(qubits: int, bosons: int, excitations_list: list[int]) -> Basis:
    """
    Create a basis with a variable number of excitations, from 0 up to 'excitations',
    using the given number of 'qubits' and 'bosons' modes.

    Args:
        qubits (int): Number of qubits or hard-core boson modes >= 0
        bosons (int): Number of bosonic modes >= 0
        excitations (int): Number of excitations of the biggest subspace >= 0

    Returns:
        basis: Collection of all the states that constitute the basis properly sorted.
    """
    basis = {}
    offset = 0
    for excitations in excitations_list:
        subspace = construct_basis(qubits, bosons, excitations)
        for state, index in subspace.items():
            basis[state] = index + offset
        offset += len(subspace)
    return basis


def annihilation_operator(mode: int, basis: Basis) -> csr_array:
    """Creates a sparse matrix representation of an operator that erases an excitation
    from 'mode'.

    This function creates the sparse matrix representation of a Fock anihilation
    operator in the given 'basis'.

    For this function to make sense, 'basis' must contain states from 0 up to a
    maximum number of excitations. Otherwise, when we remove 'mode' we will not find
    a good state to map it to.

    Args:
        mode (int): Index of the mode from which the excitation is going to be removed
        basis (Basis): Collection of physical states (see: construct_basis)

    Returns:
        Operator (csr_matrix): Matrix representation of the quantum operator
    """
    row: list[int] = []
    column: list[int] = []
    coefficient: list[int] = []
    for state in basis:
        # We run over all states in the basis. If the 'mode' is present
        # we construct a new state where we have removed _one_ occurrence of
        # this mode, thus eliminating a particle.
        count = state.count(mode)
        if count:
            # Since the modes are sorted in the state, we can assume the outcome
            # is sorted.
            mode_position = state.index(mode)
            transformed = tuple(state[:mode_position] + state[mode_position + 1 :])
            row.append(basis[transformed])
            column.append(basis[state])
            coefficient.append(np.sqrt(count))

    return csr_array((coefficient, (row, column)), shape=(len(basis), len(basis)))  # type: ignore
