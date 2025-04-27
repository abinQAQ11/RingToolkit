from typing import Optional
from .functions import *
from .constants import *
from .elements import Drift
# ----------------------------------------------------------------------------------------------------------------------
class Lattice:
    def __init__(self, half_cell: Optional[list] = None, full_cell: Optional[list] = None,  **kwargs):
        if not (half_cell or full_cell):
            raise ValueError("Please enter either 'half_cell' or 'cell'")
        self.half_cell = half_cell
        self.full_cell = full_cell
        self.name = kwargs.get("name", "RING")
        self.energy = kwargs.get("energy")
        self.periodicity = kwargs.get("periodicity", 1)
        self.slice_length = 0.01

        self.chromat_x0 = self.chromat_y0 = self.chromat_x1 = self.chromat_y1 = 0.0
        self.i1 = self.i2 = self.i3 = self.i4 = self.i5 = 0.0
        self.d_terms_m = self.d_term = self.twiss = []
        self.f_rms = self.rdts = self.H = {}
        self.tune_x = self.tune_y = 0.0
        self.f3_rms = 0.0

    @property
    def cell(self):
        return (self.half_cell + self.half_cell[::-1]) if self.half_cell else self.full_cell

    @property
    def components(self) -> list:
        return list(dict.fromkeys(self.cell))

    @property
    def ring(self) -> list:
        return self.cell * self.periodicity

    @property
    def cell_length(self) -> float:
        return sum(item.length for item in self.cell)

    @property
    def long_drift(self):
        elem = self.cell[0]
        return elem.length

    @property
    def mediate_drift(self):
        elem = self.half_cell[-1]
        if isinstance(elem, Drift):
            return elem.length
        else:
            raise ValueError("Without setting the mid - straight section!")

    @property
    def circumference(self) -> float:
        return self.periodicity * self.cell_length

    @property
    def beam_rigidity(self) -> float:
        return self.beta * self.energy / clight

    @property
    def gamma(self) -> float:
        if self.energy is None:
            raise ValueError("Please enter the ring's energy.")
        return self.energy / e_mass

    @property
    def beta(self) -> float:
        return math.sqrt(1.0 - 1.0 / (self.gamma ** 2))

    @property
    def u0(self) -> float:
        return Cgamma / 2.0 / math.pi * self.energy ** 4 * self.i2

    @property
    def jx(self) -> float:
        return 1.0 - self.i4/self.i2

    @property
    def jy(self) -> float:
        return 1.0

    @property
    def je(self) -> float:
        return 2.0 + self.i4/self.i2

    @property
    def revolution_period(self) -> float:
        return self.circumference / (clight * self.beta)

    @property
    def ct(self) -> float:
        return 2.0 * self.energy / self.u0 * self.revolution_period

    @property
    def emittance(self) -> float:
        """ * 1e9 -> nm """
        if not self.twiss:
            return 0.0
        else:
            return Cq * self.gamma ** 2 * self.i5 / (self.jx * self.i2) * 1e9

    @property
    def alpha_c(self) -> float:
        return self.i1 / self.circumference

    @property
    def eta_c(self) -> float:
        return 1.0/ self.gamma**2 - self.alpha_c

    @property
    def tau(self) -> np.ndarray:
        return self.ct / np.array([self.jx, self.jy, self.je])

    @property
    def e_spread(self) -> float:
        return math.sqrt(Cq * self.gamma**2 * self.i3 / (self.je * self.i2))

    @property
    def cell_m66(self) -> np.ndarray:
        """Calculate cell transfer matrix using matrix product"""
        return np.linalg.multi_dot([elem.matrix for elem in reversed(self.cell)])

    @property
    def ring_m66(self) -> np.ndarray:
        """Calculate full ring transfer matrix"""
        return np.linalg.multi_dot([elem.matrix for elem in reversed(self.ring)])

    def slice_elem(self) -> None:
        """Slice elements excluding octupoles"""
        for item in self.components:
            if item.type != "octupole":
                item.slice(self.slice_length)

    def radiation_parameters(self) -> None:
        """Calculate radiation-related parameters"""
        self.slice_elem()
        self.twiss = get_twiss(self.cell, self.cell_m66)
        if self.twiss:
            tunes, integrals, chromat_0, chromat_1 = get_parameters(self.twiss)
            self.i1, self.i2, self.i3, self.i4, self.i5 = np.array(integrals) * self.periodicity
            self.chromat_x0, self.chromat_y0 = np.array(chromat_0) * self.periodicity
            self.chromat_x1, self.chromat_y1 = np.array(chromat_1) * self.periodicity
            self.tune_x, self.tune_y = np.array(tunes) * self.periodicity

    def driving_parameters(self):
        if self.twiss:
            self.d_terms_m, self.d_term, self.H = get_driving_terms(self.twiss, self.periodicity)
            self.rdts, self.f_rms, self.f3_rms = get_rdts(self.d_term)

    def __str__(self) -> str:
        return (f"                    Tunes: [{self.tune_x}  {self.tune_y}]\n"
                f"         Chromaticities 0: [{self.chromat_x0}  {self.chromat_y0}]\n"
                f"         Chromaticities 1: [{self.chromat_x1}  {self.chromat_y1}]\n"
                f" Momentum compact. factor: {self.alpha_c}\n"
                f"                   Energy: {self.energy} eV\n"
                f"       Energy loss / turn: {self.u0} eV\n"
                f" Radiation integrals - I1: {self.i1} m\n"
                f"                       I2: {self.i2} m^-1\n"
                f"                       I3: {self.i3} m^-2\n"
                f"                       I4: {self.i4} m^-1\n"
                f"                       I5: {self.i5} m^-1\n"
                f"        Natural Emittance: {self.emittance}\n"
                f"Damping partition numbers: [{self.jx}  {self.jy}  {self.je}]\n"
                f"            Damping times: {self.tau} seconds\n"
                f"            Energy spread: {self.e_spread}\n"
                f"                   f3_rms: {self.f3_rms}")
# ----------------------------------------------------------------------------------------------------------------------