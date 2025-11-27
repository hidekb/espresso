#
# Copyright (C) 2024 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import dataclasses
import typing
import ase
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
if typing.TYPE_CHECKING:
    from espressomd.system import System


@dataclasses.dataclass
class ASEInterface:
    """
    ASE interface for ESPResSo.
    """

    _system: typing.Union["System", None] = None

    def register_system(self, system):
        """Register the system."""
        self._system = system

    def get(self, folded=False, integrator="none") -> ase.Atoms:
        """Export the ESPResSo system particle data to an ASE atoms object."""
        particles = self._system.part.all()
        positions = np.copy(particles.pos_folded if folded else particles.pos)
        velocities = np.copy(particles.v)
        types = np.copy(particles.type)
        forces = np.copy(particles.f)
        if any(p.is_virtual() for p in particles):
            raise RuntimeError("ASE doesn't support virtual sites")

        # We assume here, for simplicity, that
        # the interaction is computed via ASE only.
        # In the previous implementation,
        # the ASE-based force evaluation proceeds as follows:
        #   atoms = system.ase.get()
        #   atoms.calc = lj
        #   system.part.all().ext_force = atoms.get_forces()
        #   system.integrator.run(1)
        #
        # Within system.integrator.run(1), the following sequence is executed:
        #   1. calculate_forces()
        #      p.force() = p.ext_force()
        #      p.force() += langevin_noise()
        #   2. integrator_step1()
        #   3. calculate_forces()
        #   4. integrator_step2()
        # If external_force() is evaluated using the posistion at time r(t),
        # then the particle forces are updated according to r(t)
        # before the velocities are advanced.
        # Consequently, in implemented symlectic Euler scheme,
        # r(t), v(t), F(t) are evolved as:
        #   F = F(t)
        #   v = v(t) + dt * F / m
        #   r = r(t) + dt * v
        # This correspond exactly the standard symplectic Euler integrator.
        #
        # In contrast, the currently implemented velocity verlet scheme updates
        # r(t), v(t) and F(t) as follows:
        #   F = F(t)
        #   v = v(t) + 0.5 * dt * F / m
        #   r = r(t) + dt * v
        #   F = F(t) (because p.ext_force() is not recomputed)
        #   v = v + 0.5 * dt *F / m
        # This doses NOT correspond to a true velocity verlet update.
        # Thereore this leads to an artificial increase in system temperature.
        #
        # To recover a correct velocity verlet implementation,
        # we manually compute r(t+dt) from r(t), v(t) and F(t),
        # and then evaluate p.ext_force() using r(t+dt).
        # if ASE integration is constructed as:
        #   atoms = system.ase.get(integrator="vv")
        #   atoms.calc = lj
        #   system.part.all().ext_force = atoms.get_forces()
        #   system.integrator.run(1, reuse_forces=True)
        # then the updates of r(t), v(t), F(t) become:
        #   F = F(t) (identical to the last step due to reuse_forces=true)
        #   v = v(t) + 0.5 * dt * F / m
        #   r = r(t) + dt * v
        #   F = F(t + dt) (because p.ext_force() is evaluated at the updated positions)
        #   v = v + 0.5 * dt *F / m
        # This sequence ideintical with the standard velocity verlet scheme.
        if integrator == "vv":
            v = velocities + 0.5 * forces * self._system.time_step / np.copy(particles.mass)[:, np.newaxis]
            pos = positions + v * self._system.time_step
        else:
            pos = positions

        atoms = ase.Atoms(
            #positions=positions,
            positions=pos,
            numbers=types,
            pbc=np.copy(self._system.periodicity),
            cell=np.copy(self._system.box_l),
        )
        atoms.calc = SinglePointCalculator(atoms, forces=forces)
        return atoms
