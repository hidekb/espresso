#
# Copyright (C) 2013-2022 The ESPResSo project
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
import unittest as ut
import unittest_decorators as utx
import numpy as np
import espressomd
import tests_common


@utx.skipIfMissingFeatures(["NPT", "LENNARD_JONES"])
class IntegratorNPT(ut.TestCase):

    """This compares pressure and compressibility of a LJ system against
       expected values."""
    system = espressomd.System(box_l=[1.0, 1.0, 1.0])

    def setUp(self):
        self.system.box_l = [5] * 3
        self.system.time_step = 0.01
        self.system.cell_system.skin = 0.25

    def tearDown(self):
        self.system.part.clear()
        self.system.thermostat.turn_off()
        self.system.integrator.set_vv()

    def test_compressibility_and_pressure(self):
        system = self.system
        system.box_l = [5.86326165] * 3

        data = np.genfromtxt(tests_common.data_path("npt_lj_system.data"))
        p_ext = 1.0

        system.part.add(pos=data[:, :3], v=data[:, 3:])
        system.non_bonded_inter[0, 0].lennard_jones.set_params(
            epsilon=1, sigma=1, cutoff=1.12246, shift=0.25)

        system.thermostat.set_npt(kT=1.0, gamma0=2, gammav=0.004, seed=42)
        system.integrator.set_isotropic_npt(ext_pressure=p_ext, piston=0.0001)

        system.integrator.run(800)
        avp = 0
        avpV_sim = 0
        avp_sim_vir = 0
        avpV_inst = 0
        avp_inst_vir = 0
        n = 30000
        skip_p = 8
        ls = np.zeros(n)
        for t in range(n):
            system.integrator.run(2)
            if t % skip_p == 0:
                volume = float(np.prod(system.box_l))
                pressure = system.analysis.pressure()
                avp += pressure['total']
                avpV_sim += pressure['kinetic'] * volume
                avp_sim_vir += pressure['non_bonded']

                p_inst_vir = system.analysis.get_instantaneous_pressure_virial()
                avp_inst_vir += p_inst_vir
                p_inst = system.analysis.get_instantaneous_pressure()
                p_inst_kin = p_inst - p_inst_vir
                avpV_inst += p_inst_kin * volume
            ls[t] = system.box_l[0]

        avp /= (n / skip_p)
        avpV_sim /= (n / skip_p)
        avp_sim_vir /= (n / skip_p)
        avpV_inst /= (n / skip_p)
        avp_inst_vir /= (n / skip_p)
        Vs = np.array(ls)**3
        compressibility = np.var(Vs) / np.average(Vs)

        self.assertAlmostEqual(avp, p_ext, delta=0.02)
        self.assertAlmostEqual(compressibility, 0.5, delta=0.05)
        np.testing.assert_allclose(avp_sim_vir, avp_inst_vir, atol=1e-10)
        self.assertAlmostEqual(avpV_sim, 100, delta=0.5)
        self.assertAlmostEqual(avpV_inst, 100, delta=0.5)


if __name__ == "__main__":
    ut.main()
