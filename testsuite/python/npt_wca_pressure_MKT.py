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
import espressomd
import espressomd.propagation
import tests_common
import numpy as np


@utx.skipIfMissingFeatures("NPT")
class NPTPressure(ut.TestCase):

    """Test NpT dynamics"""
    system = espressomd.System(box_l=[1.0, 1.0, 1.0])
    system.cell_system.skin = 0.
    system.periodicity = [True, True, True]

    def setUp(self):
        np.random.seed(42)

    def tearDown(self):
        self.system.time_step = 0.01
        self.system.cell_system.skin = 0.0
        self.system.part.clear()
        self.system.thermostat.turn_off()
        self.system.integrator.set_vv()

    @utx.skipIfMissingFeatures("WCA")
    def test_pressure_compared_with_instantaneous(self):
        """Test for Npt."""

        data = np.genfromtxt(tests_common.data_path("npt_lj_system.data"))
        #ref_box_l = 1.01 * np.max(data[:, 0:3])
        ref_box_l = np.max(data[:, 0:3])

        system = self.system
        system.box_l = 3 * [ref_box_l]
        system.non_bonded_inter[2, 2].wca.set_params(epsilon=1., sigma=1.)
        dt = 0.005
        system.time_step = dt

        direction = [True] * 3
        ext_pressure = 1.0
        system.box_l = 3 * [ref_box_l]
        system.part.add(pos=data[:, 0:3], type=len(data) * [2])
        system.part.all().pos = data[:, 0:3]
        system.part.all().v = data[:, 3:6]
        self.system.integrator.set_vv()
        #system.integrator.run(100)

        system.thermostat.set_npt(kT=1.0, gamma0=0.5, gammav=0.001, seed=42)
        #system.thermostat.set_npt(kT=1.0, gamma0=0., gammav=0., seed=42)
        system.integrator.set_isotropic_npt(ext_pressure=ext_pressure,
                                            piston=4.0,
                                            direction=direction)
        verbose = 1
        delta_p = list()
        volume = list()
        p_kinV = list()
        H_MKT = list()
        steps = int(0.1/dt)
        for n in range(2000):
            system.integrator.run(steps)
            p_sim = system.analysis.pressure()['total']
            p_kin = system.analysis.pressure()['kinetic']
            p_vir = p_sim - p_kin
            p_inst = system.analysis.get_instantaneous_pressure()
            #p_inst_tpdt = system.analysis.get_instantaneous_pressure_tpdt()
            p_inst_vir = system.analysis.get_instantaneous_pressure_virial()
            p_inst_kin1 = p_inst - p_inst_vir
            energy = system.analysis.energy()['total']
            kin_v = system.analysis.get_kinetic_energy_for_volume()
            H_MKT.append(energy + kin_v + p_kin*np.prod(system.box_l))
            #p_inst_kin2 = p_inst_tpdt - p_inst_vir
            if verbose:
                print(n)
                print('total', p_sim, p_inst, system.box_l)
                print('kin', p_kin, p_inst_kin1)
                print('vir', p_vir, p_inst_vir)
                print('P_{kin}V', p_kin*np.prod(system.box_l))
                #print('H_{MKT}', energy + kin_v + p_kin*np.prod(system.box_l))
            #np.testing.assert_allclose(p_kin, p_inst_kin1, atol=1e-10)
            np.testing.assert_allclose(p_vir, p_inst_vir, atol=1e-10)
            delta_p.append(p_sim - ext_pressure)
            volume.append(system.box_l[0]*system.box_l[1]*system.box_l[2])
            p_kinV.append(p_kin*np.prod(system.box_l))
        verbose_acf = 1
        if verbose_acf:
            acf_p = np.correlate(delta_p, delta_p, mode='same') 
            for n in range(len(acf_p)):
                print(n*0.1, acf_p[n], delta_p[n])
        print("#dP AVE", np.mean(np.array(delta_p)))
        print("#dP STD", np.std(np.array(delta_p)))
        print("#V AVE", np.mean(np.array(volume)))
        print("#V STD", np.std(np.array(volume)))
        print("#PV AVE", np.mean(np.array(p_kinV)))
        print("#PV STD", np.std(np.array(p_kinV)))
        #print("#H_MKT AVE", np.mean(np.array(H_MKT)))
        #print("#H_MKT STD", np.std(np.array(H_MKT)))

    def test_integrator_exceptions(self):
        system = self.system

        # invalid parameters should throw exceptions
        with self.assertRaises(Exception):
            system.integrator.set_isotropic_npt(ext_pressure=-1., piston=1.)
        with self.assertRaises(Exception):
            system.integrator.set_isotropic_npt(ext_pressure=1., piston=-1.)
        with self.assertRaises(Exception):
            system.integrator.set_isotropic_npt(ext_pressure=1., piston=0.)
        with self.assertRaises(Exception):
            system.integrator.set_isotropic_npt(ext_pressure=1., piston=1.,
                                                direction=[0, 0, 0])


if __name__ == "__main__":
    ut.main()
