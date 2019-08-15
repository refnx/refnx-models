import os.path
import os
import sys

import numpy as np

from numpy.testing import (assert_almost_equal, assert_equal)
sys.path.insert(0, '../')
from md_simulation import MDSimulation


class TestSimulation(object):
    def test_init_a(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile)
        assert_equal(sim.layers.shape, [6, 16, 5])
        assert_equal(sim.av_layers.shape, [16, 5])
        assert_almost_equal(sim.av_layers[:, 0], np.ones((16)))
        assert_almost_equal(sim.av_layers[:, 3], np.zeros((16)))
        assert_almost_equal(sim.av_layers[:, 4], np.zeros((16)))
        assert_almost_equal(sim.layers[:, :, 0], np.ones((6, 16)))
        assert_almost_equal(sim.layers[:, :, 3], np.zeros((6, 16)))
        assert_almost_equal(sim.layers[:, :, 4], np.zeros((6, 16)))
        assert_equal(sim.flip, False)
        assert_almost_equal(sim.cut_off, 5)

    def test_init_b(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, flip=True)
        assert_equal(sim.layers.shape, [6, 16, 5])
        assert_equal(sim.av_layers.shape, [16, 5])
        assert_almost_equal(sim.av_layers[:, 0], np.ones((16)))
        assert_almost_equal(sim.av_layers[:, 3], np.zeros((16)))
        assert_almost_equal(sim.av_layers[:, 4], np.zeros((16)))
        assert_almost_equal(sim.layers[:, :, 0], np.ones((6, 16)))
        assert_almost_equal(sim.layers[:, :, 3], np.zeros((6, 16)))
        assert_almost_equal(sim.layers[:, :, 4], np.zeros((6, 16)))
        assert_equal(sim.flip, True)
        assert_almost_equal(sim.cut_off, 5)

    def test_init_c(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, cut_off=10)
        assert_equal(sim.layers.shape, [6, 16, 5])
        assert_equal(sim.av_layers.shape, [16, 5])
        assert_almost_equal(sim.av_layers[:, 0], np.ones((16)))
        assert_almost_equal(sim.av_layers[:, 3], np.zeros((16)))
        assert_almost_equal(sim.av_layers[:, 4], np.zeros((16)))
        assert_almost_equal(sim.layers[:, :, 0], np.ones((6, 16)))
        assert_almost_equal(sim.layers[:, :, 3], np.zeros((6, 16)))
        assert_almost_equal(sim.layers[:, :, 4], np.zeros((6, 16)))
        assert_almost_equal(sim.cut_off, 10)

    def test_init_d(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, layer_thickness=5)
        assert_equal(sim.layers.shape, [6, 4, 5])
        assert_equal(sim.av_layers.shape, [4, 5])
        assert_almost_equal(sim.av_layers[:, 0], np.ones((4)) * 5)
        assert_almost_equal(sim.av_layers[:, 3], np.zeros((4)))
        assert_almost_equal(sim.av_layers[:, 4], np.zeros((4)))
        assert_almost_equal(sim.layers[:, :, 0], np.ones((6, 4)) * 5)
        assert_almost_equal(sim.layers[:, :, 3], np.zeros((6, 4)))
        assert_almost_equal(sim.layers[:, :, 4], np.zeros((6, 4)))

    def test_init_e(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, roughness=0.2)
        assert_equal(sim.layers.shape, [6, 16, 5])
        assert_equal(sim.av_layers.shape, [16, 5])
        assert_almost_equal(sim.av_layers[:, 0], np.ones((16)))
        assert_almost_equal(sim.av_layers[:, 3], np.ones((16)) * 0.2)
        assert_almost_equal(sim.av_layers[:, 4], np.zeros((16)))
        assert_almost_equal(sim.layers[:, :, 0], np.ones((6, 16)))
        assert_almost_equal(sim.layers[:, :, 3], np.ones((6, 16)) * 0.2)
        assert_almost_equal(sim.layers[:, :, 4], np.zeros((6, 16)))

    def test_read_pdb(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, flip=True)
        a = np.arange(0, 10, 2)
        b = ['C1', 'C2', 'C3', 'N4', 'C5']
        assert_equal(len(sim.u.trajectory), 6)
        assert_equal(sim.u.atoms.names, b)
        for ts in sim.u.trajectory:
            assert_equal(len(sim.u.atoms), 5)
            assert_almost_equal(sim.u.atoms.positions[:, 0], a)
            assert_almost_equal(sim.u.atoms.positions[:, 1], a)
            assert_almost_equal(sim.u.atoms.positions[:, 2], a)
            assert_almost_equal(sim.u.dimensions[0], 10)
            assert_almost_equal(sim.u.dimensions[1], 10)
            assert_almost_equal(sim.u.dimensions[2], 15)

    def test_scat_lens_from_user(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, flip=True)
        a = [[26.659, 0], [26.659, 0], [26.659, 0], [9.36, 0], [19.998, 0]]
        c = ['C1', 'C2', 'C3', 'N4', 'C5']
        sim.assign_scattering_lengths('neutron', atom_types=c,
                                      scattering_lengths=a)
        for i in range(0, len(c)):
            assert_almost_equal(sim.scatlens[c[i]], a[i])

    def test_scat_lens_from_pt(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, flip=True)
        a = [6.6484, 6.6484, 6.6484, 9.36, 6.6484]
        b = np.zeros_like((a))
        c = ['C1', 'C2', 'C3', 'N4', 'C5']
        sim.assign_scattering_lengths('neutron')
        for i in range(0, len(c)):
            assert_almost_equal(sim.scatlens[c[i]], [a[i], b[i]])

    def test_scat_lens_from_pt_xray(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, flip=True)
        a = np.array([16.932084, 16.932084, 16.932084, 19.7685681, 16.932084])
        b = np.array([0.010924737, 0.010924737, 0.010924737, 0.0213729,
                      0.010924737])
        c = ['C1', 'C2', 'C3', 'N4', 'C5']
        sim.assign_scattering_lengths('xray', xray_energy=12)
        for i in range(0, len(c)):
            assert_almost_equal(sim.scatlens[c[i]], [a[i], b[i]])

    def test_set_atom_scattering(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        sim = MDSimulation(pdbfile, flip=True)
        sim.assign_scattering_lengths('neutron')
        a = [6.6484, 1., 6.6484, 9.36, 6.6484]
        b = [0, 1., 0, 0, 0]
        c = ['C1', 'C2', 'C3', 'N4', 'C5']
        sim.set_atom_scattering('C2', [1, 1])
        for i in range(0, len(c)):
            assert_almost_equal(sim.scatlens[c[i]], [a[i], b[i]])

    def test_read_lgt(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        lgtfile = os.path.join(pth, 'mdsim_test.lgt')
        sim = MDSimulation(pdbfile, flip=True)
        sim.assign_scattering_lengths('neutron', lgtfile=lgtfile)
        a = [26.659, 26.659, 26.659, 9.36, 19.998]
        b = np.zeros((5))
        c = ['C1', 'C2', 'C3', 'N4', 'C5']
        for i in range(0, len(c)):
            assert_almost_equal(sim.scatlens[c[i]], [a[i], b[i]])

    def test_get_sld_profile(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        lgtfile = os.path.join(pth, 'mdsim_test.lgt')
        sim = MDSimulation(pdbfile, flip=False)
        sim.assign_scattering_lengths('neutron', lgtfile=lgtfile)
        assert_equal(sim.layers.shape, [6, 16, 5])
        assert_equal(sim.av_layers.shape, [16, 5])
        sim.run()
        a = np.ones(10)
        b = np.array([26.659, 0, 26.659, 0, 26.659, 0, 9.36, 0, 19.998, 0])
        b /= 100
        c = np.zeros(10)
        aa = np.ones(16)
        bb = np.array([26.659, 0, 26.659, 0, 26.659, 0, 9.36, 0, 19.998, 0, 0,
                       0, 0, 0, 0, 0])
        bb /= 100
        cc = np.zeros(16)
        for i in range(0, sim.layers.shape[0]):
            assert_almost_equal(sim.layers[i, :, 0], aa)
            assert_almost_equal(sim.layers[i, :, 1], bb)
            assert_almost_equal(sim.layers[i, :, 2], cc)
            assert_almost_equal(sim.layers[i, :, 3], cc)
            assert_almost_equal(sim.layers[i, :, 4], cc)
        assert_almost_equal(sim.av_layers[:, 0], a)
        assert_almost_equal(sim.av_layers[:, 1], b)
        assert_almost_equal(sim.av_layers[:, 2], c)
        assert_almost_equal(sim.av_layers[:, 3], c)
        assert_almost_equal(sim.av_layers[:, 4], c)
        assert_equal(sim.layers.shape, [6, 16, 5])
        assert_equal(sim.av_layers.shape, [10, 5])
