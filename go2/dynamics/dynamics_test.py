# dynamics_test.py

import unittest
from go2.dynamics.dynamics import *
from go2.kinematics.kinematics import *
import pinocchio as pin

class TestDynamics(unittest.TestCase):
    def test_conputeFullContactForces(self):
        q = getDefaultStandState(model, data)
        pin.framesForwardKinematics(model, data, q)
        qd = np.zeros(model.nv)
        qdd = np.zeros(model.nv)
        Jc = computeFullContactForces(model, data, q, qd, qdd)
        expected = (
                    "Force at FL_EE: -0.115 -1.480 35.496\n"
                    "Force at FR_EE: -0.115 -1.480 35.496\n"
                    "Force at RL_EE: -0.115 -1.480 35.496\n"
                    "Force at RR_EE: -0.115 -1.480 35.496\n"
                    )
        self.assertEqual(printContactForces(Jc), expected)
    



if __name__ == '__main__':
    unittest.main()