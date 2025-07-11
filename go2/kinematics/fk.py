# fk.py

from go2.robot.robot import *
from go2.robot.morphology import *
from go2.utils.math_utils import *

def fk(q):
    if isinstance(q, (ca.SX, ca.MX)):
        cs_x = ca.SX.sym("x", 15, 1)
        cs_q = ca.SX.sym("q", NUM_Q, 1)
        pinocchio.casadi.forwardKinematics(ad_model, ad_data, cs_q)
        bodyPos = data.oMf[BODY_FRAME].translation        
        eePos = []
        for frameId in EE_FRAME_IDS:
            ee_pos = data.oMf[frameId].translation
            eePos.append(ee_pos)
        cs_x = ca.vertcat(
            bodyPos,
            *eePos
        )
        cs_x_fn = ca.Function("x_fn", [cs_q], [cs_x])
        x = cs_x_fn(q)
    else:
        x = []
        pin.forwardKinematics(model, data, q)
        bodyPosition = data.oMf[Frame.BASE_LINK].translation
        x[:3] = bodyPosition
        for frameId in EE_FRAME_IDS:
            EEPosition = data.oMf[frameId].translation
            x = np.hstack((x, EEPosition))
    return x


def printFk(q):
    x = fk(q)
    basePos = x[:3].copy()
    basePos[np.abs(basePos) < 1e-6] = 0.0
    print("{}: {: .3f} {: .3f} {: .3f}".format(BODY_NAME, *basePos))
    for i, frameName in enumerate(EE_FRAME_NAMES):
        start = 3 + 3*i
        end = start + 3
        eePos = x[start:end].copy()
        eePos[np.abs(eePos) < 1e-6] = 0.0
        print("{}: {: .3f} {: .3f} {: .3f}".format(frameName, *eePos))