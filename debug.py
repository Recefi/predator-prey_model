import pandas as pd
import numpy as np

import libs.utility as ut
import libs.gen_selection as gs
import libs.param as param


def calcFit(p, q, r, s, i):
    return -s[i]*F-p[i]-q[i]*F+(np.sqrt((4*r[i]*p[i]+(p[i]+q[i]*F-s[i]*F)**2)))

def manualCalcPqrs(Aj, Bj, Aa, Ba):
    M1 = param.sigma1 * (Aj + param.D)
    M2 = -param.sigma2 * (Aj + param.D + Bj/2)
    M3 = -2*(np.pi*Bj)**2
    M4 = -((Aj+param.D0)**2 + (Bj**2)/2)

    M5 = param.sigma1 * (Aa + param.D)
    M6 = -param.sigma2 * (Aa + param.D + Ba/2)
    M7 = -2*(np.pi*Ba)**2
    M8 = -((Aa+param.D0)**2 + (Ba**2)/2)

    p = param.alpha_j*M1 + param.beta_j*M3 + param.delta_j*M4
    r = param.alpha_a*M5 + param.beta_a*M7 + param.delta_a*M8
    q = -param.gamma_j*M2
    s = -param.gamma_a*M6

    return p, q, r, s

def manualFindF(p, q, r, s):
    F, err = gs.calcFLim(p, q, r, s, F0=0.1)
    next = 4*r*p+(p+q*F-s*F)**2 < 0
    if (not next):
        z1, z2 = gs.calcZLim(p, q, r, s, F)
        roots, errs = gs.chkFLim(p, q, r, s, F, z1, z2)
        next = (roots.real > 0).any()
    if next:
        F, err = gs.calcFLim(p, q, r, s, F0=-100)
        next = 4*r*p+(p+q*F-s*F)**2 < 0
        if (not next):
            z1, z2 = gs.calcZLim(p, q, r, s, F)
            roots, errs = gs.chkFLim(p, q, r, s, F, z1, z2)
            next = (roots.real > 0).any()
        if next:
            F, err = gs.calcFLim(p, q, r, s, F0=100)
            next = 4*r*p+(p+q*F-s*F)**2 < 0
            if (not next):
                z1, z2 = gs.calcZLim(p, q, r, s, F)
                roots, errs = gs.chkFLim(p, q, r, s, F, z1, z2)
                next = (roots.real > 0).any()
            if next:
                return []
    return [F, j]

# Aj    -34.600000
# Bj     -3.300582
# Aa    -83.300000
# Ba    -51.566141
# Name: 140901

# Aj     -7.700000
# Bj      7.620953
# Aa    -87.100000
# Ba    -52.869876
# Name: 521

# Aj    -55.800000
# Bj    -11.907889
# Aa    -66.300000
# Ba    -45.733644
# Name: 251395

_Aj, _Bj, _Aa, _Ba = gs.genGenlStrats()
compareParamData = ut.readData("compare_param_data", "dynamic_pred")
a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = compareParamData.loc['restored']
_p, _q, _r, _s = gs.calcGenlPqrsData(_Aj, _Bj, _Aa, _Ba, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
F, j = gs.findF(_p, _q, _r, _s, 521)
fit1 = calcFit(_p, _q, _r, _s, 521)
fit2 = calcFit(_p, _q, _r, _s, 140901)
p, q, r, s = manualCalcPqrs(-34.58, -3.29, -83.32, -51.57)
fit3 = -s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2)))
print(F)
print(fit1, fit2, fit3)


F, j = gs.findF(_p, _q, _r, _s, 140901)
fit1 = calcFit(_p, _q, _r, _s, 521)
fit2 = calcFit(_p, _q, _r, _s, 140901)
p, q, r, s = manualCalcPqrs(-34.58, -3.29, -83.32, -51.57)
fit3 = -s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2)))
print(F)
print(fit1, fit2, fit3)


p, q, r, s = manualCalcPqrs(-34.58, -3.29, -83.32, -51.57)
F, j = manualFindF(p, q, r, s)
fit1 = calcFit(_p, _q, _r, _s, 521)
fit2 = calcFit(_p, _q, _r, _s, 140901)
fit3 = -s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2)))
print(F)
print(fit1, fit2, fit3)
