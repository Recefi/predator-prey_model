import pandas as pd
import numpy as np

import libs.utility as ut
import libs.gen_selection as gs
import libs.param as param


def calcFit(p, q, r, s, i):
    return -s[i]*F-p[i]-q[i]*F+(np.sqrt((4*r[i]*p[i]+(p[i]+q[i]*F-s[i]*F)**2)))

def manualCalcPqrs(Aj, Bj, Aa, Ba, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a):
    M1 = param.sigma1 * (Aj + param.D)
    M2 = -param.sigma2 * (Aj + param.D + Bj/2)
    M3 = -2*(np.pi*Bj)**2
    M4 = -((Aj+param.D0)**2 + (Bj**2)/2)

    M5 = param.sigma1 * (Aa + param.D)
    M6 = -param.sigma2 * (Aa + param.D + Ba/2)
    M7 = -2*(np.pi*Ba)**2
    M8 = -((Aa+param.D0)**2 + (Ba**2)/2)

    p = a_j*M1 + b_j*M3 + d_j*M4
    r = a_a*M5 + b_a*M7 + d_a*M8
    q = -g_j*M2
    s = -g_a*M6

    return p, q, r, s

def manualFindF(p, q, r, s, j):
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

compareParamData = ut.readData("compare_param_data", "dynamic_pred")
a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = compareParamData.loc['restored']
_Aj, _Bj, _Aa, _Ba = gs.genGenlStrats(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
_p, _q, _r, _s = gs.calcGenlPqrsData(_Aj, _Bj, _Aa, _Ba, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)

p, q, r, s = manualCalcPqrs(-22.0, 20.456154, -104.200000, -30.500824, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
F, j = manualFindF(p, q, r, s, 0)
fit1 = -s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2)))
p, q, r, s = manualCalcPqrs(-22.0, -20.456154, -104.200000, -30.500824, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
fit2 = -s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2)))
print(F)
print(fit1, fit2)

p, q, r, s = manualCalcPqrs(-22.0, -20.456154, -104.200000, -30.500824, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
F, j = manualFindF(p, q, r, s, 0)
fit2 = -s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2)))
p, q, r, s = manualCalcPqrs(-22.0, 20.456154, -104.200000, -30.500824, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
fit1 = -s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2)))
print(F)
print(fit1, fit2)

#3
# p, q, r, s = manualCalcPqrs(-7.700000, -7.620953, -87.100000, -52.869876, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
# Fsols = gs.findFsols(p, q, r, s)
# print(Fsols)
# FLams, errs = gs.chkFsols(p, q, r, s, Fsols)
# print(FLams)
# print(errs)
