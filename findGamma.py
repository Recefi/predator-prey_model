import numpy as np

import libs.param as param
import libs.gen_selection as gs


Aj, Bj, Aa, Ba = (-35, -3.93, -83, -49.2)

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

FLim = 1
_FLim = gs.calcFLim(p, q, r, s, F0=0.1)
print("FLim:", _FLim)

while(abs(FLim - _FLim) > 0.00001):
    gamma_j = param.gamma_j / _FLim
    gamma_a = param.gamma_a / _FLim
    print("gamma_j:", gamma_j)
    print("gamma_a:", gamma_a)

    q = -gamma_j*M2
    s = -gamma_a*M6

    FLim = _FLim
    _FLim = gs.calcFLim(p, q, r, s, F0=0.1)
    print("_FLim:", _FLim)
