import numpy as np
from scipy import signal

def vote_msk(sk_flags, sk, m, n):
    indices = np.where(sk < low, True, False)
    sk_flags[indices] = 1
    """sk_range = np.arange(sk.shape[1])
    for ch in np.arange(sk.shape[0]):
        for sk_idx in sk_range:

            ri = ch - n + 1
            rj = sk_idx - m + 1

            if ri < 0:
                ri = 0
            if rj < 0:
                rj = 0

            if (sk[ri:(ch+1), rj:(sk_idx+1)] < low).any():
                sk_flags[ch, sk_idx] = True"""

    return sk_flags

low = 4
sk = np.ones([2,5])*7
sk[0,0] = 0
sk[0,4] = 0
sk[1,2] = 0

print(sk.dtype)
sk_flags = np.zeros([2,5], dtype=np.float16)
sk_flags = vote_msk(sk_flags, sk, 2, 1)
print(sk_flags)
#a = np.asarray([[1,0,0,0,1],[0,0,1,0,0]])
kernel = np.ones([1,2])
sf = signal.convolve2d(sk_flags, kernel, "valid")

print(sf)