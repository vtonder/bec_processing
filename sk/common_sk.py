import numpy as np
import sys
sys.path.append("../")
from constants import num_ch
from kurtosis import spectral_kurtosis_cm

# returns SK lower limit based on given key and M value. keys map to those in constants file
def get_low_limit(low_key, M):
    if low_key == 7:
        from constants import lower_limit7 as l
        low_prefix = "l4sig"
    elif low_key == 0:
        from constants import lower_limit as l
        low_prefix = "l3sig"
    elif low_key == 4:
        from constants import lower_limit4 as l
        low_prefix = "l1pfa"
    else:
        print("LOWER KEY ERROR: Only 0 (3 sigma) and 7 (4 sigma) now supported.")

    return l[M], low_prefix

# returns SK upper limit based on given key and M value. keys map to those in constants file
def get_up_limit(up_key, M):
    if up_key == 7:
        from constants import upper_limit7 as u
        up_prefix = "u4sig"
    elif up_key == 8:
        from constants import upper_limit8 as u
        up_prefix = "uskmax"
    elif up_key == 0:
        from constants import upper_limit as u
        up_prefix = "u3sig"
    elif up_key == 4:
        from constants import upper_limit4 as u
        up_prefix = "u1pfa"
    else:
        print("UPPER KEY ERROR: Only 0 (3 sigma), 7 (4 sigma), 8 (sk max) now supported.")

    return u[M], up_prefix

def rfi_mitigation(data, M, data_window_len, std, check_thres, sk_flags, summed_flags, ndp, chunk_start, first_non_zero_idx, have_sk=None):
    for idx in np.arange(0, data_window_len, M):
        idx_start = int(idx)
        idx_stop = int(idx_start + M)
        sk_idx = int((chunk_start + idx_start - first_non_zero_idx) / M)

        if have_sk:
            sk = have_sk[:, sk_idx]
        else:
            sk = spectral_kurtosis_cm(data[:, idx_start:idx_stop, 0] + 1j * data[:, idx_start:idx_stop, 1], M, 2*num_ch)

        if idx_stop >= ndp:
            print("shortening range because otherwise it will read from memory that doesn't exist")
            print("tot_ndp : ", ndp)
            print("idx_stop: ", idx_stop)
            idx_stop = ndp - 1

        for ch, val in enumerate(sk):
            if check_thres(val):
                sk_flags[ch, sk_idx] = np.uint8(1)
                summed_flags[ch, idx_start:idx_stop] = np.uint8(np.ones(M))
                data[ch, idx_start:idx_stop, :] = np.random.normal(0, std[ch], (M, 2))

    return data, sk_flags, summed_flags

