import numpy as np

def compute_centred_coord_array(M, N):
    if M % 2 == 1:
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC
def corrcoef(X, Y, shape, use_full_fft=False):
    if len(X.shape) != 2:
        raise ValueError("X is not a two-dimensional array")

    if len(Y.shape) != 2:
        raise ValueError("Y is not a two-dimensional array")

    if X.shape != Y.shape:
        raise ValueError(
            "dimension mismatch between X and Y: "
            + "X.shape=%d,%d , " % (X.shape[0], X.shape[1])
            + "Y.shape=%d,%d" % (Y.shape[0], Y.shape[1])
        )

    n = np.real(np.sum(X * np.conj(Y))) - np.real(X[0, 0] * Y[0, 0])
    d1 = np.sum(np.abs(X) ** 2) - np.real(X[0, 0]) ** 2
    d2 = np.sum(np.abs(Y) ** 2) - np.real(Y[0, 0]) ** 2

    if not use_full_fft:
        if shape[1] % 2 == 1:
            n += np.real(np.sum(X[:, 1:] * np.conj(Y[:, 1:])))
            d1 += np.sum(np.abs(X[:, 1:]) ** 2)
            d2 += np.sum(np.abs(Y[:, 1:]) ** 2)
        else:
            n += np.real(np.sum(X[:, 1:-1] * np.conj(Y[:, 1:-1])))
            d1 += np.sum(np.abs(X[:, 1:-1]) ** 2)
            d2 += np.sum(np.abs(Y[:, 1:-1]) ** 2)

    return n / np.sqrt(d1 * d2)


def mean(X, shape):
    return np.real(X[0, 0]) / (shape[0] * shape[1])


def rapsd(
    field, fft_method=None, return_freq=False, d=1.0, normalize=False, **fft_kwargs
):

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field, **fft_kwargs))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result



def std(X, shape, use_full_fft=False):
    res = np.sum(np.abs(X) ** 2) - np.real(X[0, 0]) ** 2
    if not use_full_fft:
        if shape[1] % 2 == 1:
            res += np.sum(np.abs(X[:, 1:]) ** 2)
        else:
            res += np.sum(np.abs(X[:, 1:-1]) ** 2)

    return np.sqrt(res / (shape[0] * shape[1]) ** 2)