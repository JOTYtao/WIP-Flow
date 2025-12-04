

import numpy as np
from scipy.special import binom
from scipy import linalg as la
from scipy import ndimage


def adjust_lag2_corrcoef1(gamma_1, gamma_2):
    gamma_2 = np.maximum(gamma_2, 2 * gamma_1 * gamma_1 - 1 + 1e-10)
    gamma_2 = np.minimum(gamma_2, 1 - 1e-10)

    return gamma_2


def adjust_lag2_corrcoef2(gamma_1, gamma_2):
    gamma_2 = np.maximum(gamma_2, 2 * gamma_1 * gamma_2 - 1)
    gamma_2 = np.maximum(
        gamma_2, (3 * gamma_1**2 - 2 + 2 * (1 - gamma_1**2) ** 1.5) / gamma_1**2
    )

    return gamma_2


def ar_acf(gamma, n=None):
    ar_order = len(gamma)
    if n == ar_order or n is None:
        return gamma
    elif n < ar_order:
        raise ValueError(
            "n=%i, but must be larger than the order of the AR process %i"
            % (n, ar_order)
        )

    phi = estimate_ar_params_yw(gamma)[:-1]

    acf = gamma.copy()
    for t in range(0, n - ar_order):
        # Retrieve gammas (in reverse order)
        gammas = acf[t : t + ar_order][::-1]
        # Compute next gamma
        gamma_ = np.sum(gammas * phi)
        acf.append(gamma_)

    return acf


def estimate_ar_params_ols(
    x, p, d=0, check_stationarity=True, include_constant_term=False, h=0, lam=0.0
):

    n = x.shape[0]

    if n != p + d + h + 1:
        raise ValueError(
            "n = %d, p = %d, d = %d, h = %d, but n = p+d+h+1 = %d required"
            % (n, p, d, h, p + d + h + 1)
        )

    if len(x.shape) > 1:
        x = x.reshape((n, np.prod(x.shape[1:])))

    if d not in [0, 1]:
        raise ValueError("d = %d, but 0 or 1 required" % d)

    if d == 1:
        x = np.diff(x, axis=0)
        n -= d

    x_lhs = x[p:, :]

    Z = []
    for i in range(x.shape[1]):
        for j in range(p - 1, n - 1 - h):
            z_ = np.hstack([x[j - k, i] for k in range(p)])
            if include_constant_term:
                z_ = np.hstack([[1], z_])
            Z.append(z_)
    Z = np.column_stack(Z)

    b = np.dot(
        np.dot(x_lhs, Z.T), np.linalg.inv(np.dot(Z, Z.T) + lam * np.eye(Z.shape[0]))
    )
    b = b.flatten()

    if include_constant_term:
        c = b[0]
        phi = list(b[1:])
    else:
        phi = list(b)

    if p == 1:
        phi_pert = np.sqrt(1.0 - phi[0] * phi[0])
    elif p == 2:
        phi_pert = np.sqrt(
            (1.0 + phi[1]) * ((1.0 - phi[1]) ** 2.0 - phi[0] ** 2.0) / (1.0 - phi[1])
        )
    else:
        phi_pert = 0.0

    if check_stationarity:
        if not test_ar_stationarity(phi):
            raise RuntimeError(
                "Error in estimate_ar_params_yw: " "nonstationary AR(p) process"
            )

    if d == 1:
        phi_out = _compute_differenced_model_params(phi, p, 1, 1)
    else:
        phi_out = phi

    phi_out.append(phi_pert)
    if include_constant_term:
        phi_out.insert(0, c)

    return phi_out


def estimate_ar_params_ols_localized(
    x,
    p,
    window_radius,
    d=0,
    include_constant_term=False,
    h=0,
    lam=0.0,
    window="gaussian",
):

    n = x.shape[0]

    if n != p + d + h + 1:
        raise ValueError(
            "n = %d, p = %d, d = %d, h = %d, but n = p+d+h+1 = %d required"
            % (n, p, d, h, p + d + h + 1)
        )

    if d == 1:
        x = np.diff(x, axis=0)
        n -= d

    if window == "gaussian":
        convol_filter = ndimage.gaussian_filter
    else:
        convol_filter = ndimage.uniform_filter

    if window == "uniform":
        window_size = 2 * window_radius + 1
    else:
        window_size = window_radius

    XZ = np.zeros(np.hstack([[p], x.shape[1:]]))
    for i in range(p):
        for j in range(h + 1):
            tmp = convol_filter(
                x[p + j, :] * x[p - 1 - i + j, :], window_size, mode="constant"
            )
            XZ[i, :] += tmp

    if include_constant_term:
        v = 0.0
        for i in range(h + 1):
            v += convol_filter(x[p + i, :], window_size, mode="constant")
        XZ = np.vstack([v[np.newaxis, :], XZ])

    if not include_constant_term:
        Z2 = np.zeros(np.hstack([[p, p], x.shape[1:]]))
        for i in range(p):
            for j in range(p):
                for k in range(h + 1):
                    tmp = convol_filter(
                        x[p - 1 - i + k, :] * x[p - 1 - j + k, :],
                        window_size,
                        mode="constant",
                    )
                    Z2[i, j, :] += tmp
    else:
        Z2 = np.zeros(np.hstack([[p + 1, p + 1], x.shape[1:]]))
        Z2[0, 0, :] = convol_filter(np.ones(x.shape[1:]), window_size, mode="constant")
        for i in range(p):
            for j in range(h + 1):
                tmp = convol_filter(x[p - 1 - i + j, :], window_size, mode="constant")
                Z2[0, i + 1, :] += tmp
                Z2[i + 1, 0, :] += tmp
        for i in range(p):
            for j in range(p):
                for k in range(h + 1):
                    tmp = convol_filter(
                        x[p - 1 - i + k, :] * x[p - 1 - j + k, :],
                        window_size,
                        mode="constant",
                    )
                    Z2[i + 1, j + 1, :] += tmp

    m = np.prod(x.shape[1:])
    phi = np.empty(np.hstack([[p], m]))
    if include_constant_term:
        c = np.empty(m)
    XZ = XZ.reshape(np.hstack([[XZ.shape[0]], m]))
    Z2 = Z2.reshape(np.hstack([[Z2.shape[0], Z2.shape[1]], m]))

    for i in range(m):
        try:
            b = np.dot(XZ[:, i], np.linalg.inv(Z2[:, :, i] + lam * np.eye(Z2.shape[0])))
            if not include_constant_term:
                phi[:, i] = b
            else:
                phi[:, i] = b[1:]
                c[i] = b[0]
        except np.linalg.LinAlgError:
            phi[:, i] = np.nan
            if include_constant_term:
                c[i] = np.nan

    if p == 1:
        phi_pert = np.sqrt(1.0 - phi[0, :] * phi[0, :])
    elif p == 2:
        phi_pert = np.sqrt(
            (1.0 + phi[1, :])
            * ((1.0 - phi[1, :]) ** 2.0 - phi[0, :] ** 2.0)
            / (1.0 - phi[1, :])
        )
    else:
        phi_pert = np.zeros(m)

    phi = list(phi.reshape(np.hstack([[phi.shape[0]], x.shape[1:]])))
    if d == 1:
        phi = _compute_differenced_model_params(phi, p, 1, 1)
    phi.append(phi_pert.reshape(x.shape[1:]))
    if include_constant_term:
        phi.insert(0, c.reshape(x.shape[1:]))

    return phi


def estimate_ar_params_yw(gamma, d=0, check_stationarity=True):
    if d not in [0, 1]:
        raise ValueError("d = %d, but 0 or 1 required" % d)

    p = len(gamma)

    g = np.hstack([[1.0], gamma])
    G = []
    for j in range(p):
        G.append(np.roll(g[:-1], j))
    G = np.array(G)
    phi = np.linalg.solve(G, g[1:].flatten())

    # Check that the absolute values of the roots of the characteristic
    # polynomial are less than one.
    # Otherwise the AR(p) model is not stationary.
    if check_stationarity:
        if not test_ar_stationarity(phi):
            raise RuntimeError(
                "Error in estimate_ar_params_yw: " "nonstationary AR(p) process"
            )

    c = 1.0
    for j in range(p):
        c -= gamma[j] * phi[j]
    phi_pert = np.sqrt(c)

    # If the expression inside the square root is negative, phi_pert cannot
    # be computed and it is set to zero instead.
    if not np.isfinite(phi_pert):
        phi_pert = 0.0

    if d == 1:
        phi = _compute_differenced_model_params(phi, p, 1, 1)

    phi_out = np.empty(len(phi) + 1)
    phi_out[: len(phi)] = phi
    phi_out[-1] = phi_pert

    return phi_out


def estimate_ar_params_yw_localized(gamma, d=0):

    for i in range(1, len(gamma)):
        if gamma[i].shape != gamma[0].shape:
            raise ValueError(
                "the correlation coefficient fields gamma have mismatching shapes"
            )

    if d not in [0, 1]:
        raise ValueError("d = %d, but 0 or 1 required" % d)

    p = len(gamma)
    n = np.prod(gamma[0].shape)

    gamma_1d = [gamma[i].flatten() for i in range(len(gamma))]

    phi = np.empty((p, n))
    for i in range(n):
        g = np.hstack([[1.0], [gamma_1d[k][i] for k in range(len(gamma_1d))]])
        G = []
        for k in range(p):
            G.append(np.roll(g[:-1], k))
        G = np.array(G)
        try:
            phi_ = np.linalg.solve(G, g[1:].flatten())
        except np.linalg.LinAlgError:
            phi_ = np.ones(p) * np.nan

        phi[:, i] = phi_

    c = 1.0
    for i in range(p):
        c -= gamma_1d[i] * phi[i]
    phi_pert = np.sqrt(c)

    if d == 1:
        phi = _compute_differenced_model_params(phi, p, 1, 1)

    phi_out = np.empty((len(phi) + 1, n))
    phi_out[: len(phi), :] = phi
    phi_out[-1, :] = phi_pert

    return list(phi_out.reshape(np.hstack([[len(phi_out)], gamma[0].shape])))


def estimate_var_params_ols(
    x, p, d=0, check_stationarity=True, include_constant_term=False, h=0, lam=0.0
):
    q = x.shape[1]
    n = x.shape[0]

    if n != p + d + h + 1:
        raise ValueError(
            "n = %d, p = %d, d = %d, h = %d, but n = p+d+h+1 = %d required"
            % (n, p, d, h, p + d + h + 1)
        )

    if d not in [0, 1]:
        raise ValueError("d = %d, but 0 or 1 required" % d)

    if d == 1:
        x = np.diff(x, axis=0)
        n -= d

    x = x.reshape((n, q, np.prod(x.shape[2:])))

    X = []
    for i in range(x.shape[2]):
        for j in range(p + h, n):
            x_ = x[j, :, i]
            X.append(x_.reshape((q, 1)))
    X = np.hstack(X)

    Z = []
    for i in range(x.shape[2]):
        for j in range(p - 1, n - 1 - h):
            z_ = np.vstack([x[j - k, :, i].reshape((q, 1)) for k in range(p)])
            if include_constant_term:
                z_ = np.vstack([[1], z_])
            Z.append(z_)
    Z = np.column_stack(Z)

    B = np.dot(np.dot(X, Z.T), np.linalg.inv(np.dot(Z, Z.T) + lam * np.eye(Z.shape[0])))

    phi = []
    if include_constant_term:
        c = B[:, 0]
        for i in range(p):
            phi.append(B[:, i * q + 1 : (i + 1) * q + 1])
    else:
        for i in range(p):
            phi.append(B[:, i * q : (i + 1) * q])

    if check_stationarity:
        M = np.zeros((p * q, p * q))

        for i in range(p):
            M[0:q, i * q : (i + 1) * q] = phi[i]
        for i in range(1, p):
            M[i * q : (i + 1) * q, (i - 1) * q : i * q] = np.eye(q, q)
        r, v = np.linalg.eig(M)

        if np.any(np.abs(r) > 0.999):
            raise RuntimeError(
                "Error in estimate_var_params_ols: " "nonstationary VAR(p) process"
            )

    if d == 1:
        phi = _compute_differenced_model_params(phi, p, q, 1)

    if include_constant_term:
        phi.insert(0, c)
    phi.append(np.zeros((q, q)))

    return phi


def estimate_var_params_ols_localized(
    x,
    p,
    window_radius,
    d=0,
    include_constant_term=False,
    h=0,
    lam=0.0,
    window="gaussian",
):
    q = x.shape[1]
    n = x.shape[0]

    if n != p + d + h + 1:
        raise ValueError(
            "n = %d, p = %d, d = %d, h = %d, but n = p+d+h+1 = %d required"
            % (n, p, d, h, p + d + h + 1)
        )

    if d == 1:
        x = np.diff(x, axis=0)
        n -= d

    if window == "gaussian":
        convol_filter = ndimage.gaussian_filter
    else:
        convol_filter = ndimage.uniform_filter

    if window == "uniform":
        window_size = 2 * window_radius + 1
    else:
        window_size = window_radius

    XZ = np.zeros(np.hstack([[q, p * q], x.shape[2:]]))
    for i in range(q):
        for k in range(p):
            for j in range(q):
                for l in range(h + 1):
                    tmp = convol_filter(
                        x[p + l, i, :] * x[p - 1 - k + l, j, :],
                        window_size,
                        mode="constant",
                    )
                    XZ[i, k * q + j, :] += tmp

    if include_constant_term:
        v = np.zeros(np.hstack([[q], x.shape[2:]]))
        for i in range(q):
            for j in range(h + 1):
                v[i, :] += convol_filter(x[p + j, i, :], window_size, mode="constant")
        XZ = np.hstack([v[:, np.newaxis, :], XZ])

    if not include_constant_term:
        Z2 = np.zeros(np.hstack([[p * q, p * q], x.shape[2:]]))
        for i in range(p):
            for j in range(q):
                for k in range(p):
                    for l in range(q):
                        for m in range(h + 1):
                            tmp = convol_filter(
                                x[p - 1 - i + m, j, :] * x[p - 1 - k + m, l, :],
                                window_size,
                                mode="constant",
                            )
                            Z2[i * q + j, k * q + l, :] += tmp
    else:
        Z2 = np.zeros(np.hstack([[p * q + 1, p * q + 1], x.shape[2:]]))
        Z2[0, 0, :] = convol_filter(np.ones(x.shape[2:]), window_size, mode="constant")
        for i in range(p):
            for j in range(q):
                for k in range(h + 1):
                    tmp = convol_filter(
                        x[p - 1 - i + k, j, :], window_size, mode="constant"
                    )
                    Z2[0, i * q + j + 1, :] += tmp
                    Z2[i * q + j + 1, 0, :] += tmp
        for i in range(p):
            for j in range(q):
                for k in range(p):
                    for l in range(q):
                        for m in range(h + 1):
                            tmp = convol_filter(
                                x[p - 1 - i + m, j, :] * x[p - 1 - k + m, l, :],
                                window_size,
                                mode="constant",
                            )
                            Z2[i * q + j + 1, k * q + l + 1, :] += tmp

    m = np.prod(x.shape[2:])
    if include_constant_term:
        c = np.empty((m, q))
    XZ = XZ.reshape((XZ.shape[0], XZ.shape[1], m))
    Z2 = Z2.reshape((Z2.shape[0], Z2.shape[1], m))

    phi = np.empty((p, m, q, q))
    for i in range(m):
        try:
            B = np.dot(
                XZ[:, :, i], np.linalg.inv(Z2[:, :, i] + lam * np.eye(Z2.shape[0]))
            )
            for k in range(p):
                if not include_constant_term:
                    phi[k, i, :, :] = B[:, k * q : (k + 1) * q]
                else:
                    phi[k, i, :, :] = B[:, k * q + 1 : (k + 1) * q + 1]
            if include_constant_term:
                c[i, :] = B[:, 0]
        except np.linalg.LinAlgError:
            phi[:, i, :, :] = np.nan
            if include_constant_term:
                c[i, :] = np.nan

    phi_out = [
        phi[i].reshape(np.hstack([x.shape[2:], [q, q]])) for i in range(len(phi))
    ]
    if d == 1:
        phi_out = _compute_differenced_model_params(phi_out, p, q, 1)

    phi_out.append(np.zeros(phi_out[0].shape))
    if include_constant_term:
        phi_out.insert(0, c.reshape(np.hstack([x.shape[2:], [q]])))

    return phi_out


def estimate_var_params_yw(gamma, d=0, check_stationarity=True):
    p = len(gamma) - 1
    q = gamma[0].shape[0]

    for i in range(len(gamma)):
        if gamma[i].shape[0] != q or gamma[i].shape[1] != q:
            raise ValueError(
                "dimension mismatch: gamma[%d].shape=%s, but (%d,%d) expected"
                % (i, str(gamma[i].shape), q, q)
            )

    if d not in [0, 1]:
        raise ValueError("d = %d, but 0 or 1 required" % d)

    a = np.empty((p * q, p * q))
    for i in range(p):
        for j in range(p):
            a_tmp = gamma[abs(i - j)]
            if i > j:
                a_tmp = a_tmp.T
            a[i * q : (i + 1) * q, j * q : (j + 1) * q] = a_tmp

    b = np.vstack([gamma[i].T for i in range(1, p + 1)])
    x = np.linalg.solve(a, b)

    phi = []
    for i in range(p):
        phi.append(x[i * q : (i + 1) * q, :])

    if check_stationarity:
        if not test_var_stationarity(phi):
            raise RuntimeError(
                "Error in estimate_var_params_yw: " "nonstationary VAR(p) process"
            )

    if d == 1:
        phi = _compute_differenced_model_params(phi, p, q, 1)

    phi.append(np.zeros(phi[0].shape))

    return phi


def estimate_var_params_yw_localized(gamma, d=0):
    p = len(gamma) - 1
    q = gamma[0].shape[2]
    n = np.prod(gamma[0].shape[:-2])

    for i in range(1, len(gamma)):
        if gamma[i].shape != gamma[0].shape:
            raise ValueError(
                "dimension mismatch: gamma[%d].shape=%s, but %s expected"
                % (i, str(gamma[i].shape), str(gamma[0].shape))
            )

    if d not in [0, 1]:
        raise ValueError("d = %d, but 0 or 1 required" % d)

    gamma_1d = [g.reshape((n, q, q)) for g in gamma]
    phi_out = [np.zeros([n, q, q]) for i in range(p)]

    for k in range(n):
        a = np.empty((p * q, p * q))
        for i in range(p):
            for j in range(p):
                a_tmp = gamma_1d[abs(i - j)][k, :]
                if i > j:
                    a_tmp = a_tmp.T
                a[i * q : (i + 1) * q, j * q : (j + 1) * q] = a_tmp

        b = np.vstack([gamma_1d[i][k, :].T for i in range(1, p + 1)])
        x = np.linalg.solve(a, b)

        for i in range(p):
            phi_out[i][k, :, :] = x[i * q : (i + 1) * q, :]

    for i in range(len(phi_out)):
        phi_out[i] = phi_out[i].reshape(np.hstack([gamma[0].shape[:-2], [q, q]]))
    if d == 1:
        phi_out = _compute_differenced_model_params(phi_out, p, 1, 1)
    phi_out.append(np.zeros(gamma[0].shape))

    return phi_out


def iterate_ar_model(x, phi, eps=None):

    if x.shape[0] < len(phi) - 1:
        raise ValueError(
            "dimension mismatch between x and phi: x.shape[0]=%d, len(phi)=%d"
            % (x.shape[0], len(phi))
        )

    if len(x.shape) == 1:
        x_simple_shape = True
        x = x[:, np.newaxis]
    else:
        x_simple_shape = False

    if eps is not None and eps.shape != x.shape[1:]:
        raise ValueError(
            "dimension mismatch between x and eps: x[1:].shape=%s, eps.shape=%s"
            % (str(x[1:].shape), str(eps.shape))
        )

    x_new = 0.0

    p = len(phi) - 1

    for i in range(p):
        x_new += phi[i] * x[-(i + 1), :]

    if eps is not None:
        x_new += phi[-1] * eps

    if x_simple_shape:
        return np.hstack([x[1:], [x_new]])
    else:
        return np.concatenate([x[1:, :], x_new[np.newaxis, :]])


def iterate_var_model(x, phi, eps=None):

    if x.shape[0] < len(phi) - 1:
        raise ValueError(
            "dimension mismatch between x and phi: x.shape[0]=%d, len(phi)=%d"
            % (x.shape[1], len(phi))
        )

    phi_shape = phi[0].shape
    if phi_shape[-1] != phi_shape[-2]:
        raise ValueError(
            "phi[0].shape = %s, but the last two dimensions are expected to be equal"
            % str(phi_shape)
        )
    for i in range(1, len(phi)):
        if phi[i].shape != phi_shape:
            raise ValueError("dimension mismatch between parameter matrices phi")

    if len(x.shape) == 2:
        x_simple_shape = True
        x = x[:, :, np.newaxis]
    else:
        x_simple_shape = False

    x_new = np.zeros(x.shape[1:])
    p = len(phi) - 1

    for l in range(p):
        x_new += np.einsum("...ij,j...->i...", phi[l], x[-(l + 1), :])

    if eps is not None:
        x_new += np.dot(np.dot(phi[-1], phi[-1]), eps)

    if x_simple_shape:
        return np.vstack([x[1:, :, 0], x_new[:, 0]])
    else:
        x_new = x_new.reshape(x.shape[1:])
        return np.concatenate([x[1:, :], x_new[np.newaxis, :, :]], axis=0)


def test_ar_stationarity(phi):

    r = np.array(
        [
            np.abs(r_)
            for r_ in np.roots([1.0 if i == 0 else -phi[i] for i in range(len(phi))])
        ]
    )

    return False if np.any(r >= 1) else True


def test_var_stationarity(phi):

    q = phi[0].shape
    for i in range(1, len(phi)):
        if phi[i].shape != q:
            raise ValueError("dimension mismatch between parameter matrices phi")

    p = len(phi)
    q = phi[0].shape[0]

    M = np.zeros((p * q, p * q))

    for i in range(p):
        M[0:q, i * q : (i + 1) * q] = phi[i]
    for i in range(1, p):
        M[i * q : (i + 1) * q, (i - 1) * q : i * q] = np.eye(q, q)
    r = np.linalg.eig(M)[0]

    return False if np.any(np.abs(r) >= 1) else True


def _compute_differenced_model_params(phi, p, q, d):
    phi_out = []
    for i in range(p + d):
        if q > 1:
            if len(phi[0].shape) == 2:
                phi_out.append(np.zeros((q, q)))
            else:
                phi_out.append(np.zeros(phi[0].shape))
        else:
            phi_out.append(0.0)

    for i in range(1, d + 1):
        if q > 1:
            phi_out[i - 1] -= binom(d, i) * (-1) ** i * np.eye(q)
        else:
            phi_out[i - 1] -= binom(d, i) * (-1) ** i
    for i in range(1, p + 1):
        phi_out[i - 1] += phi[i - 1]
    for i in range(1, p + 1):
        for j in range(1, d + 1):
            phi_out[i + j - 1] += phi[i - 1] * binom(d, j) * (-1) ** j

    return phi_out