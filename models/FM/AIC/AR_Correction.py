
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from models.FM.AIC.semilagrangian import extrapolate
from models.FM.AIC.decomposition import filter_gaussian, decomposition_fft, recompose_fft, get_numpy

from models.FM.AIC.utils import nowcast_main_loop
from models.FM.AIC import autoregression


try:
    import dask

    DASK_IMPORTED = True
except ImportError:
    DASK_IMPORTED = False


def forecast(
    vil,
    velocity,
    timesteps,
    rainrate=None,
    n_cascade_levels=6,
    extrap_method="semilagrangian",
    ar_order=2,
    ar_window_radius=50,
    r_vil_window_radius=3,
    fft_method="numpy",
    apply_rainrate_mask=True,
    num_workers=1,
    extrap_kwargs=None,
    filter_kwargs=None,
    measure_time=False,
):
    _check_inputs(vil, rainrate, velocity, timesteps, ar_order)

    if extrap_kwargs is None:
        extrap_kwargs = dict()
    else:
        extrap_kwargs = extrap_kwargs.copy()

    if filter_kwargs is None:
        filter_kwargs = dict()


    if measure_time:
        starttime_init = time.time()

    m, n = vil.shape[1:]
    vil = vil.copy()

    if rainrate is None and apply_rainrate_mask:
        rainrate_mask = vil[-1, :] < 0.1
    else:
        rainrate_mask = None

    if rainrate is not None:
        # determine the coefficients fields of the relation R=a*VIL+b by
        # localized linear regression
        r_vil_a, r_vil_b = _r_vil_regression(vil[-1, :], rainrate, r_vil_window_radius)
    else:
        r_vil_a, r_vil_b = None, None

    # transform the input fields to Lagrangian coordinates by extrapolation
    extrapolator = extrapolate
    extrap_kwargs["allow_nonfinite_values"] = (
        True if np.any(~np.isfinite(vil)) else False
    )

    res = list()

    def worker(vil, i):
        return (
            i,
            extrapolator(
                vil[i, :],
                velocity,
                vil.shape[0] - 1 - i,
                **extrap_kwargs,
            )[-1],
        )

    for i in range(vil.shape[0] - 1):
        if not DASK_IMPORTED or num_workers == 1:
            vil[i, :, :] = worker(vil, i)[1]
        else:
            res.append(dask.delayed(worker)(vil, i))

    if DASK_IMPORTED and num_workers > 1:
        num_workers_ = len(res) if num_workers > len(res) else num_workers
        vil_e = dask.compute(*res, num_workers=num_workers_)
        for i in range(len(vil_e)):
            vil[vil_e[i][0], :] = vil_e[i][1]

    # compute the final mask as the intersection of the masks of the advected
    # fields
    mask = np.isfinite(vil[0, :])
    for i in range(1, vil.shape[0]):
        mask = np.logical_and(mask, np.isfinite(vil[i, :]))

    if rainrate is None and apply_rainrate_mask:
        rainrate_mask = np.logical_and(rainrate_mask, mask)

    # apply cascade decomposition to the advected input fields
    bp_filter_method = filter_gaussian
    bp_filter = bp_filter_method((m, n), n_cascade_levels, **filter_kwargs)

    fft = get_numpy(shape=vil.shape[1:], n_threads=num_workers)

    decomp_method = decomposition_fft
    recomp_method = recompose_fft

    vil_dec = np.empty((n_cascade_levels, vil.shape[0], m, n))
    for i in range(vil.shape[0]):
        vil_ = vil[i, :].copy()
        vil_[~np.isfinite(vil_)] = 0.0
        vil_dec_i = decomp_method(vil_, bp_filter, fft_method=fft)
        for j in range(n_cascade_levels):
            vil_dec[j, i, :] = vil_dec_i["cascade_levels"][j, :]

    # compute time-lagged correlation coefficients for the cascade levels of
    # the advected and differenced input fields
    gamma = np.empty((n_cascade_levels, ar_order, m, n))
    for i in range(n_cascade_levels):
        vil_diff = np.diff(vil_dec[i, :], axis=0)
        vil_diff[~np.isfinite(vil_diff)] = 0.0
        for j in range(ar_order):
            gamma[i, j, :] = _moving_window_corrcoef(
                vil_diff[-1, :], vil_diff[-(j + 2), :], ar_window_radius
            )

    if ar_order == 2:
        # if the order of the ARI model is 2, adjust the correlation coefficients
        # so that the resulting process is stationary
        for i in range(n_cascade_levels):
            gamma[i, 1, :] = autoregression.adjust_lag2_corrcoef2(
                gamma[i, 0, :], gamma[i, 1, :]
            )

    # estimate the parameters of the ARI models
    phi = []
    for i in range(n_cascade_levels):
        if ar_order > 2:
            phi_ = autoregression.estimate_ar_params_yw_localized(gamma[i, :], d=1)
        elif ar_order == 2:
            phi_ = _estimate_ar2_params(gamma[i, :])
        else:
            phi_ = _estimate_ar1_params(gamma[i, :])
        phi.append(phi_)

    vil_dec = vil_dec[:, -(ar_order + 1) :, :]

    if measure_time:
        init_time = time.time() - starttime_init


    rainrate_f = []

    extrap_kwargs["return_displacement"] = True

    state = {"vil_dec": vil_dec}
    params = {
        "apply_rainrate_mask": apply_rainrate_mask,
        "mask": mask,
        "n_cascade_levels": n_cascade_levels,
        "phi": phi,
        "rainrate": rainrate,
        "rainrate_mask": rainrate_mask,
        "recomp_method": recomp_method,
        "r_vil_a": r_vil_a,
        "r_vil_b": r_vil_b,
    }

    rainrate_f = nowcast_main_loop(
        vil[-1, :],
        velocity,
        state,
        timesteps,
        extrap_method,
        _update,
        extrap_kwargs=extrap_kwargs,
        params=params,
        measure_time=measure_time,
    )
    if measure_time:
        rainrate_f, mainloop_time = rainrate_f

    if measure_time:
        return np.stack(rainrate_f), init_time, mainloop_time
    else:
        return np.stack(rainrate_f)


def _check_inputs(vil, rainrate, velocity, timesteps, ar_order):
    if vil.ndim != 3:
        raise ValueError(
            "vil.shape = %s, but a three-dimensional array expected" % str(vil.shape)
        )
    if rainrate is not None:
        if rainrate.ndim != 2:
            raise ValueError(
                "rainrate.shape = %s, but a two-dimensional array expected"
                % str(rainrate.shape)
            )
    if vil.shape[0] != ar_order + 2:
        raise ValueError(
            "vil.shape[0] = %d, but vil.shape[0] = ar_order + 2 = %d required"
            % (vil.shape[0], ar_order + 2)
        )
    if velocity.ndim != 3:
        raise ValueError(
            "velocity.shape = %s, but a three-dimensional array expected"
            % str(velocity.shape)
        )
    if isinstance(timesteps, list) and not sorted(timesteps) == timesteps:
        raise ValueError("timesteps is not in ascending order")


# optimized version of timeseries.autoregression.estimate_ar_params_yw_localized
# for an ARI(1,1) model
def _estimate_ar1_params(gamma):
    phi = []
    phi.append(1 + gamma[0, :])
    phi.append(-gamma[0, :])
    phi.append(np.zeros(gamma[0, :].shape))

    return phi


# optimized version of timeseries.autoregression.estimate_ar_params_yw_localized
# for an ARI(2,1) model
def _estimate_ar2_params(gamma):
    phi_diff = []
    phi_diff.append(gamma[0, :] * (1 - gamma[1, :]) / (1 - gamma[0, :] * gamma[0, :]))
    phi_diff.append(
        (gamma[1, :] - gamma[0, :] * gamma[0, :]) / (1 - gamma[0, :] * gamma[0, :])
    )

    phi = []
    phi.append(1 + phi_diff[0])
    phi.append(-phi_diff[0] + phi_diff[1])
    phi.append(-phi_diff[1])
    phi.append(np.zeros(phi_diff[0].shape))

    return phi


# Compute correlation coefficients of two 2d fields in a moving window with
# a Gaussian weight function. See Section II.G of PCLH2020. Differently to the
# standard formula for the Pearson correlation coefficient, the mean value of
# the inputs is assumed to be zero.
def _moving_window_corrcoef(x, y, window_radius):
    mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x.copy()
    x[~mask] = 0.0
    y = y.copy()
    y[~mask] = 0.0
    mask = mask.astype(float)

    if window_radius is not None:
        n = gaussian_filter(mask, window_radius, mode="constant")

        ssx = gaussian_filter(x**2, window_radius, mode="constant")
        ssy = gaussian_filter(y**2, window_radius, mode="constant")
        sxy = gaussian_filter(x * y, window_radius, mode="constant")
    else:
        n = np.mean(mask)

        ssx = np.mean(x**2)
        ssy = np.mean(y**2)
        sxy = np.mean(x * y)

    stdx = np.sqrt(ssx / n)
    stdy = np.sqrt(ssy / n)
    cov = sxy / n

    mask = np.logical_and(stdx > 1e-8, stdy > 1e-8)
    mask = np.logical_and(mask, stdx * stdy > 1e-8)
    mask = np.logical_and(mask, n > 1e-3)
    corr = np.empty(x.shape)
    corr[mask] = cov[mask] / (stdx[mask] * stdy[mask])
    corr[~mask] = 0.0

    return corr


# Determine the coefficients of the regression R=a*VIL+b.
# See Section II.G of PCLH2020.
# The parameters a and b are estimated in a localized fashion for each pixel
# in the input grid. This is done using a window specified by window_radius.
# Zero and non-finite values are not included. In addition, the regression is
# done by using a Gaussian weight function depending on the distance to the
# current grid point.
def _r_vil_regression(vil, r, window_radius):
    vil = vil.copy()
    vil[~np.isfinite(vil)] = 0.0

    r = r.copy()
    r[~np.isfinite(r)] = 0.0

    mask_vil = vil > 10.0
    mask_r = r > 0.1
    mask_obs = np.logical_and(mask_vil, mask_r)
    vil[~mask_obs] = 0.0
    r[~mask_obs] = 0.0

    n = gaussian_filter(mask_obs.astype(float), window_radius, mode="constant")

    sx = gaussian_filter(vil, window_radius, mode="constant")
    sx2 = gaussian_filter(vil * vil, window_radius, mode="constant")
    sxy = gaussian_filter(vil * r, window_radius, mode="constant")
    sy = gaussian_filter(r, window_radius, mode="constant")

    rhs1 = sxy
    rhs2 = sy

    m1 = sx2
    m2 = sx
    m3 = sx
    m4 = n

    c = 1.0 / (m1 * m4 - m2 * m3)

    m_inv_11 = c * m4
    m_inv_12 = -c * m2
    m_inv_21 = -c * m3
    m_inv_22 = c * m1

    mask = np.abs(m1 * m4 - m2 * m3) > 1e-8
    mask = np.logical_and(mask, n > 0.01)
    a = np.empty(vil.shape)
    a[mask] = m_inv_11[mask] * rhs1[mask] + m_inv_12[mask] * rhs2[mask]
    a[~mask] = 0.0
    a[~mask_vil] = 0.0
    b = np.empty(vil.shape)
    b[mask] = m_inv_21[mask] * rhs1[mask] + m_inv_22[mask] * rhs2[mask]
    b[~mask] = 0.0
    b[~mask_vil] = 0.0

    return a, b


def _update(state, params):
    # iterate the ARI models for each cascade level
    for i in range(params["n_cascade_levels"]):
        state["vil_dec"][i, :] = autoregression.iterate_ar_model(
            state["vil_dec"][i, :], params["phi"][i]
        )

    # recompose the cascade to obtain the forecast field
    vil_dec_dict = {}
    vil_dec_dict["cascade_levels"] = state["vil_dec"][:, -1, :]
    vil_dec_dict["domain"] = "spatial"
    vil_dec_dict["normalized"] = False
    vil_f = params["recomp_method"](vil_dec_dict)
    vil_f[~params["mask"]] = 0.0

    if params["rainrate"] is not None:
        # convert VIL to rain rate
        rainrate_f_new = params["r_vil_a"] * vil_f + params["r_vil_b"]
    else:
        rainrate_f_new = vil_f
        if params["apply_rainrate_mask"]:
            rainrate_f_new[params["rainrate_mask"]] = 0.0

    rainrate_f_new[rainrate_f_new < 0.0] = 0.0

    return rainrate_f_new, state
