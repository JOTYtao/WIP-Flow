import numpy as np


import numpy as np
import numpy.fft as numpy_fft
from types import SimpleNamespace
from models.FM.AR_OF import spectral
def filter_uniform(shape, n):
    """
    A dummy filter with one frequency band covering the whole domain. The
    weights are set to one.

    Parameters
    ----------
    shape: int or tuple
        The dimensions (height, width) of the input field. If shape is an int,
        the domain is assumed to have square shape.
    n: int
        Not used. Needed for compatibility with the filter interface.

    Returns
    -------
    out: dict
        A dictionary containing the filter.
    """
    del n  # Unused

    out = {}

    try:
        height, width = shape
    except TypeError:
        height, width = (shape, shape)

    r_max = int(max(width, height) / 2) + 1

    out["weights_1d"] = np.ones((1, r_max))
    out["weights_2d"] = np.ones((1, height, int(width / 2) + 1))
    out["central_freqs"] = None
    out["central_wavenumbers"] = None
    out["shape"] = shape

    return out


def filter_gaussian(
    shape,
    n,
    gauss_scale=0.5,
    d=1.0,
    normalize=True,
    return_weight_funcs=False,
    include_mean=True,
):
    if n < 3:
        raise ValueError("n must be greater than 2")

    try:
        height, width = shape
    except TypeError:
        height, width = (shape, shape)

    max_length = max(width, height)

    rx = np.s_[: int(width / 2) + 1]

    if (height % 2) == 1:
        ry = np.s_[-int(height / 2) : int(height / 2) + 1]
    else:
        ry = np.s_[-int(height / 2) : int(height / 2)]

    y_grid, x_grid = np.ogrid[ry, rx]
    dy = int(height / 2) if height % 2 == 0 else int(height / 2) + 1

    r_2d = np.roll(np.sqrt(x_grid * x_grid + y_grid * y_grid), dy, axis=0)

    r_max = int(max_length / 2) + 1
    r_1d = np.arange(r_max)

    wfs, central_wavenumbers = _gaussweights_1d(
        max_length,
        n,
        gauss_scale=gauss_scale,
    )

    weights_1d = np.empty((n, r_max))
    weights_2d = np.empty((n, height, int(width / 2) + 1))

    for i, wf in enumerate(wfs):
        weights_1d[i, :] = wf(r_1d)
        weights_2d[i, :, :] = wf(r_2d)

    if normalize:
        weights_1d_sum = np.sum(weights_1d, axis=0)
        weights_2d_sum = np.sum(weights_2d, axis=0)
        for k in range(weights_2d.shape[0]):
            weights_1d[k, :] /= weights_1d_sum
            weights_2d[k, :, :] /= weights_2d_sum

    for i in range(len(wfs)):
        if i == 0 and include_mean:
            weights_1d[i, 0] = 1.0
            weights_2d[i, 0, 0] = 1.0
        else:
            weights_1d[i, 0] = 0.0
            weights_2d[i, 0, 0] = 0.0

    out = {"weights_1d": weights_1d, "weights_2d": weights_2d}
    out["shape"] = shape

    central_wavenumbers = np.array(central_wavenumbers)
    out["central_wavenumbers"] = central_wavenumbers

    # Compute frequencies
    central_freqs = 1.0 * central_wavenumbers / max_length
    central_freqs[0] = 1.0 / max_length
    central_freqs[-1] = 0.5  # Nyquist freq
    central_freqs = 1.0 * d * central_freqs
    out["central_freqs"] = central_freqs

    if return_weight_funcs:
        out["weight_funcs"] = wfs

    return out


def _gaussweights_1d(l, n, gauss_scale=0.5):
    q = pow(0.5 * l, 1.0 / n)
    r = [(pow(q, k - 1), pow(q, k)) for k in range(1, n + 1)]
    r = [0.5 * (r_[0] + r_[1]) for r_ in r]

    def log_e(x):
        if len(np.shape(x)) > 0:
            res = np.empty(x.shape)
            res[x == 0] = 0.0
            res[x > 0] = np.log(x[x > 0]) / np.log(q)
        else:
            if x == 0.0:
                res = 0.0
            else:
                res = np.log(x) / np.log(q)

        return res

    class GaussFunc:
        def __init__(self, c, s):
            self.c = c
            self.s = s

        def __call__(self, x):
            x = log_e(x) - self.c
            return np.exp(-(x**2.0) / (2.0 * self.s**2.0))

    weight_funcs = []
    central_wavenumbers = []

    for i, ri in enumerate(r):
        rc = log_e(ri)
        weight_funcs.append(GaussFunc(rc, gauss_scale))
        central_wavenumbers.append(ri)

    return weight_funcs, central_wavenumbers

def get_numpy(shape, fftn_shape=None, **kwargs):

    f = {
        "fft2": numpy_fft.fft2,
        "ifft2": numpy_fft.ifft2,
        "rfft2": numpy_fft.rfft2,
        "irfft2": lambda X: numpy_fft.irfft2(X, s=shape),
        "fftshift": numpy_fft.fftshift,
        "ifftshift": numpy_fft.ifftshift,
        "fftfreq": numpy_fft.fftfreq,
    }
    if fftn_shape is not None:
        f["fftn"] = numpy_fft.fftn
    fft = SimpleNamespace(**f)

    return fft


def decomposition_fft(field, bp_filter, **kwargs):

    fft = kwargs.get("fft_method", "numpy")
    if isinstance(fft, str):
        fft = get_numpy(fft, shape=field.shape)
    normalize = kwargs.get("normalize", False)
    mask = kwargs.get("mask", None)
    input_domain = kwargs.get("input_domain", "spatial")
    output_domain = kwargs.get("output_domain", "spatial")
    compute_stats = kwargs.get("compute_stats", True)
    compact_output = kwargs.get("compact_output", False)
    subtract_mean = kwargs.get("subtract_mean", False)

    if normalize and not compute_stats:
        compute_stats = True

    if len(field.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    if mask is not None and mask.shape != field.shape:
        raise ValueError(
            "Dimension mismatch between field and mask:"
            + "field.shape="
            + str(field.shape)
            + ",mask.shape"
            + str(mask.shape)
        )

    if field.shape[0] != bp_filter["weights_2d"].shape[1]:
        raise ValueError(
            "dimension mismatch between field and bp_filter: "
            + "field.shape[0]=%d , " % field.shape[0]
            + "bp_filter['weights_2d'].shape[1]"
            "=%d" % bp_filter["weights_2d"].shape[1]
        )

    if (
        input_domain == "spatial"
        and int(field.shape[1] / 2) + 1 != bp_filter["weights_2d"].shape[2]
    ):
        raise ValueError(
            "Dimension mismatch between field and bp_filter: "
            "int(field.shape[1]/2)+1=%d , " % (int(field.shape[1] / 2) + 1)
            + "bp_filter['weights_2d'].shape[2]"
            "=%d" % bp_filter["weights_2d"].shape[2]
        )

    if (
        input_domain == "spectral"
        and field.shape[1] != bp_filter["weights_2d"].shape[2]
    ):
        raise ValueError(
            "Dimension mismatch between field and bp_filter: "
            "field.shape[1]=%d , " % (field.shape[1] + 1)
            + "bp_filter['weights_2d'].shape[2]"
            "=%d" % bp_filter["weights_2d"].shape[2]
        )

    if output_domain != "spectral":
        compact_output = False

    if np.any(~np.isfinite(field)):
        raise ValueError("field contains non-finite values")

    result = {}
    means = []
    stds = []

    if subtract_mean and input_domain == "spatial":
        field_mean = np.mean(field)
        field = field - field_mean
        result["field_mean"] = field_mean

    if input_domain == "spatial":
        field_fft = fft.rfft2(field)
    else:
        field_fft = field
    if output_domain == "spectral" and compact_output:
        weight_masks = []
    field_decomp = []

    for k in range(len(bp_filter["weights_1d"])):
        field_ = field_fft * bp_filter["weights_2d"][k, :, :]

        if output_domain == "spatial" or (compute_stats and mask is not None):
            field__ = fft.irfft2(field_)
        else:
            field__ = field_

        if compute_stats:
            if output_domain == "spatial" or (compute_stats and mask is not None):
                if mask is not None:
                    masked_field = field__[mask]
                else:
                    masked_field = field__
                mean = np.mean(masked_field)
                std = np.std(masked_field)
            else:
                mean = spectral.mean(field_, bp_filter["shape"])
                std = spectral.std(field_, bp_filter["shape"])

            means.append(mean)
            stds.append(std)

        if output_domain == "spatial":
            field_ = field__
        if normalize:
            field_ = (field_ - mean) / std
        if output_domain == "spectral" and compact_output:
            weight_mask = bp_filter["weights_2d"][k, :, :] > 1e-12
            field_ = field_[weight_mask]
            weight_masks.append(weight_mask)
        field_decomp.append(field_)

    result["domain"] = output_domain
    result["normalized"] = normalize
    result["compact_output"] = compact_output

    if output_domain == "spatial" or not compact_output:
        field_decomp = np.stack(field_decomp)

    result["cascade_levels"] = field_decomp
    if output_domain == "spectral" and compact_output:
        result["weight_masks"] = np.stack(weight_masks)

    if compute_stats:
        result["means"] = means
        result["stds"] = stds

    return result


def recompose_fft(decomp, **kwargs):
    levels = decomp["cascade_levels"]
    if decomp["normalized"]:
        mu = decomp["means"]
        sigma = decomp["stds"]

    if not decomp["normalized"] and not (
        decomp["domain"] == "spectral" and decomp["compact_output"]
    ):
        result = np.sum(levels, axis=0)
    else:
        if decomp["compact_output"]:
            weight_masks = decomp["weight_masks"]
            result = np.zeros(weight_masks.shape[1:], dtype=complex)

            for i in range(len(levels)):
                if decomp["normalized"]:
                    result[weight_masks[i]] += levels[i] * sigma[i] + mu[i]
                else:
                    result[weight_masks[i]] += levels[i]
        else:
            result = [levels[i] * sigma[i] + mu[i] for i in range(len(levels))]
            result = np.sum(np.stack(result), axis=0)

    if "field_mean" in decomp:
        result += decomp["field_mean"]

    return result