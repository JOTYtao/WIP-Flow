
import numpy as np
from numpy.ma.core import MaskedArray
from functools import wraps
from models.FM.AR_OF.shitomasi import detection
from models.FM.AR_OF.interpolate import idwinterp2d
import warnings
import time
import cv2
import scipy.spatial
def morph_opening(input_image, thr, n):
    input_image = input_image.copy()

    # Check if a MaskedArray is used. If not, mask the ndarray
    to_ndarray = False
    if not isinstance(input_image, MaskedArray):
        to_ndarray = True
        input_image = np.ma.masked_invalid(input_image)

    np.ma.set_fill_value(input_image, input_image.min())

    # Convert to binary image
    field_bin = np.ndarray.astype(input_image.filled() > thr, "uint8")

    # Build a structuring element of size n
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))

    # Apply morphological opening (i.e. erosion then dilation)
    field_bin_out = cv2.morphologyEx(field_bin, cv2.MORPH_OPEN, kernel)

    # Build mask to be applied on the original image
    mask = (field_bin - field_bin_out) > 0

    # Filter out small isolated pixels based on mask
    input_image[mask] = np.nanmin(input_image)

    if to_ndarray:
        input_image = np.array(input_image)

    return input_image
def decluster(coord, input_array, scale, min_samples=1, verbose=False):
    coord = np.copy(coord)
    input_array = np.copy(input_array)

    # check inputs
    if np.any(~np.isfinite(input_array)):
        raise ValueError("input_array contains non-finite values")

    if input_array.ndim == 1:
        nvar = 1
        input_array = input_array[:, None]
    elif input_array.ndim == 2:
        nvar = input_array.shape[1]
    else:
        raise ValueError(
            "input_array must have 1 (n) or 2 dimensions (n, m), but it has %i"
            % input_array.ndim
        )

    if coord.ndim != 2:
        raise ValueError(
            "coord must have 2 dimensions (n, d), but it has %i" % coord.ndim
        )
    if coord.shape[0] != input_array.shape[0]:
        raise ValueError(
            "the number of samples in the input_array does not match the "
            + "number of coordinates %i!=%i" % (input_array.shape[0], coord.shape[0])
        )

    if np.isscalar(scale):
        scale = float(scale)
    else:
        scale = np.copy(scale)
        if scale.ndim != 1:
            raise ValueError(
                "scale must have 1 dimension (d), but it has %i" % scale.ndim
            )
        if scale.shape[0] != coord.shape[1]:
            raise ValueError(
                "scale must have %i elements, but it has %i"
                % (coord.shape[1], scale.shape[0])
            )
        scale = scale[None, :]

    # reduce original coordinates
    coord_ = np.floor(coord / scale)

    # keep only unique pairs of the reduced coordinates
    ucoord_ = np.unique(coord_, axis=0)

    # loop through these unique values and average data points which belong to
    # the same cluster
    dinput = np.empty(shape=(0, nvar))
    dcoord = np.empty(shape=(0, coord.shape[1]))
    for i in range(ucoord_.shape[0]):
        idx = np.all(coord_ == ucoord_[i, :], axis=1)
        npoints = np.sum(idx)
        if npoints >= min_samples:
            dinput = np.append(
                dinput, np.median(input_array[idx, :], axis=0)[None, :], axis=0
            )
            dcoord = np.append(
                dcoord, np.median(coord[idx, :], axis=0)[None, :], axis=0
            )

    if verbose:
        print("--- %i samples left after declustering ---" % dinput.shape[0])

    return dcoord, dinput
def detect_outliers(input_array, thr, coord=None, k=None, verbose=False):
    """
    Detect outliers in a (multivariate and georeferenced) dataset.

    Assume a (multivariate) Gaussian distribution and detect outliers based on
    the number of standard deviations from the mean.

    If spatial information is provided through coordinates, the outlier
    detection can be localized by considering only the k-nearest neighbours
    when computing the local mean and standard deviation.

    Parameters
    ----------
    input_array: array_like
        Array of shape (n) or (n, m), where *n* is the number of samples and
        *m* the number of variables. If *m* > 1, the Mahalanobis distance
        is used.
        All values in ``input_array`` are required to have finite values.
    thr: float
        The number of standard deviations from the mean used to define an outlier.
    coord: array_like or None, optional
        Array of shape (n, d) containing the coordinates of the input data into
        a space of *d* dimensions.
        Passing ``coord`` requires that ``k`` is not None.
    k: int or None, optional
        The number of nearest neighbours used to localize the outlier
        detection. If set to None (the default), it employs all the data points (global
        detection). Setting ``k`` requires that ``coord`` is not None.
    verbose: bool, optional
        Print out information.

    Returns
    -------
    out: array_like
        A 1-D boolean array of shape (n) with True values indicating the outliers
        detected in ``input_array``.
    """

    input_array = np.copy(input_array)

    if np.any(~np.isfinite(input_array)):
        raise ValueError("input_array contains non-finite values")

    if input_array.ndim == 1:
        nsamples = input_array.size
        nvar = 1
    elif input_array.ndim == 2:
        nsamples = input_array.shape[0]
        nvar = input_array.shape[1]
    else:
        raise ValueError(
            f"input_array must have 1 (n) or 2 dimensions (n, m), "
            f"but it has {input_array.ndim}"
        )

    if nsamples < 2:
        return np.zeros(nsamples, dtype=bool)

    if coord is not None and k is not None:
        coord = np.copy(coord)
        if coord.ndim == 1:
            coord = coord[:, None]

        elif coord.ndim > 2:
            raise ValueError(
                "coord must have 2 dimensions (n, d)," f"but it has {coord.ndim}"
            )

        if coord.shape[0] != nsamples:
            raise ValueError(
                "the number of samples in input_array does not match the "
                f"number of coordinates {nsamples}!={coord.shape[0]}"
            )

        k = np.min((nsamples, k + 1))

    # global

    if k is None or coord is None:
        if nvar == 1:
            # univariate
            zdata = np.abs(input_array - np.mean(input_array)) / np.std(input_array)
            outliers = zdata > thr
        else:
            # multivariate (mahalanobis distance)
            zdata = input_array - np.mean(input_array, axis=0)
            V = np.cov(zdata.T)
            try:
                VI = np.linalg.inv(V)
                MD = np.sqrt(np.dot(np.dot(zdata, VI), zdata.T).diagonal())
            except np.linalg.LinAlgError as err:
                warnings.warn(f"{err} during outlier detection")
                MD = np.zeros(nsamples)
            outliers = MD > thr

    # local
    else:
        tree = scipy.spatial.cKDTree(coord)
        __, inds = tree.query(coord, k=k)
        outliers = np.empty(shape=0, dtype=bool)
        for i in range(inds.shape[0]):
            if nvar == 1:
                # univariate
                thisdata = input_array[i]
                neighbours = input_array[inds[i, 1:]]
                thiszdata = np.abs(thisdata - np.mean(neighbours)) / np.std(neighbours)
                outliers = np.append(outliers, thiszdata > thr)
            else:
                # multivariate (mahalanobis distance)
                thisdata = input_array[i, :]
                neighbours = input_array[inds[i, 1:], :].copy()
                thiszdata = thisdata - np.mean(neighbours, axis=0)
                neighbours = neighbours - np.mean(neighbours, axis=0)
                V = np.cov(neighbours.T)
                try:
                    VI = np.linalg.inv(V)
                    MD = np.sqrt(np.dot(np.dot(thiszdata, VI), thiszdata.T))
                except np.linalg.LinAlgError as err:
                    warnings.warn(f"{err} during outlier detection")
                    MD = 0
                outliers = np.append(outliers, MD > thr)

    if verbose:
        print(f"--- {np.sum(outliers)} outliers detected ---")

    return outliers
def track_features(
    prvs_image,
    next_image,
    points,
    winsize=(50, 50),
    nr_levels=3,
    criteria=(3, 10, 0),
    flags=0,
    min_eig_thr=1e-4,
    verbose=False,
):
    prvs_img = prvs_image.copy()
    next_img = next_image.copy()
    p0 = np.copy(points)

    # Check if a MaskedArray is used. If not, mask the ndarray
    if not isinstance(prvs_img, MaskedArray):
        prvs_img = np.ma.masked_invalid(prvs_img)
    np.ma.set_fill_value(prvs_img, prvs_img.min())

    if not isinstance(next_img, MaskedArray):
        next_img = np.ma.masked_invalid(next_img)
    np.ma.set_fill_value(next_img, next_img.min())

    # scale between 0 and 255
    im_min = prvs_img.min()
    im_max = prvs_img.max()
    if (im_max - im_min) > 1e-8:
        prvs_img = (prvs_img.filled() - im_min) / (im_max - im_min) * 255
    else:
        prvs_img = prvs_img.filled() - im_min

    im_min = next_img.min()
    im_max = next_img.max()
    if (im_max - im_min) > 1e-8:
        next_img = (next_img.filled() - im_min) / (im_max - im_min) * 255
    else:
        next_img = next_img.filled() - im_min

    # convert to 8-bit
    prvs_img = np.ndarray.astype(prvs_img, "uint8")
    next_img = np.ndarray.astype(next_img, "uint8")

    # Lucas-Kanade
    # TODO: use the error returned by the OpenCV routine
    params = dict(
        winSize=winsize,
        maxLevel=nr_levels,
        criteria=criteria,
        flags=flags,
        minEigThreshold=min_eig_thr,
    )
    p1, st, __ = cv2.calcOpticalFlowPyrLK(prvs_img, next_img, p0, None, **params)

    # keep only features that have been found
    st = np.atleast_1d(st.squeeze()) == 1
    if np.any(st):
        p1 = p1[st, :]
        p0 = p0[st, :]

        # extract vectors
        xy = p0
        uv = p1 - p0

    else:
        xy = uv = np.empty(shape=(0, 2))

    if verbose:
        print(f"--- {xy.shape[0]} sparse vectors found ---")

    return xy, uv
def check_input_frames(
    minimum_input_frames=2, maximum_input_frames=np.inf, just_ndim=False
):
    """
    Check that the input_images used as inputs in the optical-flow
    methods have the correct shape (t, x, y ).
    """

    def _check_input_frames(motion_method_func):
        @wraps(motion_method_func)
        def new_function(*args, **kwargs):
            """
            Return new function with the checks prepended to the
            target motion_method_func function.
            """

            input_images = args[0]
            if input_images.ndim != 3:
                raise ValueError(
                    "input_images dimension mismatch.\n"
                    f"input_images.shape: {str(input_images.shape)}\n"
                    "(t, x, y ) dimensions expected"
                )

            if not just_ndim:
                num_of_frames = input_images.shape[0]

                if minimum_input_frames < num_of_frames > maximum_input_frames:
                    raise ValueError(
                        f"input_images frames {num_of_frames} mismatch.\n"
                        f"Minimum frames: {minimum_input_frames}\n"
                        f"Maximum frames: {maximum_input_frames}\n"
                    )

            return motion_method_func(*args, **kwargs)

        return new_function

    return _check_input_frames
@check_input_frames(2)
def dense_lucaskanade(
    input_images,
    lk_kwargs=None,
    fd_method="shitomasi",
    fd_kwargs=None,
    interp_method="idwinterp2d",
    interp_kwargs=None,
    dense=True,
    nr_std_outlier=3,
    k_outlier=30,
    size_opening=3,
    decl_scale=20,
    verbose=False,
):
    input_images = input_images.copy()

    if verbose:
        print("Computing the motion field with the Lucas-Kanade method.")
        t0 = time.time()

    nr_fields = input_images.shape[0]
    domain_size = (input_images.shape[1], input_images.shape[2])

    feature_detection_method = detection
    interpolation_method = idwinterp2d

    if fd_kwargs is None:
        fd_kwargs = dict()
    if fd_method == "tstorm":
        fd_kwargs.update({"output_feat": True})

    if lk_kwargs is None:
        lk_kwargs = dict()

    if interp_kwargs is None:
        interp_kwargs = dict()

    xy = np.empty(shape=(0, 2))
    uv = np.empty(shape=(0, 2))
    for n in range(nr_fields - 1):
        # extract consecutive images
        prvs_img = input_images[n, :, :].copy()
        next_img = input_images[n + 1, :, :].copy()

        # Check if a MaskedArray is used. If not, mask the ndarray
        if not isinstance(prvs_img, MaskedArray):
            prvs_img = np.ma.masked_invalid(prvs_img)
        np.ma.set_fill_value(prvs_img, prvs_img.min())

        if not isinstance(next_img, MaskedArray):
            next_img = np.ma.masked_invalid(next_img)
        np.ma.set_fill_value(next_img, next_img.min())

        # remove small noise with a morphological operator (opening)
        if size_opening > 0:
            prvs_img = morph_opening(prvs_img, prvs_img.min(), size_opening)
            next_img = morph_opening(next_img, next_img.min(), size_opening)

        # features detection
        points = feature_detection_method(prvs_img, **fd_kwargs).astype(np.float32)

        # skip loop if no features to track
        if points.shape[0] == 0:
            continue

        # get sparse u, v vectors with Lucas-Kanade tracking
        xy_, uv_ = track_features(prvs_img, next_img, points, **lk_kwargs)

        # skip loop if no vectors
        if xy_.shape[0] == 0:
            continue

        # stack vectors
        xy = np.append(xy, xy_, axis=0)
        uv = np.append(uv, uv_, axis=0)

    # return zero motion field is no sparse vectors are found
    if xy.shape[0] == 0:
        if dense:
            return np.zeros((2, domain_size[0], domain_size[1]))
        else:
            return xy, uv

    # detect and remove outliers
    outliers = detect_outliers(uv, nr_std_outlier, xy, k_outlier, verbose)
    xy = xy[~outliers, :]
    uv = uv[~outliers, :]

    if verbose:
        print("--- LK found %i sparse vectors ---" % xy.shape[0])

    # return sparse vectors if required
    if not dense:
        return xy, uv

    # decluster sparse motion vectors
    if decl_scale > 1:
        xy, uv = decluster(xy, uv, decl_scale, 1, verbose)

    # return zero motion field if no sparse vectors are left for interpolation
    if xy.shape[0] == 0:
        return np.zeros((2, domain_size[0], domain_size[1]))

    # interpolation
    xgrid = np.arange(domain_size[1])
    ygrid = np.arange(domain_size[0])
    uvgrid = interpolation_method(xy, uv, xgrid, ygrid, **interp_kwargs)

    if verbose:
        print("--- total time: %.2f seconds ---" % (time.time() - t0))

    return uvgrid