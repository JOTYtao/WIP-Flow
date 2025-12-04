

import time
import warnings
import numpy as np
from scipy.ndimage import map_coordinates


def extrapolate(
    precip,
    velocity,
    timesteps,
    outval=np.nan,
    xy_coords=None,
    allow_nonfinite_values=False,
    vel_timestep=1,
    **kwargs,
):

    verbose = kwargs.get("verbose", False)
    displacement_prev = kwargs.get("displacement_prev", None)
    n_iter = kwargs.get("n_iter", 1)
    return_displacement = kwargs.get("return_displacement", False)
    interp_order = kwargs.get("interp_order", 1)
    map_coordinates_mode = kwargs.get("map_coordinates_mode", "constant")

    if precip is not None and interp_order > 1:
        minval = np.nanmin(precip)
        mask_min = (precip > minval).astype(float)
        if allow_nonfinite_values:
            mask_finite = np.isfinite(precip)
            precip = precip.copy()
            precip[~mask_finite] = 0.0
            mask_finite = mask_finite.astype(float)
        else:
            mask_finite = np.ones(precip.shape)

    prefilter = True if interp_order > 1 else False

    if isinstance(timesteps, int):
        timesteps = np.arange(1, timesteps + 1)
        vel_timestep = 1.0
    elif np.any(np.diff(timesteps) <= 0.0):
        raise ValueError("the given timestep sequence is not monotonously increasing")

    timestep_diff = np.hstack([[timesteps[0]], np.diff(timesteps)])

    if verbose:
        print("Computing the advection with the semi-lagrangian scheme.")
        t0 = time.time()

    if precip is not None and outval == "min":
        outval = np.nanmin(precip)

    if xy_coords is None:
        x_values, y_values = np.meshgrid(
            np.arange(velocity.shape[2]), np.arange(velocity.shape[1]), copy=False
        )

        xy_coords = np.stack([x_values, y_values])

    def interpolate_motion(displacement, velocity_inc, td):
        coords_warped = xy_coords + displacement
        coords_warped = [coords_warped[1, :, :], coords_warped[0, :, :]]

        velocity_inc_x = map_coordinates(
            velocity[0, :, :], coords_warped, mode="nearest", order=1, prefilter=False
        )
        velocity_inc_y = map_coordinates(
            velocity[1, :, :], coords_warped, mode="nearest", order=1, prefilter=False
        )

        velocity_inc[0, :, :] = velocity_inc_x
        velocity_inc[1, :, :] = velocity_inc_y

        if n_iter > 1:
            velocity_inc /= n_iter

        velocity_inc *= td / vel_timestep

    precip_extrap = []
    if displacement_prev is None:
        displacement = np.zeros((2, velocity.shape[1], velocity.shape[2]))
        velocity_inc = velocity.copy() * timestep_diff[0] / vel_timestep
    else:
        displacement = displacement_prev.copy()
        velocity_inc = np.empty(velocity.shape)
        interpolate_motion(displacement, velocity_inc, timestep_diff[0])

    for ti, td in enumerate(timestep_diff):
        if n_iter > 0:
            for k in range(n_iter):
                interpolate_motion(displacement - velocity_inc / 2.0, velocity_inc, td)
                displacement -= velocity_inc
                interpolate_motion(displacement, velocity_inc, td)
        else:
            if ti > 0 or displacement_prev is not None:
                interpolate_motion(displacement, velocity_inc, td)

            displacement -= velocity_inc

        coords_warped = xy_coords + displacement
        coords_warped = [coords_warped[1, :, :], coords_warped[0, :, :]]

        if precip is not None:
            precip_warped = map_coordinates(
                precip,
                coords_warped,
                mode=map_coordinates_mode,
                cval=outval,
                order=interp_order,
                prefilter=prefilter,
            )

            if interp_order > 1:
                mask_warped = map_coordinates(
                    mask_min,
                    coords_warped,
                    mode=map_coordinates_mode,
                    cval=0,
                    order=1,
                    prefilter=False,
                )
                precip_warped[mask_warped < 0.5] = minval

                mask_warped = map_coordinates(
                    mask_finite,
                    coords_warped,
                    mode=map_coordinates_mode,
                    cval=0,
                    order=1,
                    prefilter=False,
                )
                precip_warped[mask_warped < 0.5] = np.nan

            precip_extrap.append(np.reshape(precip_warped, precip.shape))

    if verbose:
        print("--- %s seconds ---" % (time.time() - t0))

    if precip is not None:
        if not return_displacement:
            return np.stack(precip_extrap)
        else:
            return np.stack(precip_extrap), displacement
    else:
        return None, displacement