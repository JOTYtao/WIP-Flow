import os
import numpy as np
import xarray as xr
import pandas as pd 

def get_possible_starts_from_time(time_values, n_past_steps, n_future_steps, step_ns=30 * 60 * 10**9):
    if hasattr(time_values, "values"):
        times = time_values.values
    else:
        times = np.asarray(time_values)

    if times.size < (n_past_steps + n_future_steps):
        return np.array([], dtype=int)
    diff = np.diff(times).astype("timedelta64[ns]")
    frames_total = n_past_steps + n_future_steps
    counted = np.zeros(diff.shape, dtype=np.int32)
    counted[diff == np.timedelta64(step_ns, "ns")] = 1
    consec = np.zeros_like(counted)
    run = 0
    for i, flag in enumerate(counted):
        if flag == 1:
            run += 1
        else:
            run = 0
        consec[i] = run

    possible_indices = np.where(consec >= (frames_total - 1))[0]
    possible_starts = possible_indices - (frames_total - 1)
    possible_starts = possible_starts.astype(int)
    possible_starts.sort()
    valid = (possible_starts >= 0) & (possible_starts + frames_total <= times.shape[0])
    return possible_starts[valid]


def compute_and_save_possible_starts(nc_path, out_dir, nc_var="CAL",
                                     n_past_steps=4, n_future_steps=12,
                                     step_minutes=30, output_basename=None):
    assert os.path.exists(nc_path), f"File not found: {nc_path}"
    os.makedirs(out_dir, exist_ok=True)

    if output_basename is None:
        base = os.path.splitext(os.path.basename(nc_path))[0]
    else:
        base = output_basename
    ds = xr.open_dataset(nc_path) 
    if "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("No 'time' coordinate/dimension found in dataset.")
    time_values = ds["time"]
    if nc_var not in ds.variables and nc_var not in ds.data_vars:
        print(f"Warning: variable '{nc_var}' not found; proceeding with time axis only.")

    step_ns = step_minutes * 60 * 10**9
    possible_starts = get_possible_starts_from_time(
        time_values=time_values,
        n_past_steps=n_past_steps,
        n_future_steps=n_future_steps,
        step_ns=step_ns
    )

    npy_path = os.path.join(out_dir, f"{base}_possible_starts.npy")
    with open(npy_path, "wb") as f:
        np.save(f, possible_starts)
    print(f"Saved possible_starts to: {npy_path} (count={len(possible_starts)})")
    try:
        csv_path = os.path.join(out_dir, f"{base}_timestamps.csv")
        pd.DataFrame(time_values.values, columns=["StartTimeUTC"]).to_csv(csv_path, index=False)
        print(f"Saved timestamps to: {csv_path}")
    except Exception as e:
        print(f"Skip saving timestamps CSV due to: {e}")

    ds.close()
    return possible_starts


if __name__ == "__main__":
    nc_path = '/home/joty/code/flow_matching/data/test/2022/CSI_2022.nc'
    out_dir = '/home/joty/code/flow_matching/data/test/2022/'
    possible_starts = compute_and_save_possible_starts(
        nc_path=nc_path,
        out_dir=out_dir,
        nc_var="CAL",
        n_past_steps=4,
        n_future_steps=12,
        step_minutes=30,
        output_basename="CAL_2022"
    )
    print("First 10 starts:", possible_starts[:10])