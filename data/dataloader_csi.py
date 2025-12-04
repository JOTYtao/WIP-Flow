import os
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr


class SolarDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        years: Dict[str, Union[List[int], List[str]]],
        mode: str = "train",
        input_len: int = 4,
        pred_len: int = 12,
        stride: int = 1,
        batch_size: int =16,
        num_workers: int=10,
        pin_memory: bool = True,
        validation: bool = True,
        use_possible_starts: bool = True,
        nc_engine: str = "netcdf4",
        return_target_cot: bool = False,
        crop_size: Optional[Tuple[int, int]] = None,  # e.g. (256, 256)
        downsample: Optional[int] = None,             # e.g. 2, 4, ...
        seed: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.years = [str(y) for y in years[mode]]
        self.data_path = data_path
        self.stride = stride
        self.input_len = input_len
        self.pred_len = pred_len
        self.validation = validation
        self.use_possible_starts = use_possible_starts
        self.total_len = self.input_len + self.pred_len
        self.nc_engine = nc_engine
        self.return_target_cot = return_target_cot
        self.crop_size = crop_size
        self.downsample = downsample


        self.data_paths = {y: os.path.join(data_path, mode, y, f"CSI_{y}.nc") for y in self.years}
        self.cloud_paths = {y: os.path.join(data_path, mode, y, "COT.nc") for y in self.years}
        self.timestamps_csv = {y: os.path.join(data_path, mode, y, f"CSI_{y}_timestamps.csv") for y in self.years}
        self.possible_starts_paths = {y: os.path.join(data_path, mode, y, f"CSI_{y}_possible_starts.npy") for y in self.years}
        # self.cmv_path = {y: os.path.join(data_path, mode, y, f"CMV.npz") for y in self.years} 


        self.timestamps: Dict[str, pd.DataFrame] = {}
        self.time_feats: Dict[str, np.ndarray] = {}
        self.possible_starts: Dict[str, np.ndarray] = {}
        self._load_auxiliary_files()

 
        self.valid_indices: Dict[str, np.ndarray] = {}
        self.nitems_per_year: Dict[str, int] = {}
        self.year_mapping: Dict[int, Tuple[str, int]] = {}
        self.nitems = 0
        self._initialize_indices()

   
        if self.validation:
            rng = np.random.default_rng(seed)
            self.seeds = rng.integers(0, 2**31 - 1, size=self.nitems, dtype=np.int64)

     
        self._handles: Dict[Tuple[int, str], Dict[str, xr.Dataset]] = {}

        self._coords_cache: Dict[str, torch.Tensor] = {}

    # ------------ 文件加载与索引 ------------
    def _load_auxiliary_files(self):
        for y in self.years:
            ts_path = self.timestamps_csv[y]
            if not os.path.exists(ts_path):
                raise FileNotFoundError(f"Timestamps file not found: {ts_path}")
            ts = pd.read_csv(ts_path)
            self.timestamps[y] = ts
            t = pd.to_datetime(ts['StartTimeUTC'].values)
            months = (t.month.astype(np.float32)) / 12.0
            days = (t.day.astype(np.float32)) / 31.0
            hours = (t.hour.astype(np.float32)) / 24.0
            minutes = (t.minute.astype(np.float32)) / 60.0
            self.time_feats[y] = np.stack([months, days, hours, minutes], axis=1)

            if self.use_possible_starts:
                ps_path = self.possible_starts_paths[y]
                if not os.path.exists(ps_path):
                    raise FileNotFoundError(f"Possible starts file not found: {ps_path}")
                self.possible_starts[y] = np.load(ps_path)

    def _initialize_indices(self):
        self.nitems = 0
        self.year_mapping = {}
        for y in self.years:
            with xr.open_dataset(self.data_paths[y], engine=self.nc_engine) as ds:
                n_time = len(ds.time)
            if self.use_possible_starts and y in self.possible_starts:
                valid = self.possible_starts[y]
                valid = valid[valid <= n_time - self.total_len]
            else:
                valid = np.arange(0, n_time - self.total_len + 1, self.stride, dtype=np.int64)
            self.valid_indices[y] = valid
            self.nitems_per_year[y] = len(valid)
            for i, idx in enumerate(valid):
                self.year_mapping[self.nitems + i] = (y, int(idx))
            self.nitems += len(valid)


    @staticmethod
    def _normalize_coords(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lon_norm = 2 * np.pi * (lon - lon.min()) / (lon.max() - lon.min())
        lat_norm = 2 * np.pi * (lat - lat.min()) / (lat.max() - lat.min())
        return lon_norm, lat_norm

    def _prepare_coords_once(self, year: str, ds: xr.Dataset):
        if year in self._coords_cache:
            return
        lon = ds.longitude.values if 'longitude' in ds.variables else ds.lon.values
        lat = ds.latitude.values if 'latitude' in ds.variables else ds.lat.values
        lon_norm, lat_norm = self._normalize_coords(lon, lat)
        lon_grid, lat_grid = np.meshgrid(lon_norm, lat_norm, indexing='xy')
        sin_lon = np.sin(lon_grid)[np.newaxis, np.newaxis, :, :]
        cos_lon = np.cos(lon_grid)[np.newaxis, np.newaxis, :, :]
        sin_lat = np.sin(lat_grid)[np.newaxis, np.newaxis, :, :]
        cos_lat = np.cos(lat_grid)[np.newaxis, np.newaxis, :, :]
        coords = np.concatenate([sin_lon, cos_lon, sin_lat, cos_lat], axis=0)  # [4,1,H,W]
        self._coords_cache[year] = torch.from_numpy(coords.astype(np.float32))

    def _get_handle(self, year: str):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else -1
        key = (wid, year)
        handles = self._handles.get(key, None)
        if handles is None:
            ds_csi = xr.open_dataset(self.data_paths[year], engine=self.nc_engine, cache=True)
            ds_cot = xr.open_dataset(self.cloud_paths[year], engine=self.nc_engine, cache=True)
    
            # self._prepare_coords_once(year, ds_csi)
            handles = {"csi": ds_csi, "cot": ds_cot}
            self._handles[key] = handles
        return handles

    def __len__(self) -> int:
        return self.nitems

  
    @staticmethod
    def _rescale_data(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        return 2.0 * ((x - min_val) / (max_val - min_val)) - 1.0



    def __getitem__(self, idx: int):
        if self.validation:
            torch.manual_seed(int(self.seeds[idx]))
            np.random.seed(int(self.seeds[idx] % (2**32 - 1)))

        year_str, start_idx = self.year_mapping[idx]
        idx_end = start_idx + self.total_len
        h = self._get_handle(year_str)
        ds = h["csi"]
        cot = h["cot"]


        K_raw = ds["CSI"].isel(time=slice(start_idx, idx_end)).values  # [T,H,W], np
        COT_raw = cot["COT"].isel(time=slice(start_idx, idx_end)).values  # [T,H,W], np

      
        K = torch.from_numpy(K_raw).float().unsqueeze(0)   # [1,T,H,W]
        C = torch.from_numpy(COT_raw).float().unsqueeze(0) # [1,T,H,W]

        x = K[:, :self.input_len]          # [1,Tin,H,W]
        y = K[:, self.input_len:]          # [1,Tout,H,W]
        cx = C[:, :self.input_len]         # [1,Tin,H,W]
        cy = C[:, self.input_len:]         # [1,Tout,H,W]

        his_csi = self._rescale_data(x, min_val=0.05, max_val=1.2)
        y = self._rescale_data(y, min_val=0.05, max_val=1.2)
        cx = self._rescale_data(cx, min_val=0.0, max_val=150.0)
        C = self._rescale_data(C, min_val=0.0, max_val=150.0)
        if self.return_target_cot:
            cy = self._rescale_data(cy, min_val=0.0, max_val=150.0)

        out = {
            "his_csi_non": x,          # [1, Tin, H, W]
            "his_csi": his_csi,
            "input_cot": cx,       # [1, Tin, H, W]
            "target_csi": y,       # [1, Tout, H, W]
            "COT": C
        }
        if self.return_target_cot:
            out["target_cot"] = cy  # [1, Tout, H, W]
        return out