"""This module provides classes for fetching data from UDA

The main class for user applications is the UDASignal class. This class
provides a way to fetch data from a UDA server. The UDASignal class is a
subclass of the Signal class and provides the same interface as the Signal class.
"""

import re
import numpy as np
import xarray as xr
from typing import Iterable
from .signal import Signal


class UDASignal(Signal):
    def __init__(
        self,
        treepath: str,
        dims: Iterable[str] = ("times",),
        fetch_units: bool = True,
        is_image: bool = False,
        timeout: int = 10,
    ):
        """Create a signal object that fetches data from an MDSplus tree

        Arguments:
            treepath: The name of the signal within the store to load. e.g. `magentics/ip`
            dims: See documentation for the Signal class. Defaults to ('times',)
        """
        super().__init__()
        self.treepath = treepath
        self.dims = dims
        self.fetch_units = fetch_units
        self.timeout = timeout
        self.is_image = is_image

    def gather(self, shot) -> dict:
        """Gather the data for a shot

        Arguments:
            shot (int): The shot number to gather the data for

        Returns:
            dict: A dictionary containing the data gathered for the signal. The dictionary
                will contain a key 'data' with the data, and keys for each dimension of the
                data, with the values being the values of the dimensions. If the with_units
                attribute is True, the dictionary will also contain a key 'units' with the units
                of the data and dimensions.
        """
        import pyuda

        client = pyuda.Client()
        client.set_property("get_meta", True)
        client.set_property("timeout", self.timeout)

        try:
            if not self.is_image:
                signal = self.load_signal(client, shot, self.treepath)
            else:
                signal = self.load_image(client, shot, self.treepath)
        except pyuda.ServerException as e:
            raise RuntimeError(
                f'Could not load signal {self.treepath} for shot "{shot}". Encountered exception: {e}'
            )

        dim_map = dict(zip(signal.dims, self.dims))
        signal = signal.rename(dim_map)
        signal = signal.transpose(*self.dims)

        data = dict(data=signal.values)
        for dim in signal.dims:
            data[dim] = signal[dim].values

        if self.fetch_units:
            units = {"data": signal.attrs.get("units", "")}
            for dim in signal.dims:
                units[dim] = signal[dim].attrs.get("units", "")
            data["units"] = units

        return data

    def load_signal(self, client, shot_num: int, name: str) -> xr.DataArray:
        signal = client.get(name, shot_num)
        dataset = self._convert_signal_to_dataset(name, signal)
        dataset = dataset.squeeze(drop=True)
        return dataset

    def load_image(self, client, shot_num: int, name: str) -> xr.Dataset | xr.DataArray:
        image = client.get_images(name, shot_num)
        dataset = self._convert_image_to_dataset(image)
        dataset.name = name
        dataset.attrs["name"] = name
        dataset.attrs["uda_name"] = name
        return dataset

    def _convert_signal_to_dataset(self, signal_name, signal) -> xr.DataArray:
        dim_names = self._normalize_dimension_names(signal)
        coords = {}
        for name, dim in zip(dim_names, signal.dims):
            data = dim.data
            coord = xr.DataArray(
                np.atleast_1d(data), dims=[name], attrs=dict(units=dim.units)
            )
            coords[name] = coord

        data = np.atleast_1d(signal.data)
        attrs = self._get_dataset_attributes(signal_name, signal)
        uda_name = signal_name

        data = xr.DataArray(data, dims=dim_names, coords=coords, attrs=attrs)
        if signal_name == "time":
            signal_name = "time_"

        data.name = signal_name
        data.attrs["name"] = data.name
        data.attrs["uda_name"] = uda_name

        return data

    def _convert_image_to_dataset(self, image) -> xr.DataArray:
        attrs = {
            name: getattr(image, name)
            for name in dir(image)
            if not name.startswith("_") and not callable(getattr(image, name))
        }

        attrs.pop("frame_times")
        attrs.pop("frames")

        time = np.atleast_1d(image.frame_times)
        coords = {"time": xr.DataArray(time, dims=["time"], attrs=dict(units="s"))}

        if image.is_color:
            frames = [np.dstack((frame.r, frame.g, frame.b)) for frame in image.frames]
            frames = np.stack(frames)
            if frames.shape[1] != image.height:
                frames = np.swapaxes(frames, 1, 2)
            dim_names = ["time", "height", "width", "channel"]

            attrs["IMAGE_SUBCLASS"] = "IMAGE_TRUECOLOR"
        else:
            frames = [frame.k for frame in image.frames]
            frames = np.stack(frames)
            frames = np.atleast_3d(frames)
            if frames.shape[1] != image.height:
                frames = np.swapaxes(frames, 1, 2)
            dim_names = ["time", "height", "width"]

            attrs["IMAGE_SUBCLASS"] = "IMAGE_INDEXED"

        dataset = xr.DataArray(frames, dims=dim_names, coords=coords, attrs=attrs)
        return dataset

    def _normalize_dimension_names(self, signal):
        """Make the dimension names sensible"""
        dims = [dim.label for dim in signal.dims]
        count = 0
        dim_names = []
        empty_names = ["", " ", "-"]

        for name in dims:
            # Create names for unlabelled dims
            if name in empty_names:
                name = f"dim_{count}"
                count += 1

            # Normalize weird names to standard names
            dim_names.append(name)

        dim_names = list(map(lambda x: x.lower(), dim_names))
        dim_names = [re.sub("[^a-zA-Z0-9_\n\\.]", "", dim) for dim in dim_names]
        return dim_names

    def _get_dataset_attributes(self, signal_name: str, signal) -> dict:
        metadata = self._get_signal_metadata_fields(signal, signal_name)

        attrs = {}
        for field in metadata:
            try:
                attrs[field] = getattr(signal, field)
            except TypeError:
                pass

        for key, attr in attrs.items():
            if isinstance(attr, np.generic):
                attrs[key] = attr.item()
            elif isinstance(attr, np.ndarray):
                attrs[key] = attr.tolist()
            elif isinstance(attr, tuple):
                attrs[key] = list(attr)
            elif attr is None:
                attrs[key] = "null"

        attrs.pop("rank", "")
        attrs.pop("shape", "")
        attrs.pop("time_index", "")
        return attrs

    def _get_signal_metadata_fields(self, signal, signal_name):
        """Retrieves the appropriate metadata field for a given signal"""
        return [
            attribute
            for attribute in self._remove_exceptions(signal_name, signal)
            if not attribute.startswith("_")
            and attribute not in ["data", "errors", "time", "meta", "dims"]
            and not callable(getattr(signal, attribute))
        ]

    def _remove_exceptions(self, signal_name, signal):
        """Handles when signal attributes contain exception objects"""
        signal_attributes = dir(signal)
        for attribute in signal_attributes:
            try:
                getattr(signal, attribute)
            except UnicodeDecodeError as exception:
                print(f"{signal_name} {attribute}: {exception}")
                signal_attributes.remove(attribute)
        return signal_attributes

    def cleanup_shot(self, shot: int):
        """Close the tree for this shot

        Arguments:
            shot (int): The shot number to close the tree for
        """
        pass

    def cleanup(self):
        """Cleanup"""
        pass
