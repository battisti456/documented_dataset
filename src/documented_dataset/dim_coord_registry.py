from dataclasses import dataclass
from typing import Any, Generic, Unpack

import numpy as np
import xarray as xr
from attrs import Attrs

from .types import DocumentedDatasetType


@dataclass(frozen = True)
class Dim_Coord_Assn:
    dim_coord:'Dim_Coord'
    data:np.ndarray

class Dim_Coord(str,Generic['DocumentedDatasetType']):
    _dd:type['DocumentedDatasetType']
    _pre_declared_attrs:Attrs
    def __new__(
            cls, 
            name, 
            dd:type['DocumentedDatasetType'], 
        ):
        val = str.__new__(cls, name)
        val._dd = dd
        return val
    @property
    def coord(self) -> xr.DataArray:
        return self._dd._ds.coords[self]
    @property
    def dim(self) -> str:
        return str(self)
    def __call__(self,value:xr.DataArray|np.ndarray|None = None, attrs:Attrs|None = None,**kwargs:Unpack[Attrs]) -> 'Dim_Coord_Assn|Dim_Coord':
        if attrs is None:
            attrs = {}
        attrs = attrs | kwargs
        if hasattr(self,"_pre_declared_attrs"):
            attrs = self._pre_declared_attrs | attrs
        if value is None:
            self._pre_declared_attrs = attrs
            return self
        to_assign:xr.DataArray
        if isinstance(value,xr.DataArray):
            if "dim0" in value.dims:
                to_assign = value.swap_dims({"dim0": self})
            else:
                to_assign = value
            to_assign = to_assign.assign_attrs(attrs)
        else:
            if self in self._dd._ds.coords and "dtype" in self._dd._ds.coords[self].attrs:
                value = value.astype(self._dd._ds.coords[self].attrs["dtype"])
            to_assign = xr.DataArray(value, dims = (self,), attrs = attrs)
        if self in self._dd._ds.coords:
            new_values = np.concatenate([
                self.coord.values,
                np.asarray(value),
            ])
            self._dd._ds = self._dd._ds.reindex(
                {self: new_values},
                fill_value={
                    name:da.attrs["fill_value"] for name,da in self._dd._ds.items()
                    if "fill_value" in da.attrs
                }
            )
        else:
            self._dd._ds.coords.update({self:to_assign})
        return Dim_Coord_Assn(self,np.asarray(value))

class Dim_Coord_Registry(Generic['DocumentedDatasetType']):
    def __init__(self,dd:type['DocumentedDatasetType']):
        self._registered:dict[str,Dim_Coord] = {}
        self._dd = dd
    def load(self):
        for dim in self._dd._ds.dims:
            self._registered[dim] = Dim_Coord(dim,self._dd)#type:ignore
    def add_dim(self,name:str):
        self._registered[name] = Dim_Coord(name,self._dd)
    def __getattr__(self, name: str) -> Any:
        if name not in self._registered:
            self.add_dim(name)
        return self._registered[name]
    def array(self,data,*args:Dim_Coord|Dim_Coord_Assn, attrs:Attrs|None = None) -> xr.DataArray:
        if attrs is None:
            attrs = {}
        coords = {}
        dims = []
        for arg in args:
            if isinstance(arg,Dim_Coord_Assn):
                coords[arg.dim_coord] = arg.data
                dims.append(arg.dim_coord)
            elif arg in self._dd._ds.coords:
                coords[arg] = arg.coord
            else:
                dims.append(arg)
        return xr.DataArray(
            data,
            coords,
            dims,
            attrs = attrs
        )
