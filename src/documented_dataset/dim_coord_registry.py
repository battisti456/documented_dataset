from dataclasses import dataclass
from typing import Any, Generic, Unpack

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from .attrs import Attrs
from .types import DocumentedDatasetType


@dataclass(frozen = True)
class Dim_Coord_Assn:
    dim_coord:'Dim_Coord'
    data:xr.DataArray

class Dim_Coord(str,Generic[DocumentedDatasetType]):  # noqa: UP046
    _dd:type[DocumentedDatasetType]
    attrs:Attrs
    def __new__(
            cls, 
            name, 
            dd:type[DocumentedDatasetType], 
        ):
        val = str.__new__(cls, name)
        val._dd = dd
        val.attrs = {}
        return val
    @property
    def coord(self) -> xr.DataArray:
        return self._dd._ds.coords[self]
    @property
    def dim(self) -> str:
        return str(self)
    def __call__(self,value:ArrayLike|None = None, attrs:Attrs|None = None,**kwargs:Unpack[Attrs]) -> 'Dim_Coord_Assn|Dim_Coord':
        self.attrs = self.attrs | ({} if attrs is None else attrs) | kwargs
        if value is None:
            return self
        to_assign:xr.DataArray
        dtype = None
        if "dtype" in self.attrs:
            dtype = self.attrs["dtype"]
        elif self in self._dd._ds.coords and "dtype" in self._dd._ds.coords[self].attrs:
            dtype = self._dd._ds.coords[self].attrs["dtype"]
        if isinstance(value,xr.DataArray):
            if "dim0" in value.dims:
                to_assign = value.swap_dims({"dim0": self})
            else:
                to_assign = value
            to_assign = to_assign.assign_attrs(attrs)
        else:
            to_assign = xr.DataArray(np.asarray(value,dtype=dtype), dims = (self,), attrs = attrs)
        if self in self._dd._ds.coords:
            value = np.asarray(value, dtype = dtype)
            missing_coords = value[~np.isin(value,self.coord.values)]
            if len(missing_coords) > 0:
                new_values = np.concatenate([
                    self.coord.values,
                    missing_coords,
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
        return Dim_Coord_Assn(self,to_assign)
    def __copy__(self):
        return str(self)
    def __deepcopy__(self, memo):
        return str(self)

class Dim_Coord_Registry(Generic[DocumentedDatasetType]):  # noqa: UP046
    def __init__(self,dd:type[DocumentedDatasetType]):
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
    def array(self,data:ArrayLike,*args:Dim_Coord|Dim_Coord_Assn, attrs:Attrs|None = None,**kwargs:Unpack[Attrs]) -> xr.DataArray:
        if attrs is None:
            attrs = {}
        attrs = attrs | kwargs
        coords = {}
        dims = []
        for arg in args:
            if isinstance(arg,Dim_Coord_Assn):
                coords[arg.dim_coord] = arg.data
                dims.append(arg.dim_coord)
            elif arg in self._dd._ds.coords:
                coords[arg] = arg.coord
                dims.append(arg)
            else:
                dims.append(arg)
        dtype = None
        if "dtype" in attrs:
            dtype = attrs["dtype"]
        array = xr.DataArray(
            np.asarray(data, dtype = dtype),
            coords,
            dims,
            attrs = attrs
        )
        array.attrs["dtype"] = array.dtype
        fill_value = attrs.get("fill_value",np.nan)
        full_coord_assignments = {
            dim:self._dd._ds.coords[dim]
            for dim in array.dims
            if dim in self._dd._ds.coords
            and not array.get_index(dim).equals(self._dd._ds.get_index(dim))
        }
        if full_coord_assignments:
            array = array.reindex(
                full_coord_assignments,
                fill_value=fill_value#type:ignore
            )
        return array
