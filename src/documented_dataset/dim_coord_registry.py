from dataclasses import dataclass
from typing import Any, Generic, Unpack

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike

from .attrs import Attrs
from .types import DocumentedDatasetType


@dataclass(frozen = True)
class Dim_Coord_Assn:
    dim_coord:'Dim_Coord'
    data:xr.DataArray

def format_name(name:str) -> str:
    return name.lower().replace(' ','_')

class Dim_Coord(str,Generic[DocumentedDatasetType]):  # noqa: UP046
    _dd:type[DocumentedDatasetType]
    _declared_dtype:DTypeLike|None = None
    _declared_attrs:Attrs
    def __new__(
            cls, 
            name, 
            dd:type[DocumentedDatasetType], 
        ):
        val = str.__new__(cls, name)
        val._dd = dd
        val._declared_attrs = {}
        return val
    @property
    def coord(self) -> xr.DataArray:
        return self._dd._ds.coords[self]
    @property
    def dim(self) -> str:
        return str(self)
    @property
    def has_coord(self) -> bool:
        return self in self._dd._ds.coords
    def __call__(
            self,
            value:ArrayLike|None = None,
            attrs:Attrs|None = None,
            dtype:DTypeLike|None = None,
            **kwargs:Unpack[Attrs]
        ) -> 'Dim_Coord_Assn|Dim_Coord':
        attrs = self.__dict__.get("_declared_attrs",{}) | ({} if attrs is None else attrs) | kwargs#type:ignore
        assert isinstance(attrs,dict)
        if self.has_coord:
            self.coord.attrs.update(attrs)
            if dtype is not None and dtype != self.coord.dtype:
                self._dd._ds.coords[self] = self._dd._ds.coords[self].astype(dtype)
        if value is None:
            self._declared_attrs = attrs#type:ignore
            if dtype is not None:
                self._declared_dtype = dtype
            return self
        if dtype is not None:
            pass
        elif self in self._dd._ds.coords:
            dtype = self._dd._ds.coords[self].dtype
        elif hasattr(self,"_declared_dtype"):
            dtype = self._declared_dtype
        to_assign:xr.DataArray
        if isinstance(value,xr.DataArray):
            if "dim0" in value.dims:
                to_assign = value.swap_dims({"dim0": self})
            else:
                to_assign = value
            if dtype is not None:
                to_assign = to_assign.astype(dtype)
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
    def __getattr__(self, attr):
        if attr in ("keys", "__getitem__", "__iter__"):
            raise AttributeError(attr)
        attr_name = self._attr_name()
        if attr in attr_name:
            return attr_name[attr]
        raise AttributeError(attr)
    def _attr_name(self):
        if not self.has_coord: return {}
        formatted_names = {name:format_name(name) for name in self.coord.to_numpy().tolist() if isinstance(name,str)}
        return {attr:name for name, attr in formatted_names.items() if attr.isidentifier()}
    


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
    def array(
        self,
        data:ArrayLike,
        *args:Dim_Coord|Dim_Coord_Assn,
        attrs:Attrs|None = None,
        dtype:DTypeLike|None = None,
        **kwargs:Unpack[Attrs]
    ) -> xr.DataArray:
        attrs = ({} if attrs is None else attrs) | kwargs#type:ignore
        assert isinstance(attrs,dict)
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
        array = xr.DataArray(
            np.asarray(data, dtype = dtype),
            coords,
            dims,
            attrs = attrs
        )
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
