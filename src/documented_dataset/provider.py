from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import xarray as xr

from .types import DocumentedDatasetType


class DocumentedDatasetLoopingProviderExecution(Exception): ...

ReturnVar = TypeVar("ReturnVar", bound=xr.DataArray|dict[str,xr.DataArray])
CacheVar = TypeVar("CacheVar")
PriorityVar = TypeVar("PriorityVar")

@dataclass
class Base_Provider(ABC, Generic[DocumentedDatasetType,ReturnVar,CacheVar,PriorityVar]):  # noqa: UP046
    func:Callable[[type[DocumentedDatasetType]],ReturnVar]
    storage_level:PriorityVar
    cache_policy:CacheVar
    included_variables:tuple[str,...]
    coordinate:bool
    _is_running:bool = field(
        default=False,
        init=False,
        repr=False
    )
    def __call__(self,cls:type[DocumentedDatasetType]) -> dict[str,xr.DataArray]:
        if self._is_running:
            raise DocumentedDatasetLoopingProviderExecution(
                f"Registered function {self.func.__name__} is being run within its own execution." \
                "There is a loop in your variable logic."
            )
        else:
            self._is_running = True
            try:
                val = self.func(cls)
                if isinstance(val,xr.DataArray):
                    val = {self.included_variables[0]:val}
                return val
            finally:
                self._is_running = False
    


@dataclass
class Single_Provider(Base_Provider[DocumentedDatasetType,xr.DataArray,bool|None,int|None]): ...

@dataclass
class Bulk_Provider(Base_Provider[DocumentedDatasetType,dict[str,xr.DataArray],bool|None|dict[str,bool|None],int|None|dict[str,int|None]]):  ...

Provider = Single_Provider[DocumentedDatasetType]|Bulk_Provider[DocumentedDatasetType]