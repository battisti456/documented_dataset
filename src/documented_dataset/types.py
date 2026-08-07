from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

import xarray as xr

if TYPE_CHECKING:
    from .documented_dataset import Documented_Dataset

type AttributeName = str|StrEnum
DocumentedDatasetType = TypeVar("DocumentedDatasetType", bound='Documented_Dataset')
DocumentedBulkFunction = Callable[[type[DocumentedDatasetType]],xr.DataArray|dict[AttributeName,xr.DataArray]]
DocumentedFunction = Callable[[type[DocumentedDatasetType]],xr.DataArray]