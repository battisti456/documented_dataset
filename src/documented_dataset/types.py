from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import xarray as xr

if TYPE_CHECKING:
    from .documented_dataset import Documented_Dataset

DocumentedDatasetType = TypeVar("DocumentedDatasetType", bound='Documented_Dataset')
DocumentedFunction = Callable[[type[DocumentedDatasetType]],xr.DataArray]