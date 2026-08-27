import textwrap

import xarray as xr
from numpy.typing import DTypeLike

from .dim_coord_registry import Dim_Coord, Dim_Coord_Registry
from .types import DocumentedDatasetType


def indent(string:str) -> str:
    return textwrap.indent(string,"    ")
def d_a_to_unit_str(dtype:DTypeLike|None,attrs) -> str:
    return f"{dtype}" + (
        f" [{attrs['units']}]"
        if 'units' in attrs else ""
    )
def da_to_unit_str(da:xr.DataArray) -> str:
    return d_a_to_unit_str(da.dtype,da.attrs)
def da_to_str(name:str,da:xr.DataArray,coords) -> str:
    return '\n'.join((
        f"{name}:xr.DataArray",
        "\"\"\"",
        indent('\n'.join((
            da.attrs.get('long_name',name),
            "### Dimensions",
            '\n'.join(
                f"- {dim} : {da_to_unit_str(coords[dim]) if dim in coords else dim}"
                for dim in da.dims
            ),
            "### Units",
            da_to_unit_str(da),
            "### Description",
            da.attrs.get('description',"")
        ))),
        "\"\"\"",
    ))

def dc_to_str(dc:Dim_Coord) -> str:
    if dc.has_coord:
        attrs = dc.coord.attrs
        dtype = dc.coord.dtype
    else:
        attrs = dc._declared_attrs
        dtype = dc._declared_dtype
    attr_name = dc.values._attr_name()
    return '\n'.join((
        f"class _{dc}_values(Protocol):",
        indent('\n'.join(
            f"{attr}: Literal[\"{name}\"]"
            for attr, name
            in attr_name.items()
        )) if attr_name else indent("..."),
        f"{dc}:Dim_Coord[Any,_{dc}_values]",
        "\"\"\"",
        indent('\n'.join((
            attrs.get('long_name',dc),
            "### Units",
            d_a_to_unit_str(dtype,attrs),
            "### Description",
            attrs.get('description',"")
        ))),
        "\"\"\"",
    ))

def dcr_to_str(dcr:Dim_Coord_Registry):
    return '\n'.join((
        "class _dims_type(Dim_Coord_Registry):",
        indent('\n'.join(
            dc_to_str(dc) for dc in dcr._registered.values())),
    ))
    
def dd_to_str(dd:type['DocumentedDatasetType']) -> str:
    return '\n'.join((
            "from typing import Any, Literal, Protocol",
            "",
            "import xarray as xr",
            "from documented_dataset import Dim_Coord, Dim_Coord_Registry, Documented_Dataset",
            "",
            f"class {dd.__name__}(Documented_Dataset):",
            indent('\n'.join((
                dcr_to_str(dd.dims),
                "dims:_dims_type #type:ignore",
                '\n'.join((
                    da_to_str(name,da,dd._ds.coords)
                    for name, da in dd._get_name_da()
                ))
            )))
        ))