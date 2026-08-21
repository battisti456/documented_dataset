import textwrap

import xarray as xr

from .attrs import Attrs
from .dim_coord_registry import Dim_Coord, Dim_Coord_Registry
from .types import DocumentedDatasetType


def indent(string:str) -> str:
    return textwrap.indent(string,"    ")
def attrs_to_unit_str(attrs:Attrs) -> str:
    return f"{attrs.get('dtype',None)}" + (
        f" [{attrs['units']}]"
        if 'units' in attrs else ""
    )
def da_to_str(name:str,da:xr.DataArray,coords) -> str:
    return '\n'.join((
        f"{name}:xr.DataArray",
        "\"\"\"",
        indent('\n'.join((
            da.attrs.get('long_name',name),
            "### Dimensions",
            '\n'.join(
                f"- {dim} : {attrs_to_unit_str(coords[dim].attrs) if dim in coords else dim}"#type:ignore
                for dim in da.dims
            ),
            "### Units",
            attrs_to_unit_str(da.attrs),#type:ignore
            "### Description",
            da.attrs.get('description',"")
        ))),
        "\"\"\"",
    ))

def dc_to_str(dc:Dim_Coord) -> str:
    return '\n'.join((
        f"{dc}:Dim_Coord",
        "\"\"\"",
        indent('\n'.join((
            dc.attrs.get('long_name',dc),
            "### Units",
            attrs_to_unit_str(dc.attrs),
            "### Description",
            dc.attrs.get('description',"")
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