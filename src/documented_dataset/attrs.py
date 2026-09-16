from typing import Any, TypedDict, Unpack
import xarray as xr

class Attrs(TypedDict, total = False):
    units:str
    description:str
    long_name:str
    symbol:str
    fill_value:Any|None
class attrs(dict):
    def __init__(self,**kwargs: Unpack[Attrs]):
        super().__init__(**kwargs)
def repr(arr:xr.DataArray) -> str|None:
    attrs = arr.attrs
    if "symbol" in attrs:
        return attrs["symbol"]
    if "long_name" in attrs:
        return attrs["long_name"]
    return arr.name#type:ignore