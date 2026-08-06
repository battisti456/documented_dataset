import inspect
from collections.abc import Callable, Iterator
from importlib.util import module_from_spec, spec_from_file_location
from io import TextIOWrapper
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional, TypeVar

import xarray as xr


class documentedfunction:
    def __init__(self,func:Callable[['type[DocumentedDatasetType]'],xr.DataArray]):
        self.func = func
        self.name = self.func.__name__
        self._allowed_to_cache = True
    def __call__(self,cls:'type[DocumentedDatasetType]') -> xr.DataArray:
        if not self._allowed_to_cache or cls._no_caching:
            return self.func(cls)#type:ignore
        if self.name in cls._ds:
            return cls._ds[self.name]
        val = self.func(cls)#type:ignore
        cls._ds = cls._ds.assign({self.name: val})
        return val
    def __get__(self, obj, cls:'type[DocumentedDatasetType]') -> xr.DataArray:
        return self(cls)
    @classmethod
    def prevent_caching(cls):
        def wrapper(func:Callable[['type[DocumentedDatasetType]'],xr.DataArray]):
            to_return  = cls(func)
            to_return._allowed_to_cache = False
            return to_return
        return wrapper

def load_module(path:Path) -> ModuleType:
    spec = spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Documented_Dataset_Meta(type):
    def __getattr__(self, name: str) -> Any:
        try:
            attr =  object.__getattribute__(self,name)
            if isinstance(attr,documentedfunction):
                return attr(self)#type:ignore
            else:
                return attr
        except AttributeError:
            ...
        _ds:xr.Dataset = object.__getattribute__(self,"_ds")
        if name in _ds:
            return _ds[name]
        elif name in _ds.coords:
            return _ds.coords[name]
        else:
            raise AttributeError(f"Unknown attribute '{name}'.")

class NoCaching:
    def __init__(self,cls:'type[DocumentedDatasetType]'):
        self.dds = cls
        self.flag = False
    def __enter__(self):
        self.flag = True
        return self
    def __exit__(self, *exc):
        self.flag = False
        return self
    def __bool__(self):
        match(self.dds._cache_policy):
            case "never":
                return True
            case "permitted":
                return self.flag
            case "always":
                return False

class Documented_Dataset(metaclass = Documented_Dataset_Meta):
    _compile_documented:str = "_compile_documented"
    _calculated:str = "_calculated"
    _ds:xr.Dataset
    _documented_dimensions:dict[str,xr.DataArray]
    _documented_data:dict[str,xr.DataArray]
    _cache_policy:Literal["never","permitted","always"] = "permitted"
    _path:Path
    _no_caching:NoCaching
    def __init_subclass__(cls) -> None:
        cls._documented_dimensions = {}
        cls._documented_data = {}
        cls._path = Path(inspect.getfile(cls)).resolve()
        cls._no_caching = NoCaching(cls)
        calc_path = cls._path.with_name(f"{cls._path.stem}{cls._calculated}.py")
        if not calc_path.exists():
            return
        module = load_module(calc_path)
        for name, val in vars(module).items():
            if not isinstance(val,documentedfunction):
                continue
            setattr(cls,name,val)
    @classmethod
    def _load_path(cls,path:Optional[str|Path] = None):
        if path is None:
            path = cls._path.parent / cls._compile_documented
        else:
            path = Path(path)
        for path_ in path.glob("*.py"):
            load_module(path_)

    @classmethod
    def _build_dataset(cls):
        cls._ds = xr.Dataset()
        cls._ds = cls._ds.assign_coords(cls._documented_dimensions)
        cls._ds = cls._ds.assign(cls._documented_data)
    @classmethod
    def _build_documentation(cls,file:TextIOWrapper):
        file.write("import xarray as xr\n")
        file.write("from documented_dataset import Documented_Dataset\n")
        file.write(f"\n\nclass {cls.__name__}(Documented_Dataset):\n")
        for name, da in cls._ds.data_vars.items():
            file.write(f"    {name}:xr.DataArray\n    \"\"\"\n")
            if "long_name" in da.attrs:
                file.write(f"    {da.attrs['long_name']}\n")
            file.write("    ### Dimensions\n")
            for dim in da.dims:
                coord = cls._ds.coords[dim]
                file.write(f"    - {dim} : {coord.attrs.get('dtype',coord.dtype)}")
                if "units" in coord.attrs:
                    file.write(f" [{coord.attrs['units']}]")
                file.write("\n")
            file.write("    ### Units\n")
            file.write(f"    {da.attrs.get("dtype",da.dtype)}")
            if "units" in da.attrs:
                file.write(f" [{da.attrs['units']}]")
            file.write("\n    ### Description\n")
            if "description" in da.attrs:
                file.write(f"    {da.attrs['description']}")
            file.write("\n    \"\"\"\n")
        for name, da in cls._ds.coords.items():
            da:xr.DataArray
            file.write(f"    {name}:xr.DataArray\n    \"\"\"\n")
            file.write("    Coordinate.\n")
            if "long_name" in da.attrs:
                file.write(f"    {da.attrs['long_name']}\n")
            file.write("    ### Units\n")
            file.write(f"    {da.attrs.get("dtype",da.dtype)}")
            if "units" in da.attrs:
                file.write(f" [{da.attrs['units']}]")
            file.write("\n    ### Description\n")
            if "description" in da.attrs:
                file.write(f"    {da.attrs['description']}")
            file.write("\n    \"\"\"\n")
        for name, dp in cls._documented_properties():
            da = dp(cls)
            file.write(f"    {name}:xr.DataArray\n    \"\"\"\n")
            file.write("    ### Dimensions\n")
            for dim in da.dims:
                coord = cls._ds.coords[dim]
                file.write(f"    - {dim} : {coord.dtype}")
                if "units" in coord.attrs:
                    file.write(f" [{coord.attrs['units']}]")
                file.write("\n")
            file.write("    ### Units\n")
            if "units" in da.attrs:
                file.write(f"    {da.attrs['units']}")
            file.write("\n    ### Description\n")
            if "description" in da.attrs:
                file.write(f"    {da.attrs['description']}")
            file.write("\n    \"\"\"\n")
    @classmethod
    def _compile(cls,path:Optional[str|Path] = None):
        cls._load_path(path)
        cls._build_dataset()
        with open(cls._path.with_suffix(".pyi"),'w') as file:
            cls._build_documentation(file)
    @classmethod
    def _document_data(cls,**kwargs:xr.DataArray):
        cls._documented_data |= kwargs
    @classmethod
    def _document_dimensions(cls,**kwargs:xr.DataArray): 
        cls._documented_dimensions |= kwargs
    @classmethod
    def _documented_properties(cls) -> Iterator[tuple[str,documentedfunction]]:
        for name, value in vars(cls).items():
            if isinstance(value,documentedfunction):
                yield name, value
    @classmethod
    def _save(cls,path:str|Path):
        cls._ds.to_netcdf(path, engine="h5netcdf")
    @classmethod
    def _compile_and_save(cls,path:str|Path):
        with cls._no_caching:
            cls._compile()
        cls._save(path)
    @classmethod
    def _load(cls,path:str|Path):
        ds = xr.load_dataset(path)
        cls._ds = ds
    @classmethod
    def _is_initialized(cls):
        return hasattr(cls,"_ds")

DocumentedDatasetType = TypeVar("DocumentedDatasetType", bound=Documented_Dataset)