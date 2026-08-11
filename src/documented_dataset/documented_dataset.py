import inspect
from collections.abc import Sequence
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from io import TextIOWrapper
from pathlib import Path
from types import ModuleType
from typing import Any, Self, TypeVar

import xarray as xr
from xarray.core.coordinates import DatasetCoordinates

from ._dims import _Dims
from .exceptions import (
    DocumentedDatasetAttributeError,
    DocumentedDatasetDimensionError,
    DocumentedDatasetMismatchedGroupAttributes,
)
from .registry import Bulk_Provider, Provider, Registry
from .types import DocumentedDatasetType


def load_module(path:Path) -> ModuleType:
    spec = spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Documented_Dataset_Meta(type):
    def __getattr__(self:type['DocumentedDatasetType'], name: str) -> Any: # type: ignore
        try:
            return  object.__getattribute__(self,name)
        except AttributeError:
            ...
        _ds:xr.Dataset = object.__getattribute__(self,"_ds")
        if name in _ds:
            return _ds[name]
        elif name in _ds.coords:
            return _ds.coords[name]
        else:
            val = self._get_with_cache(name,self._ds)
            if val is None:
                raise DocumentedDatasetAttributeError(name=name, obj = self)
            return val


class Documented_Dataset(metaclass = Documented_Dataset_Meta):
    _autoload_dir:Path
    _ds:xr.Dataset
    _registry:'Registry[Self]'
    _default_cache_policy:bool
    _default_storage_level:int
    _currently_loading:set[Provider[Self]]
    _is_storing:None|int
    dims:_Dims
    def __init_subclass__(
            cls, *, 
            registry:'Registry[Self]|None' = None, 
            default_cache_policy:bool = False,
            default_storage_level:int = 0
        ) -> None:
        path = Path(inspect.getfile(cls)).resolve().parent
        if not hasattr(cls,'_autoload_dir'):
            cls._autoload_dir = path / "autoload"
        if registry is None:
            mod = import_module("documented_dataset.registry")
            cls._registry = mod.Registry()
        else:
            cls._registry = registry
        if cls._autoload_dir.exists():
            cls._load_path(cls._autoload_dir)
        cls._default_cache_policy = default_cache_policy
        cls._default_storage_level = default_storage_level
        cls._is_storing = None
        cls.dims = _Dims()
    @staticmethod
    def _validate_bulk_return(ret:dict[str,xr.DataArray],provider:'Bulk_Provider'):
        if set(ret.keys()) != set(provider.included_variables):
            raise DocumentedDatasetMismatchedGroupAttributes(
                f"'{provider.func.__name__} encountered a mismatch between the return dictionary keys "
                f" {set(ret.keys())} and the declared variables {set(provider.included_variables)}.")
    @classmethod
    def _get_with_cache(cls,name:str,ds:xr.Dataset) -> xr.DataArray|None:
        if name in ds:
            return ds[name]
        try:
            provider = cls._registry[name]
        except AttributeError:
            return None
        ret = provider(cls)#type:ignore
        should_cache = cls._eval_cache(provider)
        if cls._is_storing is not None:
            should_store = cls._eval_do_store(provider,cls._is_storing)
            for iname in should_store:
                should_cache[iname] = True
        for iname, val in ret.items():
            if should_cache[iname]:
                ds[iname] = val
        return ret[name]
    @classmethod
    def _load_path(cls,path:Path):
        if not path.exists():
            return
        for path_ in path.glob("*.py"):
            load_module(path_)

    @classmethod
    def _load_providers(
        cls,
        providers:list[Provider[Self]],
        ds:xr.Dataset|DatasetCoordinates,
        storage_threshold:int = -1
    ):
        for provider in providers:
            vars_to_store = cls._eval_do_store(provider,storage_threshold)
            if not vars_to_store or all(name in ds for name in vars_to_store):
                continue
            ret = provider(cls)
            cls._validate_bulk_return(ret,provider)#type:ignore
            for name in vars_to_store:
                ds[name] = ret[name]

    @classmethod
    def _build_dataset(cls, storage_threshold:int):
        cls._ds = xr.Dataset()
        try:
            cls._is_storing = storage_threshold
            cls._load_providers(list(cls._registry._dim_coords()), cls._ds.coords)
            for dim in cls._ds.dims:
                assert isinstance(dim,str)
                setattr(cls.dims,dim,dim)
            cls._load_providers(list(cls._registry._non_dim_coords()), cls._ds, storage_threshold=storage_threshold)
        finally:
            cls._is_storing = None

    @classmethod
    def _build_documentation(cls,file:TextIOWrapper):
        file.write(
            "from typing import Any, Literal"\
            "from collections.abc import Sequence"\
            ""\
            "import xarray as xr" \
            "from documented_dataset import Documented_Dataset"\
            f"class {cls.__name__}(Documented_Dataset):"\
            "   class dims:"\
        )
        file.writelines(f"        {name}:Literal[\"{name}\"]\n" for name in cls.dims._names())
        file.write(
            "   @classmethod"\
            "   def array(" \
            "       cls, " \
            "       data:Any, "
            "       *, " \
            "       dims:Sequence[str], " \
            "       attrs:dict[str,str] = ...," \
        )
        file.writelines(f"        {name}:xr.DataArray = ..." for name in cls.dims._names())
        file.write("    ) -> xr.DataArray: ...") 
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
        for provider in cls._registry._non_dim_coords():
            if all(name in cls._ds for name in provider.included_variables):
                get_from = cls._ds
            else:
                get_from = provider(cls)
            for name in provider.included_variables:
                da:xr.DataArray = get_from[name]
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
    @classmethod
    def compile(cls, storage_level = 0):
        cls._build_dataset(storage_level)
        with open(Path(inspect.getfile(cls)).resolve().with_suffix(".pyi"),'w') as file:
            cls._build_documentation(file)
    @classmethod
    def _save(cls,path:str|Path, storage_threshold = 0):
        do_not_save:list[str] = []
        for provider in cls._registry.values():
            drop_vars = cls._eval_do_not_store(provider,storage_threshold)
            do_not_save += [var for var in drop_vars if var in cls._ds]
        cls._ds.drop_vars(do_not_save).to_netcdf(path, engine="h5netcdf")
    @classmethod
    def compile_and_save(cls,path:str|Path, storage_level = 0):
        cls.compile(storage_level)
        cls._save(path)
    @classmethod
    def load(cls,path:str|Path):
        ds = xr.load_dataset(path)
        cls._ds = ds
        for dim in cls._ds.dims:
            assert isinstance(dim,str)
            setattr(cls.dims,dim,dim)
    @classmethod
    def _is_initialized(cls):
        return hasattr(cls,"_ds")
    @classmethod
    def _eval_cache(cls,provider:Provider[Self]) -> dict[str,bool]:
        if isinstance(provider.cache_policy,dict):
            to_return:dict[str,bool] = {}
            for name in provider.included_variables:
                if name not in provider.cache_policy or provider.cache_policy[name] is None:
                    policy = cls._default_cache_policy
                else:
                    policy = provider.cache_policy[name]
                assert policy is not None
                to_return[name] = policy
            return to_return
        else:
            policy = cls._default_cache_policy if provider.cache_policy is None else provider.cache_policy
            return {name:policy for name in provider.included_variables}
    @classmethod
    def _eval_storage_level(cls,provider:Provider[Self]) -> dict[str,int]:
        if isinstance(provider.storage_level,dict):
            to_return:dict[str,int] = {}
            for name in provider.included_variables:
                if name not in provider.storage_level or provider.storage_level[name] is None:
                    storage_level = cls._default_storage_level
                else:
                    storage_level = provider.storage_level[name]
                assert storage_level is not None
                to_return[name] = storage_level
            return to_return
        else:
            storage_level = cls._default_storage_level if provider.storage_level is None else provider.storage_level
            return {name:storage_level for name in provider.included_variables}
    @classmethod
    def _eval_store(cls,provider:Provider[Self],storage_threshold:int) -> dict[str,bool]:
        return {name:storage_level <= storage_threshold for name,storage_level in cls._eval_storage_level(provider).items()}
    @classmethod
    def _eval_do_not_store(cls,provider:Provider[Self],storage_threshold:int) -> list[str]:
        return [name for name,storage_level in cls._eval_storage_level(provider).items() if storage_level > storage_threshold]
    @classmethod
    def _eval_do_store(cls,provider:Provider[Self],storage_threshold:int) -> list[str]:
        return [name for name,storage_level in cls._eval_storage_level(provider).items() if storage_level <= storage_threshold]
    @classmethod
    def array(cls,data:Any,*,dims:Sequence[str],attrs:dict[str,str]|None = None,**kwargs:Any):
        if any(coord not in dims for coord in kwargs):
            raise DocumentedDatasetDimensionError(
                "All assignments must be specifically stated in dims."
            )
        coords = {
            dim: cls._ds.coords[dim]
            for dim in dims
            if dim in cls._ds.coords
        }
        coords.update(kwargs)
        return xr.DataArray(
            data,
            dims = dims,
            coords = coords,
            attrs={} if attrs is None else attrs
        )

DocumentedDatasetType = TypeVar("DocumentedDatasetType", bound=Documented_Dataset)  # noqa: F811
