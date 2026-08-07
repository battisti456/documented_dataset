import inspect
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from io import TextIOWrapper
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast

import xarray as xr
from xarray.core.coordinates import DatasetCoordinates

from .caching_manager import Caching_Manager

if TYPE_CHECKING:
    from .registry import Bulk_Provider, Registry, RegistryComponent, Single_Provider
    from .types import DocumentedDatasetType


def load_module(path:Path) -> ModuleType:
    spec = spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DocumentedDatasetAttributeError(AttributeError): ...
class DocumentedDatasetLoopingAttributeError(Exception): ...
class DocumentedDatasetMismatchedGroupAttributes(Exception): ...

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
        elif name in self._registry.derived:
            val = self._get_variable_from_registry(name,self._registry.derived,self._ds)
            return val
        else:
            raise DocumentedDatasetAttributeError(name=name, obj = self)


class Documented_Dataset(metaclass = Documented_Dataset_Meta):
    _autoload_dir:Path
    _ds:xr.Dataset
    _registry:'Registry[Self]'
    _no_caching:'Caching_Manager[Self]'
    _cache_policy:Literal['never','always','permitted'] = 'permitted'
    def __init_subclass__(
            cls, *, 
            registry:'Registry[Self]|None' = None, 
            cache_policy:Literal['never','always','permitted'] = "permitted"
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
        cls._cache_policy = cache_policy
        cls._no_caching = Caching_Manager(cls)#type:ignore
    @staticmethod
    def _validate_bulk_return(ret:dict[str,xr.DataArray],provider:'Bulk_Provider'):
        if set(ret.keys()) != set(provider.included_variables):
            raise DocumentedDatasetMismatchedGroupAttributes(f"'{provider.func.__name__} encountered a mismatch between the return dictionary keys {set(ret.keys())} and the declared variables {set(provider.included_variables)}.")
    @classmethod
    def _get_variable_from_registry(cls,name:str,registry_component:'RegistryComponent[Self]',ds:xr.Dataset) -> xr.DataArray:
        provider = registry_component[name]
        if provider.IS_BULK:
            provider = cast('Bulk_Provider[Self]',provider)
            ret = provider.func(cls)#type:ignore
            cls._validate_bulk_return(ret,provider)#type:ignore
            if not cls._no_caching:
                if isinstance(provider.cache_policy,dict):
                    for iname, val in ret.items():
                        cache_policy = provider.cache_policy.get(iname)
                        if cache_policy or (cache_policy is None and registry_component.default_cache_policy):
                            ds[iname] = val
                if provider.cache_policy or (provider.cache_policy is None and registry_component.default_cache_policy):
                    for iname, val in ret.items():
                        ds[iname] = val
            return ret[name]
        else:
            provider = cast('Single_Provider[Self]',provider)
            val = provider.func(cls)#type:ignore
            if not cls._no_caching and (provider.cache_policy or provider.cache_policy is None and registry_component.default_cache_policy):
                ds[name] = val
            return val

    @classmethod
    def _load_path(cls,path:Path):
        if not path.exists():
            return
        for path_ in path.glob("*.py"):
            load_module(path_)

    @classmethod
    def _load_registry_component(cls,registry_component:'RegistryComponent[Self]',ds:xr.Dataset|DatasetCoordinates) -> dict[str,xr.DataArray]:
        to_return = {}
        providers = list(registry_component.registered.values())
        last_missing_attrs = None
        missing_attrs = set()
        while providers:
            next_providers = []
            missing_attrs = set()
            for provider in providers:
                try:
                    if provider.IS_BULK:
                        provider = cast('Bulk_Provider[Self]',provider)
                        ret = provider.func(cls)#type:ignore
                        cls._validate_bulk_return(ret,provider)#type:ignore
                        for name, val in ret.items():
                            ds[name] = val
                    else:
                        provider = cast('Single_Provider[Self]',provider)
                        val = provider.func(cls)#type:ignore
                        ds[provider.func.__name__] = val
                except DocumentedDatasetAttributeError as err:
                    missing_attrs.add(err.name)
                    next_providers.append(provider)
            providers = next_providers
            if missing_attrs == last_missing_attrs:
                break
            else:
                last_missing_attrs = missing_attrs
        if providers:
            raise DocumentedDatasetLoopingAttributeError(f"Encountered a looping set of not yet seen attributes while loading: {missing_attrs}")
        return to_return

    @classmethod
    def _build_dataset(cls):
        cls._ds = xr.Dataset()
        cls._load_registry_component(cls._registry.coordinates,cls._ds.coords)
        cls._load_registry_component(cls._registry.source,cls._ds)

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
        with cls._no_caching:
            for name, dp in cls._registry.derived.items():
                da = dp(cls)#type:ignore
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
    def compile(cls):
        cls._build_dataset()
        with open(cls._path.with_suffix(".pyi"),'w') as file:
            cls._build_documentation(file)
    @classmethod
    def _save(cls,path:str|Path):
        cls._ds.to_netcdf(path, engine="h5netcdf")
    @classmethod
    def compile_and_save(cls,path:str|Path):
        with cls._no_caching:
            cls.compile()
        cls._save(path)
    @classmethod
    def load(cls,path:str|Path):
        ds = xr.load_dataset(path)
        cls._ds = ds
    @classmethod
    def _is_initialized(cls):
        return hasattr(cls,"_ds")

DocumentedDatasetType = TypeVar("DocumentedDatasetType", bound=Documented_Dataset)