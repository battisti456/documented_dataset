import inspect
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from io import TextIOWrapper
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import xarray as xr

from .caching_manager import Caching_Manager

if TYPE_CHECKING:
    from .registry import Registry, RegistryComponent
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


class Documented_Dataset_Meta(type, Generic['DocumentedDatasetType']):
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
            val = self._registry.derived[name](self)#type:ignore
            if self._no_caching or not self._registry.derived.should_cache(name):
                return val
            self._ds = _ds.assign_attrs({name: val})
            return val
        else:
            raise DocumentedDatasetAttributeError(name=name, obj = self)


class Documented_Dataset(Generic['DocumentedDatasetType'],metaclass = Documented_Dataset_Meta):
    _compile_dir:Path
    _derived_dir:Path
    _ds:xr.Dataset
    _registry:'Registry[DocumentedDatasetType]'
    _no_caching:'Caching_Manager[DocumentedDatasetType]'
    _cache_policy:Literal['never','always','permitted'] = 'permitted'
    def __init_subclass__(
            cls, *, 
            registry:'Registry[DocumentedDatasetType]|None' = None, 
            cache_policy:Literal['never','always','permitted'] = "permitted"
        ) -> None:
        path = Path(inspect.getfile(cls)).resolve().parent
        if not hasattr(cls,'_compile_dir'):
            cls._compile_dir = path / "compile_dir"
        if not hasattr(cls,'_derived_dir'):
            cls._derived_dir = path / "derived_dir"

        if registry is None:
            mod = import_module("documented_dataset.registry")
            cls._registry = mod.Registry()
        else:
            cls._registry = registry

        cls._load_path(cls._derived_dir)
        cls._cache_policy = cache_policy
        cls._no_caching = Caching_Manager(cls)#type:ignore
    
    @classmethod
    def _load_path(cls,path:Path):
        if not path.exists():
            return
        for path_ in path.glob("*.py"):
            load_module(path_)

    @classmethod
    def _load_registry_component(cls,registry_component:'RegistryComponent[DocumentedDatasetType]',is_coord = False) -> dict[str,xr.DataArray]:
        to_return = {}
        funcs = registry_component.registered.copy()
        last_missing_attrs = None
        missing_attrs = set()
        while funcs:
            next_funcs = []
            missing_attrs = set()
            for func in funcs:
                try:
                    ret = func(cls)#type:ignore
                    if isinstance(ret,dict):
                        if is_coord:
                            cls._ds = cls._ds.assign_coords(ret)
                        else:
                            cls._ds = cls._ds.assign(ret)
                    else:
                        if is_coord:
                            cls._ds = cls._ds.assign_coords({func.__name__ : ret})
                        else:
                            cls._ds = cls._ds.assign({func.__name__ : ret})
                except DocumentedDatasetAttributeError as err:
                    missing_attrs.add(err.name)
                    next_funcs.append(func)
            funcs = next_funcs
            if missing_attrs == last_missing_attrs:
                break
            else:
                last_missing_attrs = missing_attrs
        if funcs:
            raise DocumentedDatasetLoopingAttributeError(f"Encountered a looping set of not yet seen attributes while loading {'coordinates' if is_coord else 'attributes'}: {missing_attrs}")
        return to_return

    @classmethod
    def _build_dataset(cls):
        cls._ds = xr.Dataset()
        cls._load_registry_component(cls._registry.coordinates, is_coord=True)
        cls._load_registry_component(cls._registry.source,is_coord=False)

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
        cls._load_path(cls._compile_dir)
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