from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Generic, overload

from .types import (
    DocumentedBulkFunction,
    DocumentedDatasetType,
    DocumentedFunction,
)


@dataclass
class Single_Provider(Generic[DocumentedDatasetType]):  # noqa: UP046
    func: DocumentedFunction[DocumentedDatasetType]
    cache_policy:bool|None
    IS_BULK:ClassVar[bool] = False

@dataclass
class Bulk_Provider(Generic[DocumentedDatasetType]):  # noqa: UP046
    func: DocumentedBulkFunction[DocumentedDatasetType]
    cache_policy:bool|None|dict[str,bool|None]
    included_variables:tuple[str,...]
    IS_BULK:ClassVar[bool] = True

type Provider[T] = Single_Provider[T]|Bulk_Provider[T]# type:ignore

class RegistryComponent(Generic[DocumentedDatasetType]):  # noqa: UP046
    def __init__(self, default_cache_policy:bool = False):
        super().__init__()
        self.default_cache_policy = default_cache_policy
        self.registered:dict[str,Provider[DocumentedDatasetType]] = {}
    @overload
    def __call__(
        self,
        func:DocumentedFunction[DocumentedDatasetType],
        *,
        cache:bool|None
    )-> DocumentedFunction[DocumentedDatasetType]: ...
    @overload
    def __call__(self,
        func:None, 
        *,
        cache:bool|None
    ) -> Callable[
        [DocumentedFunction[DocumentedDatasetType]],
        DocumentedFunction[DocumentedDatasetType]
    ]: ...
    def __call__(#type:ignore
            self,
            func:DocumentedFunction[DocumentedDatasetType]|None = None,
            *,
            cache:bool|None = None
        ) -> DocumentedFunction[DocumentedDatasetType]|Callable[
            [DocumentedFunction[DocumentedDatasetType]],
            DocumentedFunction[DocumentedDatasetType]
        ]:
        if func is not None:
            self.registered[func.__name__] = Single_Provider(func,cache)
            return func
        def flagged_call(func:DocumentedFunction[DocumentedDatasetType]) -> DocumentedFunction[DocumentedDatasetType]:
            self(func,cache=cache)
            return func
        return flagged_call
    def outputs(self,*args:str,cache:bool|None|dict[str,bool|None] = None) -> Callable[[DocumentedBulkFunction[DocumentedDatasetType]],DocumentedBulkFunction[DocumentedDatasetType]]:
        if isinstance(cache,dict):
            args += tuple(cache.keys())
        else:
            ...
        def bulk_call(func:DocumentedBulkFunction[DocumentedDatasetType]) -> DocumentedBulkFunction[DocumentedDatasetType]:
            for name in args:
                self.registered[name] = Bulk_Provider(func,cache,args)
            return func
        return bulk_call
    def items(self):
        yield from self.registered.items()
    def values(self):
        yield from self.registered.values()
    def __contains__(self, item):
        return self.registered.__contains__(item)
    def __getitem__(self, key):
        return self.registered.__getitem__(key)


class Registry(Generic[DocumentedDatasetType]):  # noqa: UP046
    def __init__(self):
        self.coordinates:RegistryComponent[DocumentedDatasetType] = RegistryComponent()
        self.source:RegistryComponent[DocumentedDatasetType] = RegistryComponent()
        self.derived:RegistryComponent[DocumentedDatasetType] = RegistryComponent(default_cache_policy=False)

