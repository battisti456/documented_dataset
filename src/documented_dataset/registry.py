from collections.abc import Callable
from typing import Generic, overload

from .types import DocumentedBulkFunction, DocumentedDatasetType, DocumentedFunction


class RegistryComponent(Generic[DocumentedDatasetType]):  # noqa: UP046
    def __init__(self):
        self.registered:list[DocumentedBulkFunction[DocumentedDatasetType]] = []
    def __call__(
            self,
            func:DocumentedBulkFunction[DocumentedDatasetType], 
        ) -> DocumentedBulkFunction[DocumentedDatasetType]:
        self.registered.append(func)
        return func
    def __contains__(self, item:str):
        return self.registered.__contains__(item)

class Cache_Enabled_Registry_Component(Generic[DocumentedDatasetType]):  # noqa: UP046
    def __init__(self, default_cache_policy:bool = False):
        super().__init__()
        self.default_cache_policy = default_cache_policy
        self.registered:dict[str,DocumentedFunction[DocumentedDatasetType]] = {}
        self.cache_policy:dict[str,bool|None] = {}
    @overload
    def __call__(
        self,
        func:DocumentedFunction[DocumentedDatasetType], 
        cache:bool|None
    )-> DocumentedFunction[DocumentedDatasetType]: ...
    @overload
    def __call__(self,
        func:None, 
        cache:bool|None
    ) -> Callable[
        [DocumentedFunction[DocumentedDatasetType]],
        DocumentedFunction[DocumentedDatasetType]
    ]: ...
    def __call__(#type:ignore
            self,
            func:DocumentedFunction[DocumentedDatasetType]|None = None, 
            cache:bool|None = None
        ) -> DocumentedFunction[DocumentedDatasetType]|Callable[
            [DocumentedFunction[DocumentedDatasetType]],
            DocumentedFunction[DocumentedDatasetType]
        ]:
        if func is not None:
            self.registered[func.__name__] = func
            self.cache_policy[func.__name__] = cache
            return func
        def flagged_call(func:DocumentedFunction[DocumentedDatasetType]) -> DocumentedFunction[DocumentedDatasetType]:
            self(func,cache)
            return func
        return flagged_call
    def should_cache(self,data_variable_name:str) -> bool:
        cache_policy = self.cache_policy[data_variable_name]
        return cache_policy if cache_policy is not None else self.default_cache_policy
    def items(self):
        yield from self.registered.items()
    def __contains__(self, item):
        return self.registered.__contains__(item)

class Registry(Generic[DocumentedDatasetType]):  # noqa: UP046
    def __init__(self):
        self.coordinates:RegistryComponent[DocumentedDatasetType] = RegistryComponent()
        self.source:RegistryComponent[DocumentedDatasetType] = RegistryComponent()
        self.derived:Cache_Enabled_Registry_Component[DocumentedDatasetType] = Cache_Enabled_Registry_Component(default_cache_policy=False)

