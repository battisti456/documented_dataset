from collections.abc import Callable
from typing import Generic, overload

from .provider import Bulk_Provider, Provider, Single_Provider
from .types import (
    DocumentedBulkFunction,
    DocumentedDatasetType,
    DocumentedFunction,
)


class Registry(Generic[DocumentedDatasetType]):  # noqa: UP046
    def __init__(self):
        super().__init__()
        self.registered:dict[str,Provider[DocumentedDatasetType]] = {}
    @overload
    def __call__(self,
        func:None = None, 
        *,
        storage_level:int = 0,
        cache:bool|None = None
    ) -> Callable[
        [DocumentedFunction[DocumentedDatasetType]],
        DocumentedFunction[DocumentedDatasetType]
    ]: ...
    @overload
    def __call__(
        self,
        func:str,
        *args:str,
        storage_level:int|None|dict[str,int|None] = None,
        cache:bool|None|dict[str,bool|None] = None
    ) -> Callable[
        [DocumentedBulkFunction[DocumentedDatasetType]],
        DocumentedBulkFunction[DocumentedDatasetType]
    ]: ...
    @overload
    def __call__(
        self,
        func:DocumentedFunction[DocumentedDatasetType],
        *,
        storage_level:int|None = None,
        cache:bool|None = None
    )-> DocumentedFunction[DocumentedDatasetType]: ...
    def __call__(#type:ignore
        self,
        func:DocumentedFunction[DocumentedDatasetType]|None|str = None,
        *args:str,
        storage_level:int|None|dict[str,int|None] = None,
        cache:bool|None|dict[str,bool|None] = None
    ) -> DocumentedFunction[DocumentedDatasetType]|Callable[
        [DocumentedFunction[DocumentedDatasetType]],
        DocumentedFunction[DocumentedDatasetType]
    ]|Callable[
        [DocumentedBulkFunction[DocumentedDatasetType]],
        DocumentedBulkFunction[DocumentedDatasetType]
    ]:
        if func is None:# flagging call, single function with set values
            assert not isinstance(storage_level,dict)
            assert not isinstance(cache,dict)
            def flagged_call(func:DocumentedFunction[DocumentedDatasetType]) -> DocumentedFunction[DocumentedDatasetType]:
                self.__call__(func,storage_level=storage_level,cache=cache)
                return func
            return flagged_call
        elif isinstance(func,str):#bulk function
            args = (func,) + args
            if isinstance(cache,dict):
                args += tuple(cache.keys())
            if isinstance(storage_level,dict):
                args += tuple(storage_level.keys())
            args = tuple(set(args))
            def bulk_call(func:DocumentedBulkFunction[DocumentedDatasetType]) -> DocumentedBulkFunction[DocumentedDatasetType]:
                provider= Bulk_Provider(func,storage_level,cache,args)
                for name in args:
                    self.registered[name] = provider
                return func
            return bulk_call
        else:#default behavior, single function, no flags
            assert not isinstance(storage_level,dict)
            assert not isinstance(cache,dict)
            self.registered[func.__name__] = Single_Provider(func,storage_level,cache,(func.__name__,))
            return func
    def items(self):
        yield from self.registered.items()
    def values(self):
        yield from self.registered.values()
    def __contains__(self, item):
        return self.registered.__contains__(item)
    def __getitem__(self, key):
        return self.registered.__getitem__(key)




