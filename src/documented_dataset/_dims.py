from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .documented_dataset import Documented_Dataset

class _Dims:
    def __init__(self,cls:'type[Documented_Dataset]'):
        self._cls = cls
    def _names(self):
        return [name for name in dir(self) if name[0] != '_']
    def __getattr__(self, name):
        try:
            return object.__getattribute__(self,name)
        except AttributeError:
            #allow coordinate initializations with non-existent dimensions
            if any(prov.dim_coord and prov._is_running for prov in self._cls._registry.registered.values()):
                return name
            else:
                raise