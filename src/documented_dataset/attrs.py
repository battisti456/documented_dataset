from typing import Any, TypedDict, Unpack


class Attrs(TypedDict, total = False):
    units:str
    description:str
    long_name:str
    fill_value:Any|None
class attrs(dict):
    def __init__(self,**kwargs: Unpack[Attrs]):
        super().__init__(**kwargs)