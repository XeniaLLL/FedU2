import inspect
import functools
from typing import Any, Callable, TypeVar, cast, overload
from collections import UserDict
from datetime import datetime

F = TypeVar('F', bound=Callable)


class Summary(UserDict[str, Any]):
    @overload
    def __init__(self): ...

    @overload
    def __init__(self, func: Callable, arguments: dict[str, Any]): ...

    def __init__(self, func=None, arguments=None):
        super().__init__(arguments)
        self.func = cast(Callable, func)
        self.start_time = datetime.now()

    @property
    def arguments(self) -> dict[str, Any]:
        return self.data

    def __str__(self) -> str:
        timestamp = self.start_time.strftime('%m-%d %H:%M:%S')
        sb = [
            f'[{self.func.__module__}.{self.func.__qualname__}] {timestamp}',
            *(f'{name}: {value!r}' for name, value in self.arguments.items())
        ]
        return '\n'.join(sb)

    @staticmethod
    def record(func: F) -> F:
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            summary_param = next((
                name for name, param in inspect.signature(func).parameters.items()
                if isinstance(param.default, Summary)
            ), None)
            if summary_param is not None:
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                arguments = {
                    name: value for name, value in bound_args.arguments.items()
                    if name != summary_param
                }
                return func(*args, **kwargs, **{summary_param: Summary(func, arguments)})
            else:
                return func(*args, **kwargs)
    
        return cast(F, func_wrapper)
