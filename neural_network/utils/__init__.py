'''
Contains utility functions, classes and decorators used in this project

Most of these are used internally by the API, hence they do not reside
in the package namespace, they need to be explicitly imported using
the full qualified name of the function or class

Unlike the rest of the API, these are listed here in order of usefulness
not alphabetically

In module typesafety
    type_safe(func: Callable = None, *,
              skip: Iterable = SENTINEL_TUPLE) -> Callable
        A decorator to validate parameter types at runtime

    not_none(nullable: Union[Callable, Iterable] = SENTINEL_TUPLE, *,
             follow_wrapped: Union[bool, int] = True) -> Callable
        A decorator to ensure paramters are not None at runtime

In module exports
    export(fn_cls: Union[Callable, Type]) -> Callable
        A decorator to mark a function or class as public and add it to
        the __name_to_symbol_map__

In module timeit
    class timeit
        Provides methods to track the time of execution of a function
        or, a snippet using a context manager

In module functools
    unwrap(func: Callable, depth: Union[bool, int] = True) -> Callable
    class MethodInvalidator
'''
