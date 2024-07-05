"""Defines syntax sugar utilities."""

from typing import Callable, TypeVar, cast, overload

T = TypeVar("T")
Ti = TypeVar("Ti")


@overload
def default(value: Ti | None, if_none: T | Callable[[], T], fn: Callable[[Ti], T]) -> T: ...


@overload
def default(value: T | None, if_none: T | Callable[[], T]) -> T: ...


def default(value: T | Ti | None, if_none: T | Callable[[], T], fn: Callable[[Ti], T] | None = None) -> T:
    if value is None:
        return if_none() if callable(if_none) else if_none
    if fn is None:
        return cast(T, value)
    return fn(cast(Ti, value))
