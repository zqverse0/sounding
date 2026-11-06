"""scout 的 `@scanner` 注册与属性标记的最小替身。

scout 用 `@scanner` 把扫描器登记进扫描作业（加载哪些内容、并发等）；单会话直接调用扫描函数，
不需要登记，所以装饰器原样返回工厂。标记属性名保留，被摘取的代码照常设置。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, TypeVar

from .result import Result

T = TypeVar("T", contravariant=True)

SCANNER_NAME_ATTR = "__scanner_name__"
SCANNER_CONTENT_ATTR = "__scanner_content__"
SCANNER_TIMELINE_NEEDED_ATTR = "__scanner_timeline_needed__"


class Scanner(Protocol[T]):
    def __call__(self, input: T, /) -> Awaitable[Result | list[Result]]: ...


F = TypeVar("F", bound=Callable[..., Any])


def scanner(**options: Any) -> Callable[[F], F]:
    """`@scanner(messages=..., events=...)`：单会话不登记，原样返回工厂。"""

    def decorate(factory: F) -> F:
        return factory

    return decorate


def mark_streaming_support(scan: Any, supported: bool) -> None:
    pass
