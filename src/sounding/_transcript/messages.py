# 摘自 inspect_scout 7521262 src/inspect_scout/_transcript/messages.py（MIT）：只保留消息分段。
# 去掉：事件 / 时间线来源、流式分段、scorers 过滤。保留部分逐字未改。
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import anyio

from .._ai.model import ChatMessage, Model, get_model, get_model_info
from .._ai.util import tg_collect
from .._scanner.extract import MessagesAsStr

if TYPE_CHECKING:
    from .types import Transcript

DEFAULT_CONTEXT_WINDOW = 128_000
_TOKENIZER_SAFETY_MARGIN = 0.05

# ``count_tokens`` can be a network call (e.g. Anthropic's count_tokens
# endpoint), so counting each message individually floods the connection
# pool and rate limits on large transcripts. Messages are instead grouped
# into chunks targeting a small fraction of the segment budget (sized via
# a rough chars-per-token estimate), each chunk is counted with a single
# call, and only a handful of calls are in flight at once. Segments are
# then packed from whole chunks, trading a small amount of packing
# headroom (at most one chunk's worth per segment) for far fewer calls.
_COUNT_CHUNK_BUDGET_FRACTION = 0.05
_COUNT_EST_CHARS_PER_TOKEN = 4
_COUNT_MAX_CONCURRENCY = 8


def _effective_segment_budget(
    *,
    model: Model,
    context_window: int | None,
    prompt_reserve: int | float = 0.2,
) -> int:
    """Compute the token budget available for rendered messages.

    ``prompt_reserve`` controls how much of the context window is reserved
    for prompt scaffolding wrapped around the messages (e.g. a scanner
    template). A ``float`` reserves that fraction of the window; an
    ``int`` reserves that many tokens and additionally subtracts a small
    safety margin to absorb ``count_tokens`` imprecision.
    """
    if context_window is not None:
        window = context_window
    else:
        model_info = get_model_info(model)
        window = (
            model_info.input_tokens
            if model_info is not None and model_info.input_tokens is not None
            else DEFAULT_CONTEXT_WINDOW
        )

    if isinstance(prompt_reserve, float):
        return int(window * (1.0 - prompt_reserve))
    return int(window * (1.0 - _TOKENIZER_SAFETY_MARGIN)) - prompt_reserve


@dataclass(frozen=True)
class MessagesSegment:
    """A segment of rendered messages that fits within a token budget.

    Attributes:
        messages: The original ChatMessage objects in this segment.
        messages_str: Pre-rendered string from messages_as_str.
        segment: 0-based segment index, auto-increments across yields.
    """

    messages: list[ChatMessage]
    messages_str: str
    segment: int


async def segment_messages(
    source: list[ChatMessage],
    *,
    messages_as_str: MessagesAsStr,
    model: Model | str | None = None,
    context_window: int | None = None,
    compaction: Literal["all", "last"] | int = "all",
    prompt_reserve: int | float = 0.2,
) -> AsyncIterator[MessagesSegment]:
    """Render messages and split them into segments that fit within a token budget.

    Renders each message individually via ``messages_as_str``, groups
    the rendered messages into chunks and counts tokens per chunk (one
    ``count_tokens`` call each, with bounded concurrency), then packs
    chunks into segments that fit within the effective budget (context
    window minus the portion reserved by ``prompt_reserve``).

    When given events or a ``TimelineSpan``, delegates to
    ``span_messages()`` to extract and merge messages (handling
    compaction boundaries), then segments the result.

    Args:
        source: A list of ChatMessage, a list of Event, or a
            TimelineSpan. Events and spans are processed via
            ``span_messages()`` first.
        messages_as_str: Rendering function from ``message_numbering()``.
            Must be called sequentially to preserve counter ordering.
        model: Model used for token counting.
        context_window: Override for context window size. If None,
            looked up via ``get_model_info(model)``.
        compaction: How to handle compaction boundaries when source
            contains events. Passed through to ``span_messages()``.
        prompt_reserve: Context-window allowance for prompt scaffolding
            wrapped around the rendered messages (e.g. a scanner
            template). A ``float`` reserves that fraction of the window;
            an ``int`` reserves that many tokens and additionally
            subtracts a small safety margin for ``count_tokens``
            imprecision. Default ``0.2`` leaves 80% of the window for
            messages.

    Yields:
        MessagesSegment instances, each fitting within the token budget.
        Segment counter increments across all yields.
    """
    # Resolve model
    model = get_model(model)

    # sounding：只有消息来源（输入是 OpenAI chat 消息，没有事件 / 时间线）。
    messages = list(source)

    if not messages:
        return

    # Compute effective budget
    effective_budget = max(
        1,
        _effective_segment_budget(
            model=model,
            context_window=context_window,
            prompt_reserve=prompt_reserve,
        ),
    )

    # Pass 1: Render each message sequentially (counter ordering matters)
    rendered: list[tuple[ChatMessage, str]] = []
    for msg in messages:
        text = await messages_as_str([msg])
        if text:  # Skip empty renders (e.g. filtered system messages)
            rendered.append((msg, text))

    if not rendered:
        return

    # Pass 2: Group messages into chunks and count tokens per chunk
    # (see _COUNT_* constants above for why counting is chunked).
    chunk_char_target = max(
        1,
        int(
            effective_budget * _COUNT_CHUNK_BUDGET_FRACTION * _COUNT_EST_CHARS_PER_TOKEN
        ),
    )
    chunks: list[list[tuple[ChatMessage, str]]] = [[]]
    chunk_chars = 0
    for msg, text in rendered:
        if chunks[-1] and chunk_chars + len(text) > chunk_char_target:
            chunks.append([])
            chunk_chars = 0
        chunks[-1].append((msg, text))
        chunk_chars += len(text)

    count_limiter = anyio.Semaphore(_COUNT_MAX_CONCURRENCY)

    async def count_chunk(text: str) -> int:
        async with count_limiter:
            return await model.count_tokens(text)

    chunk_texts = ["\n".join(text for _, text in chunk) for chunk in chunks]
    token_counts = await tg_collect(
        [lambda t=text: count_chunk(t) for text in chunk_texts]  # type: ignore[misc]
    )

    # Pass 3: Segment based on accumulated chunk token counts
    segment_counter = 0
    current_messages: list[ChatMessage] = []
    current_texts: list[str] = []
    running_tokens = 0

    for chunk, tokens in zip(chunks, token_counts, strict=True):
        if current_messages and running_tokens + tokens > effective_budget:
            yield MessagesSegment(
                messages=current_messages,
                messages_str="\n".join(current_texts),
                segment=segment_counter,
            )
            segment_counter += 1
            current_messages = []
            current_texts = []
            running_tokens = 0

        current_messages.extend(msg for msg, _ in chunk)
        current_texts.extend(text for _, text in chunk)
        running_tokens += tokens

    # Yield remaining segment
    if current_messages:
        yield MessagesSegment(
            messages=current_messages,
            messages_str="\n".join(current_texts),
            segment=segment_counter,
        )


async def transcript_messages(
    transcript: "Transcript",
    *,
    messages_as_str: MessagesAsStr,
    model: Model | str | None = None,
    context_window: int | None = None,
    timeline: str | None = None,
    compaction: Literal["all", "last"] | int = "all",
    depth: int | None = None,
    include_scorers: bool = False,
    prompt_reserve: int | float = 0.2,
) -> AsyncIterator[MessagesSegment]:
    """Yield pre-rendered message segments from a transcript.

    Automatically selects the best extraction strategy based on
    what data is available on the transcript:

    - If timelines are present, delegates to ``timeline_messages()``
    - If events are present (no timelines), delegates to
      ``segment_messages()`` with compaction handling
    - If only messages are present, delegates to ``segment_messages()``
      for context window segmentation only

    By default, scorer events are excluded from extraction. This
    applies to both the timeline path (the scorers span is pruned)
    and the events path (the scorers section is removed).

    Since ``TimelineMessages`` is structurally compatible with
    ``MessagesSegment``, callers get a uniform interface. Those needing
    span context can isinstance-check for ``TimelineMessages``.

    Args:
        transcript: The transcript to extract messages from.
        messages_as_str: Rendering function from ``message_numbering()``.
        model: The model used for scanning.
        context_window: Override for the model's context window size.
        timeline: Name of the timeline to extract from. ``None``
            (default) uses the transcript's first timeline. Raises
            ``ValueError`` if no timeline with the given name exists.
        compaction: How to handle compaction boundaries when extracting
            messages from events.
        depth: Maximum nesting level of scannable spans to process when
            timelines are present. Counts only spans that actually get
            scanned — non-scannable container and utility spans are
            transparent. ``1`` = outermost scannable spans on each
            branch; ``None`` (default) = unlimited. Ignored for
            events-only or messages-only paths.
        include_scorers: Whether to include scorer events in message
            extraction. Defaults to ``False``.
        prompt_reserve: Context-window allowance for prompt scaffolding
            wrapped around the rendered messages (e.g. a scanner
            template). A ``float`` reserves that fraction of the window;
            an ``int`` reserves that many tokens (plus a small safety
            margin). Default ``0.2`` leaves 80% of the window for
            messages. Forwarded to ``segment_messages()`` /
            ``timeline_messages()``.

    Yields:
        ``MessagesSegment`` (or ``TimelineMessages``) for each segment.

    Raises:
        ValueError: If ``timeline`` names a timeline that does not
            exist on the transcript.
    """
    if timeline is not None:
        raise ValueError(
            f"Timeline '{timeline}' not found in transcript "
            f"'{transcript.transcript_id}'. Transcript has no timelines."
        )
    async for seg in segment_messages(
        transcript.messages,
        messages_as_str=messages_as_str,
        model=model,
        context_window=context_window,
        prompt_reserve=prompt_reserve,
    ):
        yield seg


