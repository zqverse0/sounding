# 摘自 inspect_ai 0.3.287.dev30+g7340817a6 model/_model_info.py（MIT）：模型资料库查找（精确 → 大小写无关 → 模糊）。
# 资料库在 _model_info.json：inspect_ai 解析好的「名字 / 别名 → 输入 token 上限」，键顺序保留（模糊匹配平分时取先到的）。
# 函数原样复制；`ModelInfo` 换成输入 token 数（int | None）。同步上游时重新生成。
import json
import re
from functools import cache
from pathlib import Path
from typing import Any

ModelInfo = Any


@cache
def _get_model_info_db() -> dict[str, int | None]:
    data: dict[str, int | None] = json.loads(
        (Path(__file__).parent / "_model_info.json").read_text()
    )
    return data


@cache
def _get_lookup_index() -> dict[str, str]:
    return {_normalize_for_lookup(key): key for key in _get_model_info_db()}


def _normalize_for_lookup(name: str) -> str:
    """Normalize a model name for case-insensitive lookup."""
    return name.lower().replace("_", "-")


SERVICE_PREFIXES = {"azure", "bedrock", "vertex"}


HOSTING_PROVIDERS = {"azureai", "bedrock", "vertex"}


PROVIDER_SCOPED_ORGS = {"fireworks"}


def _detect_org_from_model_name(model_name: str) -> str | None:
    """Detect the organization from a model name pattern.

    Args:
        model_name: The model name (without provider prefix).

    Returns:
        The detected organization, or None if not detected.
    """
    name = model_name.lower()

    # OpenAI models: gpt-*, o1*, o3*, o4*
    if (
        name.startswith("gpt-")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    ):
        return "openai"

    # Anthropic models: claude-*
    if name.startswith("claude"):
        return "anthropic"

    # Mistral models: mistral-*, mixtral-*
    if name.startswith("mistral") or name.startswith("mixtral"):
        return "mistral"

    # Google models: gemini-*
    if name.startswith("gemini"):
        return "google"

    # Moonshot AI models: kimi-*
    if name.startswith("kimi"):
        return "moonshotai"

    # DeepSeek models: deepseek-*
    if name.startswith("deepseek"):
        return "deepseek"

    return None


def _extract_model_name(name: str) -> str:
    """Extract the model name component from a full path.

    For paths like org/model, returns both parts.
    For paths like provider/org/model, returns org/model.
    For paths like provider/service/model where service is a known hosting service
    (azure, bedrock, vertex), strips the service and returns provider/model.
    For hosting providers (azureai, bedrock, vertex), detects the organization
    from the model name.

    Examples:
        "openai/gpt-4o" → "openai/gpt-4o"
        "together/meta-llama/Llama-3" → "meta-llama/Llama-3"
        "openai/azure/gpt-4o" → "openai/gpt-4o" (service prefix stripped)
        "anthropic/bedrock/claude-3" → "anthropic/claude-3" (service prefix stripped)
        "google/vertex/gemini-2" → "google/gemini-2" (service prefix stripped)
        "azureai/gpt-4o" → "openai/gpt-4o" (org detection)
        "azureai/claude-3-sonnet" → "anthropic/claude-3-sonnet" (org detection)
        "bedrock/claude-3-sonnet" → "anthropic/claude-3-sonnet" (org detection)
        "vertex/gemini-2.0-flash" → "google/gemini-2.0-flash" (org detection)
    """
    parts = name.split("/")
    if len(parts) >= 3:
        # Check if the second-to-last part is a known service prefix
        # e.g., "openai/azure/gpt-4o" → parts[-2] is "azure"
        if parts[-2].lower() in SERVICE_PREFIXES:
            # Strip the service prefix: return provider/model
            return f"{parts[-3]}/{parts[-1]}"
    if len(parts) >= 2:
        # Check if this is a hosting provider that needs org detection
        provider = parts[-2].lower()
        if provider in HOSTING_PROVIDERS:
            model_name = parts[-1]
            detected_org = _detect_org_from_model_name(model_name)
            if detected_org:
                return f"{detected_org}/{model_name}"
        return "/".join(parts[-2:])
    return parts[-1] if parts else name


def _normalize_for_fuzzy(name: str) -> str:
    """Aggressive normalization for fuzzy matching.

    Normalizes case, underscores/hyphens, and strips version suffixes.
    """
    name = name.lower()
    name = name.replace("_", "-")
    # Strip version suffixes (-v1, -v2, :0, etc.)
    name = re.sub(r"-v\d+$", "", name)
    name = re.sub(r":\d+$", "", name)
    return name


def _compute_match_score(query: str, target: str) -> int:
    """Compute match score between query and target.

    Returns:
        Score from 0-100. Higher is better.
        - 100: Exact match
        - 50-99: Substring match (scored by overlap ratio)
        - 0: No match
    """
    if query == target:
        return 100  # Exact match

    if query in target or target in query:
        # Substring match - score by overlap ratio
        overlap = len(query) if query in target else len(target)
        max_len = max(len(query), len(target))
        return 50 + int(50 * overlap / max_len)

    return 0


def _fuzzy_match(name: str, db: dict[str, ModelInfo]) -> ModelInfo | None:
    """Try to fuzzy match a model name against the database.

    Extracts the model name component and tries to find the best match
    in the database using normalized comparison.

    Args:
        name: The canonical model name to look up.
        db: The model info database.

    Returns:
        The best matching ModelInfo, or None if no match found.
    """
    model_part = _extract_model_name(name)
    normalized_query = _normalize_for_fuzzy(model_part)

    best_match: tuple[int, str, ModelInfo] | None = None  # (score, key, info)

    for key, info in db.items():
        if key.split("/", 1)[0].lower() in PROVIDER_SCOPED_ORGS:
            continue

        key_model = _extract_model_name(key)
        key_normalized = _normalize_for_fuzzy(key_model)

        score = _compute_match_score(normalized_query, key_normalized)
        if score > 0:
            if best_match is None:
                best_match = (score, key, info)
            elif score > best_match[0]:
                best_match = (score, key, info)

    # Only return matches with reasonable confidence
    if best_match and best_match[0] >= 60:
        return best_match[2]

    return None


def _lookup_in_db(name: str, db: dict[str, ModelInfo]) -> ModelInfo | None:
    """Try to find a model in the database using various matching strategies.

    Args:
        name: The model name to look up.
        db: The model info database.

    Returns:
        ModelInfo if found, None otherwise.
    """
    # Try exact match
    if name in db:
        return db[name]

    # Try case-insensitive match using the lookup index
    index = _get_lookup_index()
    normalized = _normalize_for_lookup(name)
    if normalized in index:
        return db[index[normalized]]

    # Try fuzzy matching
    return _fuzzy_match(name, db)
