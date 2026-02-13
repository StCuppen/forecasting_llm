"""Tests for evidence pipeline improvements: primary source detection, signpost anchoring."""
from __future__ import annotations

import re

from bot.agent.lean_ensemble import (
    _anchor_signposts,
    _identify_primary_source_urls,
    _is_primary_source_url,
)


def test_identify_primary_source_urls_openai():
    urls = _identify_primary_source_urls(
        "Will OpenAI API token prices fall before March 14?",
        "Resolution via openai.com API pricing page",
    )
    assert any("openai.com" in u for u in urls), f"Expected openai.com, got {urls}"


def test_identify_primary_source_urls_gov():
    urls = _identify_primary_source_urls(
        "Will US CPI exceed 3% in 2026?",
        "Consumer Price Index published by BLS",
    )
    assert any("bls.gov" in u for u in urls)


def test_identify_primary_source_urls_no_match():
    urls = _identify_primary_source_urls(
        "Will it rain tomorrow in Amsterdam?",
        "Weather observation",
    )
    assert urls == []


def test_is_primary_source_url_question_aware():
    assert _is_primary_source_url("https://openai.com/pricing", ["openai.com"]) is True
    assert _is_primary_source_url("https://reuters.com/article", ["openai.com"]) is False


def test_is_primary_source_url_static_domains():
    assert _is_primary_source_url("https://bls.gov/cpi/data", None) is True
    assert _is_primary_source_url("https://example.edu/research", None) is True
    assert _is_primary_source_url("https://random-blog.com/post", None) is False


def test_anchor_signposts_clips_negative():
    signposts = [{"event": "Bad news", "direction": "down", "magnitude": "-10pp"}]
    anchored = _anchor_signposts(signposts, final_probability=0.047)
    # 4.7% - 10pp = -5.3% -> clipped to 1%, effective = (1% - 4.7%) = -3.7pp -> ceil to -3pp
    mag_match = re.search(r"([+-]?\d+)", anchored[0]["magnitude"])
    assert mag_match is not None
    effective_pp = int(mag_match.group(1))
    assert effective_pp > -10, f"Should be clipped from -10pp, got {effective_pp}pp"
    # Truncation toward zero ensures applying the magnitude keeps probability >= 1%
    adjusted = 0.047 + effective_pp / 100.0
    assert adjusted >= 0.01, f"Adjusted {adjusted:.3f} should be >= 0.01"


def test_anchor_signposts_clips_above_99():
    signposts = [{"event": "Great news", "direction": "up", "magnitude": "+20pp"}]
    anchored = _anchor_signposts(signposts, final_probability=0.92)
    # 92% + 20pp = 112% -> clipped to 99%, effective = +7pp
    mag_match = re.search(r"\+(\d+)", anchored[0]["magnitude"])
    assert mag_match is not None
    effective_pp = int(mag_match.group(1))
    assert effective_pp <= 7, f"Expected <=7pp, got +{effective_pp}pp"


def test_anchor_signposts_preserves_valid():
    signposts = [{"event": "Some event", "direction": "up", "magnitude": "+15pp"}]
    anchored = _anchor_signposts(signposts, final_probability=0.50)
    # 50% + 15pp = 65% -> no clipping needed
    assert anchored[0]["magnitude"] == "+15pp"


def test_anchor_signposts_unparseable_magnitude():
    signposts = [{"event": "Mystery", "direction": "up", "magnitude": "unknown"}]
    anchored = _anchor_signposts(signposts, final_probability=0.50)
    assert anchored[0]["magnitude"] == "unknown"
