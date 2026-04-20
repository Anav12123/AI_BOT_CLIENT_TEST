"""
WebSearch.py — SerpAPI Google Search
Query conversion handled by Agent.py (LLM-based).
Rotates between 2 API keys to avoid rate limits.
"""

import os
import re
import httpx
from typing import Optional


class WebSearch:
    def __init__(self):
        self._keys = []
        # Load SERPAPI_KEY_1 through SERPAPI_KEY_17 (only those set)
        for i in range(1, 18):
            k = os.environ.get(f"SERPAPI_KEY_{i}", "").strip().strip('"\'')
            if k:
                self._keys.append(k)
        # Backwards compatibility: also accept SERPAPI_KEY (unnumbered)
        k_plain = os.environ.get("SERPAPI_KEY", "").strip().strip('"\'')
        if k_plain and k_plain not in self._keys:
            self._keys.append(k_plain)
        if not self._keys:
            print("[WebSearch] ⚠️  No SERPAPI keys — web search disabled")
        else:
            first = self._keys[0]
            print(f"[WebSearch] {len(self._keys)} key(s) loaded (key1: {first[:8]}...)")
        self._key_index = 0
        self._client = httpx.AsyncClient(timeout=20.0)

    def _next_key(self) -> str:
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def _trim_query(self, query: str, max_words: int = 20) -> str:
        """Clean up query — strip tags, prefixes, cap length."""
        clean = re.sub(r'\[LANG:\w+\]\s*', '', query).strip()
        for prefix in ["sam,", "sam ", "hey sam,", "hey sam ",
                        "can you tell me", "could you tell me",
                        "please tell me", "do you know",
                        "i want to know", "tell me"]:
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):].strip().lstrip(",. ")
        words = clean.split()
        return " ".join(words[:max_words])

    async def search(self, query: str) -> Optional[str]:
        if not self._keys:
            return None

        trimmed = self._trim_query(query)
        api_key = self._next_key()
        print(f"[WebSearch] SerpAPI query: \"{trimmed}\" (key #{self._key_index})")

        try:
            resp = await self._client.get(
                "https://serpapi.com/search.json",
                params={"engine": "google", "q": trimmed, "api_key": api_key, "num": 3},
            )
            if resp.status_code != 200:
                print(f"[WebSearch] HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            data = resp.json()

            # 1. Answer box
            ab = data.get("answer_box", {})
            answer = ab.get("answer", "") or ab.get("snippet", "")
            if answer:
                print(f"[WebSearch] Answer box ({len(answer)} chars)")
                return answer[:800]

            # 2. Knowledge graph
            kg = data.get("knowledge_graph", {})
            if kg.get("description"):
                title = kg.get("title", "")
                result = f"{title}: {kg['description']}" if title else kg["description"]
                print(f"[WebSearch] Knowledge graph ({len(result)} chars)")
                return result[:800]

            # 3. AI overview
            ai = data.get("ai_overview", {})
            if ai:
                parts = [b.get("snippet", "") for b in ai.get("text_blocks", []) if b.get("snippet")]
                if parts:
                    combined = " ".join(parts)[:800]
                    print(f"[WebSearch] AI overview ({len(combined)} chars)")
                    return combined

            # 4. Organic results
            organic = data.get("organic_results", [])
            parts = [r.get("snippet", "") for r in organic[:3] if r.get("snippet")]
            if parts:
                combined = " ".join(parts)[:800]
                print(f"[WebSearch] Organic results ({len(combined)} chars)")
                return combined

            return None

        except httpx.TimeoutException:
            print(f"[WebSearch] TIMEOUT: {trimmed}")
            return None
        except Exception as e:
            print(f"[WebSearch] Error: {type(e).__name__}: {e}")
            return None

    async def search_raw(self, query: str, max_length: int = 2000) -> Optional[str]:
        """Search via SerpAPI Google AI Mode engine — for rich persona-based queries.

        IMPORTANT: Uses engine='google_ai_mode' (NOT 'google'). These are
        different SerpAPI products:
          - engine='google' → regular Google Search (returns organic + AI Overview)
          - engine='google_ai_mode' → Google AI Mode (returns AI-generated answer)

        Response structure for google_ai_mode:
          - text_blocks: list of structured blocks (paragraphs, lists)
          - reconstructed_markdown: complete markdown answer (top-level field)
          - references: citation links
          - search_metadata.status: 'Success' or 'Error'

        Returns the answer text (up to 1500 chars) or None on failure.

        DEBUG MODE: logs full query + response + saves raw JSON to disk.
        Set SERPAPI_DEBUG=0 in env to disable JSON dumps.
        """
        if not self._keys:
            return None

        # Cap length to avoid query limits
        if len(query) > max_length:
            print(f"[WebSearch] ⚠️  Query truncated: {len(query)} → {max_length} chars")
            query = query[:max_length]

        api_key = self._next_key()
        debug_enabled = os.environ.get("SERPAPI_DEBUG", "1") != "0"

        # ── DEBUG: Log full query ──
        print(f"[WebSearch] ═══════════════════════════════════════════════════════")
        print(f"[WebSearch] Google AI Mode query: {len(query)} chars (key #{self._key_index})")
        print(f"[WebSearch] ── Full query ──")
        for line in query.split("\n"):
            print(f"[WebSearch] │ {line}")
        print(f"[WebSearch] ── End of query ──")

        try:
            resp = await self._client.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_ai_mode",  # Google AI Mode engine
                    "q": query,
                    "api_key": api_key,
                    # Location/language params help AI Mode ground properly.
                    # Without these, SerpAPI's US servers can hit a variant of
                    # AI Mode that skips grounding and just makes things up.
                    # "in" = India (matches AnavClouds' web presence).
                    "gl": os.environ.get("SERPAPI_GL", "in"),
                    "hl": os.environ.get("SERPAPI_HL", "en"),
                    "location": os.environ.get("SERPAPI_LOCATION", "India"),
                },
            )
            if resp.status_code != 200:
                print(f"[WebSearch] ❌ HTTP {resp.status_code}: {resp.text[:300]}")
                print(f"[WebSearch] ═══════════════════════════════════════════════════════")
                return None
            data = resp.json()

            # ── DEBUG: Save raw response ──
            if debug_enabled:
                try:
                    import json as _json
                    import time as _t
                    timestamp = _t.strftime("%Y%m%d-%H%M%S")
                    debug_dir = "serpapi_debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_file = os.path.join(debug_dir, f"aimode_{timestamp}.json")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        _json.dump({
                            "query": query,
                            "response": data,
                        }, f, indent=2, ensure_ascii=False)
                    print(f"[WebSearch] 💾 Debug saved: {debug_file}")
                except Exception as _e:
                    print(f"[WebSearch] ⚠️  Debug save failed: {_e}")

            # Check search status
            meta = data.get("search_metadata", {})
            status = meta.get("status", "?")
            if status != "Success":
                err = data.get("error") or meta.get("error", "")
                print(f"[WebSearch] ❌ AI Mode status: {status}, error: {err[:200]}")
                print(f"[WebSearch] ═══════════════════════════════════════════════════════")
                return None

            # ── DEBUG: Log which fields came back ──
            top_keys = [k for k in data.keys() if k not in ("search_metadata", "search_parameters")]
            print(f"[WebSearch] Response top-level fields: {top_keys}")

            # ── Grounding signal (informational only, not gating) ──
            # Earlier versions rejected responses missing 'references' as ungrounded.
            # Removed because it threw away factually correct answers (e.g. AnavClouds
            # office location) where AI Mode just didn't surface references in the
            # response format. We log the signal but trust the response either way.
            references = data.get("references", [])
            if isinstance(references, list) and len(references) > 0:
                print(f"[WebSearch] ℹ️  Response has {len(references)} reference(s) (grounded)")
            else:
                print(f"[WebSearch] ℹ️  Response has no references field (grounding status unknown)")

            # Strategy 1: reconstructed_markdown (preferred — complete answer)
            reconstructed = data.get("reconstructed_markdown", "") or ""
            if reconstructed and reconstructed.strip():
                print(f"[WebSearch] ✅ reconstructed_markdown ({len(reconstructed)} chars)")
                print(f"[WebSearch] ── Content (first 300 chars) ──")
                print(f"[WebSearch] {reconstructed[:300]}")
                print(f"[WebSearch] ═══════════════════════════════════════════════════════")
                return reconstructed[:1500]

            # Strategy 2: text_blocks (fallback if reconstructed_markdown missing)
            text_blocks = data.get("text_blocks", [])
            if text_blocks:
                print(f"[WebSearch] text_blocks count: {len(text_blocks)}")
                parts = []
                for b in text_blocks:
                    snippet = b.get("snippet") or b.get("text") or ""
                    if snippet:
                        parts.append(snippet)
                    for item in b.get("list", []):
                        item_text = item.get("snippet") or item.get("text") or ""
                        if item_text:
                            parts.append(item_text)
                if parts:
                    combined = " ".join(parts)
                    print(f"[WebSearch] ✅ text_blocks assembled ({len(combined)} chars)")
                    print(f"[WebSearch] ── Content (first 300 chars) ──")
                    print(f"[WebSearch] {combined[:300]}")
                    print(f"[WebSearch] ═══════════════════════════════════════════════════════")
                    return combined[:1500]
                else:
                    print(f"[WebSearch] ⚠️  text_blocks exist but no usable text")

            # Neither strategy worked
            print(f"[WebSearch] ❌ No usable content in AI Mode response → caller falls back to Azure")
            print(f"[WebSearch] ═══════════════════════════════════════════════════════")
            return None

        except httpx.TimeoutException:
            print(f"[WebSearch] ❌ AI Mode TIMEOUT")
            print(f"[WebSearch] ═══════════════════════════════════════════════════════")
            return None
        except Exception as e:
            print(f"[WebSearch] ❌ AI Mode error: {type(e).__name__}: {e}")
            print(f"[WebSearch] ═══════════════════════════════════════════════════════")
            return None
            return None

    async def close(self):
        await self._client.aclose()