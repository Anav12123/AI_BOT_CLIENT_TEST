"""
websocket_server.py — Production voice pipeline with Output Media streaming

Audio delivery:
  Primary: Cartesia WebSocket → PCM chunks → audio page WebSocket → AudioWorklet
  Fallback: Cartesia REST → MP3 → output_audio API (if audio page not connected)
"""

import asyncio
import json
import time
import base64
import os
import re as _re
import random
from aiohttp import web
import aiohttp
from collections import deque

from Trigger import TriggerDetector
from Agent import PMAgent, FILLERS
from Speaker import CartesiaSpeaker, _mix_noise, get_duration_ms
from vad import RmsVAD
from JiraClient import JiraClient, JiraAuthError, JiraNotFoundError, JiraTransitionError, JiraPermissionError
from jira_prompts import AzureExtractor, JIRA_RESPONSE_PROMPT, JIRA_INTENT_PROMPT
from standup import StandupFlow
import session_store

# Deepgram Flux for standup STT (separate from Recall's Deepgram)
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 FLAGS — Flip to False to revert to previous behavior
# ══════════════════════════════════════════════════════════════════════════════
# Set via env vars for easy toggling without code changes. Default ON.
# Example: USE_UNIFIED_RESEARCH=0 python main_meeting.py ... (disables new flow)

USE_UNIFIED_RESEARCH = os.environ.get("USE_UNIFIED_RESEARCH", "1") != "0"
USE_DYNAMIC_FILLERS  = os.environ.get("USE_DYNAMIC_FILLERS",  "1") != "0"
USE_SMART_PRELOAD    = os.environ.get("USE_SMART_PRELOAD",    "1") != "0"
USE_CONVERSATION_MEMORY = os.environ.get("USE_CONVERSATION_MEMORY", "1") != "0"

# Preload caps for smart-preload mode
SMALL_PROJECT_THRESHOLD = 100   # At/below this many tickets → load everything
PRELOAD_MAX_LARGE       = 100   # Cap for large projects (sprint + priority + recent)


def ts():
    return time.strftime("%H:%M:%S")

def elapsed(since: float) -> str:
    return f"{(time.time() - since)*1000:.0f}ms"

WORDS_PER_SECOND = 3.2
PCM_SAMPLE_RATE  = 48000  # Cartesia WebSocket output
PCM_BYTES_PER_SEC = PCM_SAMPLE_RATE * 2  # 16-bit mono

_ACK_PHRASES = frozenset({
    "sure", "ok", "okay", "yeah", "yes", "go ahead", "alright",
    "right", "hmm", "mhm", "cool", "got it", "fine", "yep", "yup",
    "carry on", "go on", "continue", "waiting", "i'm waiting",
    "i am waiting", "no problem", "take your time", "np",
    "hello", "hi", "hey", "huh", "what", "sorry",
})

_INTERRUPT_ACKS = [
    "Oh sorry, go ahead.",
    "My bad, what were you saying?",
    "Sure, I'm listening.",
    "Oh, go on.",
]

_TRANSCRIPTION_FIXES = [
    (_re.compile(r'\b(?:NF\s*Cloud|Enuf\s*Cloud|Enough\s*Cloud|Nav\s*Cloud|Anav\s*Cloud|Arnav\s*Cloud|Anab\s*Cloud|NFClouds?|EnoughClouds?|NavClouds?|AnavCloud)\b', _re.IGNORECASE), 'AnavClouds'),
    (_re.compile(r'\b(?:Sales\s*Force|Sells\s*Force|Cells\s*Force|SalesForce)\b', _re.IGNORECASE), 'Salesforce'),
]

def _fix_transcription(text):
    result = text
    for p, r in _TRANSCRIPTION_FIXES:
        result = p.sub(r, result)
    return result

def _is_ack(text):
    fragments = _re.split(r'[.!?,]+', text.strip().lower())
    return all(f.strip() in _ACK_PHRASES or f.strip() == "" for f in fragments) and text.strip() != ""


# ── Spoken number → ticket ID pre-converter ──────────────────────────────────
_SPOKEN_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50",
}

def _convert_spoken_ticket_refs(text: str, project_key: str) -> str:
    """Convert spoken ticket references to proper IDs before LLM processing.
    'scrum five' → 'SCRUM-5', 'ticket number twenty three' → 'SCRUM-23'."""
    result = text
    pk_lower = project_key.lower()
    words = result.split()
    new_words = []
    i = 0
    while i < len(words):
        word_lower = words[i].strip(".,!?").lower()

        # Check if this word is a ticket reference trigger
        is_trigger = word_lower in (pk_lower, "ticket", "issue", "number", "task")

        if is_trigger and i + 1 < len(words):
            # Collect following number words
            num_parts = []
            j = i + 1
            while j < len(words):
                w = words[j].strip(".,!?").lower()
                # Skip filler words between trigger and number
                if w in ("number", "no", "num", "#"):
                    j += 1
                    continue
                if w in _SPOKEN_NUMBERS:
                    num_parts.append(_SPOKEN_NUMBERS[w])
                    j += 1
                elif w.isdigit():
                    num_parts.append(w)
                    j += 1
                else:
                    break

            if num_parts:
                # Handle compound numbers: "twenty" + "three" = 23
                if len(num_parts) == 2 and int(num_parts[0]) >= 20 and int(num_parts[1]) < 10:
                    ticket_num = str(int(num_parts[0]) + int(num_parts[1]))
                else:
                    ticket_num = "".join(num_parts)
                ticket_id = f"{project_key}-{ticket_num}"
                new_words.append(ticket_id)
                i = j
                print(f"[STT→Ticket] Converted: \"{' '.join(words[i-j+i:j])}\" → {ticket_id}")
                continue

        new_words.append(words[i])
        i += 1

    converted = " ".join(new_words)
    if converted != text:
        print(f"[STT→Ticket] \"{text}\" → \"{converted}\"")
    return converted


class BotSession:
    STRAGGLER_WAIT = 0.2   # Reduced from 0.4 for faster non-direct response
    STRAGGLER_DIRECT = 0.0  # No straggler for "Sam, ..." — direct address is complete
    WAIT_TIMEOUT   = 2.0

    def __init__(self, session_id, bot_id, server):
        self.session_id = session_id
        self.bot_id = bot_id
        self.server = server
        self.tag = f"[S:{session_id[:8]}]"

        self.username = ""
        self.meeting_url = ""
        self.mode = "client_call"
        self.started_at = time.time()

        self.agent = PMAgent()
        self.speaker = CartesiaSpeaker(bot_id=bot_id)
        self.trigger = TriggerDetector()
        self.vad = RmsVAD()

        self.jira = JiraClient()
        self.azure_extractor = AzureExtractor()

        self.speaking = False
        self.audio_playing = False
        self.convo_history = deque(maxlen=10)
        self.current_task = None
        self.current_text = ""
        self.current_speaker = ""
        self.interrupt_event = asyncio.Event()
        self.generation = 0

        self.buffer = []
        self.partial_text = ""
        self.partial_speaker = ""
        self.last_flushed_text = ""

        self.was_interrupted = False
        self.playing_ack = False
        self._partial_interrupted = False  # Interrupted via interim transcript (fast path)
        self._partial_interrupt_time = 0   # When partial interrupt happened
        self._current_audio_duration = 0   # Duration of currently playing audio (seconds)
        self.eot_task = None
        self.searching = False

        self.audio_event_count = 0
        self.max_conf = 0.0
        self.debug_audio_file = None

        self._jira_context = ""
        self._ticket_cache = []  # Structured list of pre-loaded tickets

        # ── Option C: Deferred Ticket Creation ──
        # Instead of creating tickets mid-meeting (which often produces garbage
        # titles from ambiguous requests), we record the user's intent here and
        # let Azure create high-quality tickets at meeting-end with full context.
        #
        # Each entry: {"user_said": "...", "extracted_summary": "...", "at_time": "..."}
        # Passed to Azure extraction as EXPLICIT CREATION SIGNALS (not meta-actions
        # to ignore, but confirmed user requests to act on).
        self._pending_creation_intents: list[dict] = []

        # ── Feature 4 Memory: Conversation-scoped summaries ──
        # Attendees in this meeting (populated from Recall.ai participant events
        # during first 60 seconds). Sorted, lowercase-normalized for stable IDs.
        self._attendees: set[str] = set()
        self._attendees_locked = False  # True after lock window passes
        self._memory_lock_task_started = False  # Guard: only start the lock task once

        # Derived from attendees: stable ID for the conversation memory thread
        self._conversation_id: str = ""

        # Memory loaded at session start from past meetings with this same group.
        # _memory_full: ~1000 tokens, injected into PM/research response prompts
        # _memory_header: ~50 tokens, injected into unified research planner
        self._memory_full: str = ""
        self._memory_header: str = ""

        # Output Media: audio page WebSocket connection
        self.audio_ws = None  # Set when audio page connects

        # Standup mode
        self.standup_flow = None
        self._standup_buffer = []
        self._standup_timer = None
        self._standup_finished = False  # Guard: prevent double finish
        self._auto_left = False         # Guard: prevent double leave

        # Flux STT (own Deepgram connection for standup)
        self._stt_queue = None       # asyncio.Queue for audio chunks → Flux
        self._stt_task = None        # Background task running stream_deepgram()
        self._flux_enabled = False   # True when Flux is active for this session
        self._flux_audio_buf = b""   # Re-chunk buffer for 80ms chunks (2560 bytes)
        _FLUX_CHUNK_SIZE = 2560      # 80ms at 16kHz S16LE (recommended by Deepgram)
        self._FLUX_CHUNK_SIZE = _FLUX_CHUNK_SIZE

        # Flux speech_off debounce — Recall's participant_events.speech_off signals
        # user stopped speaking. We convert the current Flux interim text to FINAL
        # after a debounce window (allows mid-sentence pauses without premature finalize).
        # If speech_on fires within the debounce window, timer is cancelled.
        self._flux_last_interim_text = ""         # latest interim text from Flux (cleared on FINAL)
        self._flux_speech_off_task = None         # debounce timer task
        self._FLUX_SPEECH_OFF_DEBOUNCE_MS = 300   # delay before treating speech_off as turn-end

        # Speculative processing (EagerEndOfTurn → pre-compute Groq before EndOfTurn confirms)
        self._speculative_task = None   # Background Groq classify task
        self._speculative_text = ""     # Transcript used for speculation

    @property
    def _streaming_mode(self) -> bool:
        """True if Output Media audio page is connected."""
        return self.audio_ws is not None and not self.audio_ws.closed

    async def setup(self):
        self.agent.start()
        await self.speaker.warmup()
        await self.vad.setup()
        if self.jira.enabled:
            await self.jira.test_connection()
            await self._sync_pending_tickets()
            await self._preload_jira_context()

        # Feature 4 Memory: lock task will be started when first attendee joins
        # (not at session init — the bot takes time to join the meeting, so
        # starting the timer at init causes the lock to fire before anyone is present).
        # See participant_events.join handler for the actual trigger.

        print(f"[{ts()}] {self.tag} ✅ Session ready (bot: {self.bot_id[:12]})")

    async def _sync_pending_tickets(self):
        pending = session_store.get_pending_tickets()
        if not pending:
            return
        print(f"[{ts()}] {self.tag} 🔄 Syncing {len(pending)} pending ticket(s)...")
        synced = 0
        for item in pending:
            try:
                await self.jira.create_ticket(
                    summary=item.get("summary", ""), issue_type=item.get("type", "Task"),
                    priority=item.get("priority", "Medium"), description=item.get("description", ""),
                    labels=item.get("labels", []),
                )
                synced += 1
            except Exception as e:
                print(f"[{ts()}] {self.tag} ⚠️  Pending sync failed: {e}")
                break
        if synced > 0:
            session_store.clear_pending_tickets()
            print(f"[{ts()}] {self.tag} ✅ Synced {synced} pending ticket(s)")

    async def _preload_jira_context(self):
        """Load Jira tickets into cache at session start.

        Smart mode (USE_SMART_PRELOAD=True, default):
          - Small projects (≤100 tickets): load everything by recent update
          - Large projects (>100 tickets): load current sprint + high priority +
            recent (14d) + non-Done — capped at 100 most relevant

        Legacy mode (USE_SMART_PRELOAD=False): original 50-ticket recent-update load
        """
        try:
            base_filter = f"project = {self.jira.project} AND summary !~ 'Standup —'"

            if USE_SMART_PRELOAD:
                # Size probe — cheap (maxResults=0)
                total = await self.jira.count_tickets(base_filter)
                print(f"[{ts()}] {self.tag} 📊 Project size: {total} ticket(s)")

                if total <= SMALL_PROJECT_THRESHOLD:
                    # Small project: load everything by recency (your current behavior)
                    jql = f"{base_filter} ORDER BY updated DESC"
                    tickets = await self.jira.search_jql(jql, max_results=SMALL_PROJECT_THRESHOLD)
                    print(f"[{ts()}] {self.tag} 📥 Small project mode — loading all")
                else:
                    # Large project: current sprint + high priority + recent updates + open
                    # Fallback query without sprint clause if sprint() fails on some Jira configs
                    jql = (
                        f"{base_filter} AND ("
                        f"sprint in openSprints() "
                        f"OR priority in (Highest, High) "
                        f"OR updated >= -14d "
                        f"OR statusCategory != Done"
                        f") ORDER BY priority DESC, updated DESC"
                    )
                    try:
                        tickets = await self.jira.search_jql(jql, max_results=PRELOAD_MAX_LARGE)
                    except Exception as e:
                        # Sprint function not available on this Jira instance — retry without it
                        print(f"[{ts()}] {self.tag} ⚠️  Smart JQL failed ({e}) — retrying without sprint clause")
                        jql_safe = (
                            f"{base_filter} AND ("
                            f"priority in (Highest, High) "
                            f"OR updated >= -14d "
                            f"OR statusCategory != Done"
                            f") ORDER BY priority DESC, updated DESC"
                        )
                        tickets = await self.jira.search_jql(jql_safe, max_results=PRELOAD_MAX_LARGE)
                    print(f"[{ts()}] {self.tag} 📥 Large project mode — filtered load (sprint+priority+recent)")
            else:
                # Legacy mode — original 50-ticket recent-update load
                jql = f"{base_filter} ORDER BY updated DESC"
                tickets = await self.jira.search_jql(jql, max_results=50)

            if tickets:
                self._ticket_cache = tickets
                self._rebuild_jira_context()
                done_count = sum(1 for t in tickets if t.get('status') == 'Done')
                print(f"[{ts()}] {self.tag} 📥 Pre-loaded {len(tickets)} ticket(s) ({done_count} Done, standup subtasks excluded)")
            else:
                print(f"[{ts()}] {self.tag} ⚠️  No tickets returned from pre-load")
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Pre-load failed: {e}")

    def _rebuild_jira_context(self):
        """Rebuild _jira_context string from _ticket_cache."""
        if not self._ticket_cache:
            self._jira_context = "(no tickets loaded)"
            return
        lines = []
        for t in self._ticket_cache:
            line = f"  {t['key']}: {t['summary']} [{t['status']}] ({t['priority']}, {t['assignee']})"
            if t.get('description'):
                line += f" — {t['description'][:100]}"
            lines.append(line)
        self._jira_context = "JIRA TICKETS:\n" + "\n".join(lines)

    def _update_ticket_cache(self, ticket: dict):
        """Add or update a ticket in the cache."""
        for i, t in enumerate(self._ticket_cache):
            if t['key'] == ticket['key']:
                self._ticket_cache[i] = ticket
                self._rebuild_jira_context()
                return
        self._ticket_cache.append(ticket)
        self._rebuild_jira_context()

    def _get_ticket_context_for_search(self) -> str:
        """Get a compact summary of project tech stack from ticket descriptions for search query generation."""
        if not self._ticket_cache:
            return ""
        parts = []
        for t in self._ticket_cache[:10]:
            desc = t.get("description", "")
            if desc:
                parts.append(f"{t['summary']}: {desc[:80]}")
            else:
                parts.append(t['summary'])
        return "Project tickets: " + "; ".join(parts)

    async def cleanup(self):
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        if self.eot_task and not self.eot_task.done():
            self.eot_task.cancel()

        # Save standup data if standup was in progress
        if self.standup_flow:
            self.standup_flow._cancel_silence_timer()
            if self._standup_timer and not self._standup_timer.done():
                self._standup_timer.cancel()
            # Stop Flux STT
            await self._stop_flux_stt()
            if self.standup_flow.data.get("yesterday", {}).get("raw"):
                await self._finish_standup()

        try:
            await self.speaker.close()
        except Exception:
            pass

        if len(self.agent.rag._entries) > 3:
            # Skip Jira extraction for standup mode — standup already creates subtasks
            skip_jira = self.mode == "standup"
            if skip_jira:
                print(f"[{ts()}] {self.tag} ℹ️  Standup mode — skipping post-meeting Jira extraction")
            await self._post_meeting_save(extract_jira=self.jira.enabled and self.azure_extractor.enabled and not skip_jira)

        try:
            await self.jira.close()
        except Exception:
            pass
        try:
            await self.azure_extractor.close()
        except Exception:
            pass
        self.agent.reset()
        print(f"[{ts()}] {self.tag} 🧹 Session cleaned up")

    async def _post_meeting_save(self, extract_jira=True):
        print(f"[{ts()}] {self.tag} 📋 Meeting ended — processing session...")
        transcript_entries = [{"speaker": e.get("speaker", "?"), "text": e["text"], "time": e.get("time", 0)} for e in self.agent.rag._entries]
        transcript_text = "\n".join(e["text"] for e in self.agent.rag._entries if e.get("speaker", "").lower() != "sam")
        duration_min = int((time.time() - self.started_at) / 60)

        items = []
        if extract_jira:
            try:
                # OPTION C: Pass pending creation intents (from in-meeting CREATE
                # requests that were deferred). Azure treats these as EXPLICIT
                # user intents to create tickets, using full transcript context
                # for proper titles and descriptions.
                items = await self.azure_extractor.extract_action_items(
                    transcript_text,
                    pending_intents=self._pending_creation_intents,
                )
            except Exception as e:
                print(f"[{ts()}] {self.tag} ❌ Extraction failed: {e}")

        created_tickets = []
        if items and self.jira.enabled:
            user_cache = {}
            for item in items:
                try:
                    related = await self.jira.find_related_tickets(item.get("summary", ""))
                    if related:
                        rs = ", ".join(f"{r['key']} ({r['summary'][:40]})" for r in related[:3])
                        item["description"] = item.get("description", "") + f"\n\nRelated: {rs}"

                    assignee_id = None
                    an = item.get("assignee")
                    if an and an.lower() not in ("null", "none", "unassigned", ""):
                        assignee_id = user_cache.get(an) or await self.jira.search_user(an)
                        if an not in user_cache:
                            user_cache[an] = assignee_id

                    result = await self.jira.create_ticket(
                        summary=item["summary"], issue_type=item.get("type", "Task"),
                        priority=item.get("priority", "Medium"), description=item.get("description", ""),
                        labels=item.get("labels", ["client-feedback"]), assignee_id=assignee_id,
                    )
                    item["jira_key"] = result.get("key", "?")
                    created_tickets.append(item)
                    print(f"[{ts()}] {self.tag} 📤 Created {item['jira_key']}: {item['summary']}")
                except JiraAuthError:
                    session_store.save_pending_ticket(item)
                    break
                except Exception as e:
                    print(f"[{ts()}] {self.tag} ⚠️  Failed — saving locally: {e}")
                    session_store.save_pending_ticket(item)

        session_data = {
            "session_id": self.session_id, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user": self.username, "mode": self.mode,
            "project": self.jira.project if self.jira.enabled else "",
            "meeting_url": self.meeting_url, "duration_minutes": duration_min,
            "transcript": transcript_entries,
            "summary": "; ".join(e["text"][:80] for e in transcript_entries if not e["text"].startswith("Sam:"))[:200],
            "feedback_count": len(items), "tickets_created": len(created_tickets),
            "action_items": [{"type": i.get("type"), "summary": i.get("summary"), "priority": i.get("priority"), "jira_key": i.get("jira_key", ""), "assignee": i.get("assignee", "")} for i in items],
        }

        # Feature 4 Memory: generate conversation-scoped summary + save under conv_id.
        # This populates memory for future meetings with the same group of attendees.
        # Done before session_store.save_session so conv_id is available in session too.
        if USE_CONVERSATION_MEMORY and self.mode != "standup" and self.azure_extractor.enabled:
            try:
                # Lock attendees if not already locked (handles very short meetings)
                if not self._attendees_locked:
                    self._attendees_locked = True
                    if self._attendees:
                        self._conversation_id = self._compute_conversation_id(self._attendees)

                if self._conversation_id and self._attendees and transcript_text:
                    attendee_list = sorted(self._attendees)
                    print(f"[{ts()}] {self.tag} 🧠 Generating conversation summary for "
                          f"conv_id={self._conversation_id[:8]}... attendees={attendee_list}")

                    summary = await self.azure_extractor.extract_session_summary(
                        transcript=transcript_text,
                        attendees=attendee_list,
                    )

                    if summary:
                        session_store.save_conversation_summary(
                            conversation_id=self._conversation_id,
                            summary=summary,
                            participants=attendee_list,
                            session_id=self.session_id,
                        )
                        session_data["conversation_id"] = self._conversation_id
                        session_data["conversation_summary"] = summary
            except Exception as e:
                print(f"[{ts()}] {self.tag} ⚠️  Conversation summary failed (non-fatal): {e}")

        try:
            session_store.save_session(session_data)
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Save session failed: {e}")

    # ── Conversation Memory (Feature 4) ───────────────────────────────────────

    @staticmethod
    def _compute_conversation_id(attendees: set) -> str:
        """Compute a stable conversation_id from a set of attendees.
        Same set of names → same ID. Different set → different ID.
        Privacy-safe by design: different attendee groups = different memory threads.
        """
        import hashlib
        if not attendees:
            return ""
        # Normalize: strip, lowercase, sort for stability
        names = sorted(a.strip().lower() for a in attendees if a and a.strip())
        if not names:
            return ""
        canonical = "|".join(names)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    async def _lock_attendees_and_load_memory(self, delay: float = 60.0):
        """Wait for attendees to gather, then lock the set and load memory.

        Runs as a background task shortly after session start. After `delay`
        seconds, we lock the attendee set (late joiners don't change conv_id)
        and compute the conversation_id, then load + reconcile memory.
        """
        if not USE_CONVERSATION_MEMORY:
            return

        await asyncio.sleep(delay)

        if self._attendees_locked:
            return  # already done
        self._attendees_locked = True

        if not self._attendees:
            print(f"[{ts()}] {self.tag} 🧠 No attendees detected — memory load skipped")
            return

        self._conversation_id = self._compute_conversation_id(self._attendees)
        attendee_list = sorted(self._attendees)
        print(f"[{ts()}] {self.tag} 🧠 Conversation locked: "
              f"conv_id={self._conversation_id[:8]}... attendees={attendee_list}")

        await self._load_memory()

    async def _load_memory(self):
        """Load prior meeting summaries for this conversation_id.
        Builds _memory_full (for response generation) and _memory_header (for planner).
        """
        if not self._conversation_id:
            return

        try:
            summaries = session_store.get_conversation_summaries(
                self._conversation_id, limit=3
            )
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Memory load failed: {e}")
            return

        if not summaries:
            print(f"[{ts()}] {self.tag} 🧠 No prior meetings with this group — fresh start")
            return

        print(f"[{ts()}] {self.tag} 🧠 Loaded {len(summaries)} prior summary(ies)")

        # Reconcile with live Jira before formatting
        summaries = self._reconcile_memory_with_jira(summaries)

        # Build full memory (for PM/research response prompts)
        self._memory_full = self._format_full_memory(summaries)

        # Build compact header (for unified research planner)
        self._memory_header = self._format_memory_header(summaries)

    def _reconcile_memory_with_jira(self, summaries: list) -> list:
        """Annotate each summary's ticket references with CURRENT Jira status.

        Prevents stale memory from contradicting live data. If memory says
        'SCRUM-162 in progress' but Jira shows it Done, we annotate the summary
        so Sam uses current status.
        """
        if not self._ticket_cache:
            return summaries

        # Build lookup of current ticket states
        current_states = {t.get("key"): t for t in self._ticket_cache if t.get("key")}

        for entry in summaries:
            summary = entry.get("summary", {})
            refs = summary.get("tickets_referenced", [])
            updates = []

            for ref in refs:
                key = ref.get("key") if isinstance(ref, dict) else None
                if not key:
                    continue
                current = current_states.get(key)
                if not current:
                    continue

                historical_status = ref.get("status_at_time", "?") if isinstance(ref, dict) else "?"
                current_status = current.get("status", "?")

                if historical_status != current_status:
                    updates.append(
                        f"{key}: was '{historical_status}' in that meeting, "
                        f"NOW '{current_status}' (use current)"
                    )

            if updates:
                summary["_reconciliation"] = updates

        return summaries

    def _format_full_memory(self, summaries: list) -> str:
        """Build the full memory block for PM/research response prompts.
        Returns ~1000-token structured context block.
        """
        lines = ["=== MEMORY FROM PRIOR MEETINGS WITH THIS GROUP ==="]
        lines.append("(Use as context; do not recite unless asked. Current Jira state overrides historical status.)")
        lines.append("")

        for i, entry in enumerate(summaries):
            date = entry.get("date", "?")[:10]  # YYYY-MM-DD
            s = entry.get("summary", {})

            lines.append(f"--- Meeting {i+1} ({date}) ---")

            summary_text = s.get("summary_text", "").strip()
            if summary_text:
                lines.append(f"Summary: {summary_text}")

            decisions = s.get("decisions", [])
            if decisions:
                lines.append(f"Decisions: {'; '.join(decisions[:5])}")

            commitments = s.get("commitments", [])
            if commitments:
                commit_strs = []
                for c in commitments[:5]:
                    if isinstance(c, dict):
                        commit_strs.append(
                            f"{c.get('who', '?')} → {c.get('what', '?')} "
                            f"({c.get('when', 'unspecified')})"
                        )
                if commit_strs:
                    lines.append(f"Commitments: {'; '.join(commit_strs)}")

            open_items = s.get("open_items", [])
            if open_items:
                lines.append(f"Open items: {'; '.join(open_items[:5])}")

            # Reconciliation notes (stale ticket states)
            reconciliation = s.get("_reconciliation", [])
            if reconciliation:
                lines.append("Ticket status updates since this meeting:")
                for update in reconciliation:
                    lines.append(f"  - {update}")

            lines.append("")

        lines.append("=== END MEMORY ===")
        lines.append("")
        lines.append("Rules for using memory:")
        lines.append("- Reference specific items only if user asks about past or continuity helps")
        lines.append("- If memory doesn't cover a topic, say you don't have that detail")
        lines.append("- Current Jira state OVERRIDES historical status in memory")
        lines.append("- Don't invent details not in the memory above")

        return "\n".join(lines)

    def _format_memory_header(self, summaries: list) -> str:
        """Build compact header for unified research planner (~50 tokens)."""
        if not summaries:
            return ""

        # Aggregate topics and open items across summaries
        topics = []
        open_items = []
        for entry in summaries:
            s = entry.get("summary", {})
            topics.extend(s.get("topics", []))
            open_items.extend(s.get("open_items", []))

        # Dedupe preserving order
        seen = set()
        topics = [t for t in topics if not (t.lower() in seen or seen.add(t.lower()))][:5]
        seen = set()
        open_items = [i for i in open_items if not (i.lower() in seen or seen.add(i.lower()))][:3]

        parts = ["Context: Returning session with prior history."]
        if topics:
            parts.append(f"Recent topics: {', '.join(topics)}.")
        if open_items:
            parts.append(f"Open: {', '.join(open_items)}.")

        return " ".join(parts)

    # ── Event dispatch ────────────────────────────────────────────────────────

    async def handle_event(self, raw):
        t = time.time()
        try:
            payload = json.loads(raw)
        except Exception:
            return
        event = payload.get("event", "")

        if event == "transcript.data":
            inner = payload.get("data", {}).get("data", {})
            words = inner.get("words", [])
            text = " ".join(w.get("text", "") for w in words).strip()
            speaker = inner.get("participant", {}).get("name", "Unknown")
            if not text or speaker.lower() == "sam":
                return
            text = _fix_transcription(text)
            text = _re.sub(r'\s+', ' ', text).strip().lstrip("-–— ").strip()
            if not text:
                return

            self.agent.log_exchange(speaker, text)
            print(f"\n[{ts()}] {self.tag} [{speaker}] {text}")

            # ── Standup mode: buffer transcript, restart timer only when safe ──
            if self.standup_flow and not self.standup_flow.is_done:
                # Clear interrupt flag if this is the final transcript after a fast interrupt
                if self._partial_interrupted:
                    self._partial_interrupted = False
                    self._partial_interrupt_time = 0
                    self.partial_text = ""

                # When Flux is active, skip Recall's transcripts — Flux handles STT
                if self._flux_enabled:
                    return

                # Fallback: use Recall's transcripts with 1.2s timer
                self._standup_buffer.append(text)
                # Only cancel/restart timer when NOT processing
                if not self.standup_flow._processing:
                    if self._standup_timer and not self._standup_timer.done():
                        self._standup_timer.cancel()
                    self._standup_timer = asyncio.create_task(self._flush_standup_buffer(speaker))
                return

            # ── Partial interrupt: audio already stopped via interim transcript ──
            if self._partial_interrupted:
                latency_ms = (time.time() - self._partial_interrupt_time) * 1000
                self._partial_interrupted = False
                self._partial_interrupt_time = 0
                print(f"[{ts()}] {self.tag} ⚡ Fast interrupt complete: final transcript arrived {latency_ms:.0f}ms after audio stopped")
                self.buffer.clear()
                self.partial_text = ""
                await self._play_interrupt_ack()
                # Don't return — fall through to process this text normally
                self.buffer.append((speaker, text, t))
                self._schedule_eot_check(speaker)
                return

            if self.was_interrupted:
                self.was_interrupted = False
                self.buffer.clear()
                self.partial_text = ""
                await self._play_interrupt_ack()
                return

            self.partial_text = ""
            self.partial_speaker = ""

            if self.last_flushed_text:
                flushed_w = set(self.last_flushed_text.lower().split())
                incoming_w = set(text.lower().split())
                sim = len(flushed_w & incoming_w) / max(len(flushed_w), len(incoming_w), 1)
                if sim >= 0.7 or text.lower().strip() in self.last_flushed_text.lower():
                    self.last_flushed_text = ""
                    return
            self.last_flushed_text = ""

            if self.speaking and not self.playing_ack:
                if self.current_speaker == speaker and _is_ack(text.lstrip("-–— ")):
                    return
                if self.audio_playing:
                    print(f"[{ts()}] {self.tag} 🛑 Interrupt via final transcript (slow path)")
                    try:
                        await self._stop_all_audio()
                    except Exception as e:
                        print(f"[{ts()}] {self.tag} ⚠️  Stop audio failed (non-fatal): {e}")
                    self.interrupt_event.set()
                    if self.current_task and not self.current_task.done():
                        self.current_task.cancel()
                    self.speaking = False
                    self.audio_playing = False
                    self.buffer.clear()
                    self.vad.end_turn()
                    await self._play_interrupt_ack()
                    return
                elif self.current_speaker != speaker or len(text.split()) > 8:
                    if self.current_task and not self.current_task.done():
                        self.current_task.cancel()
                    self.interrupt_event.set()
                    await asyncio.sleep(0.05)
                    self.speaking = False
                    self.buffer.append((speaker, text, t))
                    self._schedule_eot_check(speaker)
                    return

            self.buffer.append((speaker, text, t))
            self._schedule_eot_check(speaker)

        elif event == "transcript.partial_data":
            inner = payload.get("data", {}).get("data", {})
            text = " ".join(w.get("text", "") for w in inner.get("words", [])).strip()
            speaker = inner.get("participant", {}).get("name", "Unknown")
            if text and speaker.lower() != "sam":
                self.partial_text = _fix_transcription(text)
                self.partial_speaker = speaker
                if self.eot_task and not self.eot_task.done():
                    self.eot_task.cancel()

                # ── Fast interrupt: stop Sam's audio on first interim words ──
                if self.speaking and self.audio_playing and not self.playing_ack and not self._partial_interrupted:
                    # In standup Q&A phase, don't interrupt — user speech gets buffered
                    # EXCEPTION: if a re-prompt is playing, user is finally answering → interrupt!
                    if self.standup_flow and not self.standup_flow.is_done:
                        from standup import StandupState
                        is_reprompt = getattr(self.standup_flow, '_playing_reprompt', False)
                        if not is_reprompt:
                            if self.standup_flow.state not in (StandupState.CONFIRM, StandupState.SUMMARY):
                                return  # Q&A phase — buffer, don't interrupt
                            # Only interrupt long audio (summary >5s), not short acks (<3s)
                            if self._current_audio_duration < 5.0:
                                return  # Short response — don't interrupt, buffer instead
                        else:
                            print(f"[{ts()}] {self.tag} ⚡ Re-prompt interrupted — user is answering")

                    # Don't interrupt for very short interims (single word could be noise)
                    word_count = len(text.split())
                    if word_count >= 2:
                        self._partial_interrupted = True
                        self._partial_interrupt_time = time.time()
                        print(f"[{ts()}] {self.tag} ⚡ FAST INTERRUPT via interim: \"{text[:40]}\" ({word_count} words) — stopping audio")
                        try:
                            await self._stop_all_audio()
                        except Exception as e:
                            print(f"[{ts()}] {self.tag} ⚠️  Stop audio failed (non-fatal): {e}")
                        self.interrupt_event.set()
                        if self.current_task and not self.current_task.done():
                            self.current_task.cancel()
                        # Cancel standup flush so _processing gets released
                        if self.standup_flow and not self.standup_flow.is_done:
                            if self._standup_timer and not self._standup_timer.done():
                                self._standup_timer.cancel()
                            self.standup_flow._processing = False
                        self.speaking = False
                        self.audio_playing = False
                        self.vad.end_turn()

        elif event == "participant_events.speech_off":
            # In standup mode with Flux: user stopped producing audio.
            # Flux can't detect silence-based EOT when mic is muted (no audio packets).
            # Start a debounce timer — if silence persists past threshold, treat Flux's
            # current interim as FINAL. If speech_on fires within debounce, cancel the timer.
            if (self.standup_flow
                and not self.standup_flow.is_done
                and self._flux_enabled
                and self._flux_last_interim_text):
                speaker_name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "")
                if speaker_name == self.standup_flow.developer:
                    # Cancel any existing debounce (rare — back-to-back speech_off)
                    if self._flux_speech_off_task and not self._flux_speech_off_task.done():
                        self._flux_speech_off_task.cancel()
                    self._flux_speech_off_task = asyncio.create_task(self._flux_debounce_finalize())

        elif event == "participant_events.speech_on":
            speaker = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            # Cancel pending speech_off debounce — user resumed speaking, Flux will get more audio
            if self._flux_speech_off_task and not self._flux_speech_off_task.done():
                print(f"[{ts()}] {self.tag} 🎤 speech_on — cancelling speech_off debounce")
                self._flux_speech_off_task.cancel()
                self._flux_speech_off_task = None
            # In standup mode, don't interrupt on speech_on — too aggressive (fires on mic noise)
            # Only partial_data (actual transcribed words) should interrupt during CONFIRM/SUMMARY
            if self.standup_flow and not self.standup_flow.is_done:
                return
            if self.speaking and self.audio_playing and self.current_speaker != speaker:
                try:
                    await self._stop_all_audio()
                except Exception as e:
                    print(f"[{ts()}] {self.tag} ⚠️  Stop audio failed (non-fatal): {e}")
                self.interrupt_event.set()
                if self.current_task and not self.current_task.done():
                    self.current_task.cancel()
                # Release standup processing lock on interrupt
                if self.standup_flow and not self.standup_flow.is_done:
                    if self._standup_timer and not self._standup_timer.done():
                        self._standup_timer.cancel()
                    self.standup_flow._processing = False
                self.speaking = False
                self.audio_playing = False
                self.was_interrupted = True

        elif event == "audio_mixed_raw.data":
            if not self.vad.ready or self.audio_playing:
                return
            audio_b64 = payload.get("data", {}).get("data", {}).get("buffer", "")
            if audio_b64:
                try:
                    pcm = base64.b64decode(audio_b64)
                    for rms in self.vad.process_chunk(pcm):
                        self.vad.update_state(rms)
                    self.audio_event_count += 1
                    if self.audio_event_count == 1:
                        print(f"[{ts()}] {self.tag} 🔊 First audio received ({len(pcm)} bytes)")
                except Exception:
                    pass

        elif event == "audio_separate_raw.data":
            # Feed developer's audio to Flux STT (rechunked to 80ms)
            if self._flux_enabled and self._stt_queue:
                inner = payload.get("data", {}).get("data", {})
                participant = inner.get("participant", {}).get("name", "")
                if participant and self.standup_flow and participant == self.standup_flow.developer:
                    audio_b64 = inner.get("buffer", "")
                    if audio_b64:
                        try:
                            pcm = base64.b64decode(audio_b64)
                            self._flux_audio_buf += pcm
                            # Send 80ms chunks (2560 bytes) — Flux recommended size
                            # IMPORTANT: Send ALL audio including silence — Flux needs
                            # to hear silence to detect EndOfTurn via its internal VAD.
                            while len(self._flux_audio_buf) >= self._FLUX_CHUNK_SIZE:
                                chunk = self._flux_audio_buf[:self._FLUX_CHUNK_SIZE]
                                self._flux_audio_buf = self._flux_audio_buf[self._FLUX_CHUNK_SIZE:]
                                await self._stt_queue.put(chunk)
                        except Exception:
                            pass

        elif event == "participant_events.join":
            name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            if name and name.lower() != "sam":
                print(f"[{ts()}] {self.tag} 👋 {name} joined")
                # Feature 4 Memory: record attendee + start lock task on first join.
                # We start the 30s timer when the FIRST attendee joins (not at session
                # init) because the bot/participants take variable time to join.
                if USE_CONVERSATION_MEMORY and not self._attendees_locked:
                    self._attendees.add(name.strip())
                    if not self._memory_lock_task_started and self.mode != "standup":
                        self._memory_lock_task_started = True
                        print(f"[{ts()}] {self.tag} 🧠 Memory lock timer started (30s window)")
                        asyncio.create_task(self._lock_attendees_and_load_memory(delay=30.0))
                if self.mode == "standup":
                    asyncio.create_task(self._start_standup(name))
                else:
                    asyncio.create_task(self._greet(name, t))

        elif event == "participant_events.leave":
            name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            if name and name.lower() != "sam":
                print(f"[{ts()}] {self.tag} 👋 {name} left")

    # ── EOT ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_direct_address(text: str) -> bool:
        """Check if text starts with 'Sam' — direct address, skip EOT + straggler."""
        t = text.strip().lower()
        return t.startswith("sam,") or t.startswith("sam ") or t == "sam" or \
               t.startswith("hey sam") or t.startswith("hi sam") or t.startswith("hello sam")

    def _schedule_eot_check(self, speaker):
        if self.eot_task and not self.eot_task.done():
            self.eot_task.cancel()
        self.eot_task = asyncio.create_task(self._run_eot_check(speaker))

    async def _run_eot_check(self, speaker):
        try:
            result = self._get_buffer_text()
            if not result or self.speaking:
                return
            spk, full_text, t0 = result
            context = "\n".join(self.convo_history)

            # ── Fast path: direct address skips EOT classifier + straggler ──
            if self._is_direct_address(full_text):
                print(f"[{ts()}] {self.tag} ⚡ Direct address — skipping EOT + straggler")
                # No straggler wait at all
            else:
                # Normal path: run EOT classifier
                decision = await self.agent.check_end_of_turn(full_text, context)
                if decision == "RESPOND":
                    await asyncio.sleep(self.STRAGGLER_WAIT)  # 200ms
                else:
                    await asyncio.sleep(self.WAIT_TIMEOUT)

            if self.speaking or not self.buffer:
                return
            result = self._get_buffer_text()
            if not result:
                return
            spk, full_text, t0 = result
            self.buffer.clear()
            self.partial_text = ""
            self.vad.end_turn()
            self.last_flushed_text = full_text
            self._start_process(full_text, spk, t0)
        except asyncio.CancelledError:
            return

    def _get_buffer_text(self):
        if not self.buffer and not self.partial_text:
            return None
        if self.buffer:
            speaker = self.buffer[-1][0]
            t0 = self.buffer[0][2]
            full_text = " ".join(txt for _, txt, _ in self.buffer)
            if self.partial_text and self.partial_text not in full_text:
                full_text += " " + self.partial_text
        else:
            speaker = self.partial_speaker or "Unknown"
            t0 = time.time()
            full_text = self.partial_text
        return speaker, full_text, t0

    def _start_process(self, text, speaker, t0):
        self.generation += 1
        self.current_text = text
        self.current_speaker = speaker
        self.interrupt_event.clear()
        self.convo_history.append(f"{speaker}: {text}")
        self.current_task = asyncio.create_task(self._process(text, speaker, t0, self.generation))

    # ── Audio helpers ─────────────────────────────────────────────────────────

    async def _stop_all_audio(self):
        """Stop audio in both streaming and fallback mode."""
        if self._streaming_mode:
            # Streaming mode: only clear AudioWorklet buffer via WebSocket
            # Do NOT call speaker.stop_audio() — the Recall.ai DELETE kills the output media pipeline
            try:
                await self.audio_ws.send_str(json.dumps({"type": "stop"}))
            except Exception:
                pass
        else:
            # Fallback mode: stop MP3 injection via Recall.ai API
            try:
                await self.speaker.stop_audio()
            except Exception:
                pass

    async def _stream_and_relay(self, text: str, my_gen: int) -> float:
        """Stream TTS via Cartesia WebSocket → relay PCM to audio page.
        Returns duration in seconds. Used in streaming mode."""
        total_bytes = 0
        t0 = time.time()
        try:
            async for pcm_chunk in self.speaker._stream_tts(text):
                if self.interrupt_event.is_set() or my_gen != self.generation:
                    return 0
                await self.audio_ws.send_bytes(pcm_chunk)
                total_bytes += len(pcm_chunk)
            # Send flush to let AudioWorklet know this utterance is done
            await self.audio_ws.send_str(json.dumps({"type": "flush"}))
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Stream relay error: {e}")
            return 0

        duration = total_bytes / PCM_BYTES_PER_SEC
        self._current_audio_duration = duration
        print(f"[{ts()}] {self.tag} ⏱ Streamed {total_bytes} bytes ({duration:.1f}s) in {elapsed(t0)}")
        return duration

    async def _wait_for_playback(self, duration_sec: float, my_gen: int) -> bool:
        """Wait for audio playback to finish, interruptible."""
        if duration_sec <= 0:
            return True
        self.audio_playing = True
        try:
            # Add 200ms for the AudioWorklet buffer delay
            await asyncio.wait_for(self.interrupt_event.wait(), timeout=duration_sec + 0.2)
            self.audio_playing = False
            return False  # interrupted
        except asyncio.TimeoutError:
            self.audio_playing = False
            return True  # completed

    async def _speak_streaming(self, text: str, my_gen: int) -> bool:
        """Stream TTS + wait for playback. Returns True if completed, False if interrupted."""
        duration = await self._stream_and_relay(text, my_gen)
        if duration <= 0:
            return False
        return await self._wait_for_playback(duration, my_gen)

    async def _speak_fallback(self, text: str, label: str, my_gen: int) -> bool:
        """Fallback: REST TTS + MP3 inject. Returns True if completed."""
        try:
            async with self.server._tts_semaphore:
                audio = await self.speaker._synthesise(text)
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Fallback TTS error: {e}")
            return True

        if self.interrupt_event.is_set() or my_gen != self.generation:
            return False

        try:
            await self.speaker.stop_audio()
        except Exception:
            pass
        b64 = base64.b64encode(audio).decode("utf-8")
        await self.speaker._inject_into_meeting(b64)
        self.audio_playing = True

        play_dur = max(500, get_duration_ms(audio))
        try:
            await asyncio.wait_for(self.interrupt_event.wait(), timeout=play_dur / 1000)
            self.audio_playing = False
            return False
        except asyncio.TimeoutError:
            self.audio_playing = False
            return True

    async def _speak(self, text: str, label: str, my_gen: int) -> bool:
        """Speak text using best available method. Returns True if completed."""
        if self._streaming_mode:
            return await self._speak_streaming(text, my_gen)
        else:
            return await self._speak_fallback(text, label, my_gen)

    async def _stream_pipelined(self, queue: asyncio.Queue, my_gen: int,
                                 cancel_task=None, extra_duration: float = 0,
                                 relay_start_override: float = 0) -> tuple:
        """Read sentences from queue, relay PCM back-to-back (no gap), wait at end.
        extra_duration: audio already in ring buffer (e.g. filler) to account for in final wait.
        relay_start_override: when the first audio (filler) was relayed, for accurate wait calculation.
        Returns (all_sentences: list, interrupted: bool)."""
        all_sentences = []
        relay_start = relay_start_override if relay_start_override > 0 else time.time()
        total_duration = extra_duration  # Include filler audio already playing

        while True:
            if self.interrupt_event.is_set() or my_gen != self.generation:
                if cancel_task:
                    cancel_task.cancel()
                return all_sentences, True

            try:
                item = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                break
            if item is None:
                break
            if item == "__FLUSH__":
                continue
            all_sentences.append(item)

            # Relay PCM to AudioWorklet — NO wait for playback (pipelined)
            duration = await self._stream_and_relay(item, my_gen)
            if duration <= 0:
                return all_sentences, True  # interrupted during relay
            total_duration += duration

        # Wait for remaining playback after all sentences relayed
        if total_duration > 0:
            elapsed_since_start = time.time() - relay_start
            remaining = total_duration - elapsed_since_start + 0.2  # 200ms AudioWorklet buffer
            if remaining > 0:
                self.audio_playing = True
                try:
                    await asyncio.wait_for(self.interrupt_event.wait(), timeout=remaining)
                    self.audio_playing = False
                    return all_sentences, True  # interrupted during final playback
                except asyncio.TimeoutError:
                    self.audio_playing = False

        return all_sentences, False  # completed normally

    async def _greet(self, name, t0):
        await asyncio.sleep(1.0)
        if self.speaking:
            return
        greeting = f"Hey {name}, welcome to the call!"
        self._log_sam(greeting)
        self.speaking = True
        try:
            await self._speak(greeting, "greeting", self.generation)
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Greet error: {e}")
        finally:
            self.speaking = False
            self.audio_playing = False

    async def _start_standup(self, developer_name: str):
        """Initialize and start the standup flow for a developer."""
        await asyncio.sleep(1.0)
        if self.speaking:
            return
        print(f"[{ts()}] {self.tag} 📋 Starting standup for {developer_name}")

        # Create standup flow with speaker function
        async def speak_fn(text, label, gen):
            self._log_sam(text)
            return await self._speak(text, label, gen)

        self.standup_flow = StandupFlow(
            developer_name=developer_name,
            agent=self.agent,
            speaker_fn=speak_fn,
            jira_client=self.jira if self.jira.enabled else None,
            jira_context=self._jira_context,
            azure_extractor=self.azure_extractor if self.azure_extractor.enabled else None,
        )

        # Connect buffer check so re-prompt can detect if user started speaking
        self.standup_flow._check_buffer_fn = lambda: bool(self._standup_buffer or self.partial_text)

        self.generation += 1
        self.speaking = True
        try:
            await self.standup_flow.start(self.generation)
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Standup start error: {e}")
        finally:
            self.speaking = False
            self.audio_playing = False

        # Start Flux STT if Deepgram key available
        if DEEPGRAM_API_KEY and not self._stt_task:
            await self._start_flux_stt(developer_name)

    # ── Flux STT (own Deepgram for standup) ──────────────────────────────────

    async def _start_flux_stt(self, developer_name: str):
        """Start Flux STT connection for standup. Receives audio_separate_raw for this developer.

        Flux params tuned for standup Q&A:
          eot_threshold=0.65      — fires EndOfTurn earlier on marginal-confidence turns
          eager_eot_threshold=0.35 — enables EagerEndOfTurn for speculative Groq pre-compute
          eot_timeout_ms=1500     — forces EndOfTurn after 1.5s silence (standup = short answers)
        """
        from stt_clients import stream_deepgram
        self._stt_queue = asyncio.Queue()
        self._flux_enabled = True
        self._flux_developer = developer_name
        self._speculative_task = None
        self._speculative_text = ""
        print(f"[{ts()}] {self.tag} 🎯 Starting Flux STT for {developer_name}")

        async def _flux_transcript_callback(text, is_final, sentiment=None):
            """Called by Flux for every transcript update."""
            if not text or not text.strip():
                return
            if is_final:
                # EndOfTurn — Flux confirmed user is done speaking
                project_key = self.jira.project if self.jira and self.jira.enabled else "SCRUM"
                clean_text = _convert_spoken_ticket_refs(text.strip(), project_key)
                print(f"[{ts()}] {self.tag} 🎯 Flux FINAL: \"{clean_text[:60]}\"")
                # Clear Nova-3's stale partial_text so silence-timer skip check is accurate
                self.partial_text = ""
                self.partial_speaker = ""
                # Clear Flux interim — turn processed, speech_off debounce won't re-finalize
                self._flux_last_interim_text = ""
                # Cancel any pending speech_off debounce — Flux already delivered the FINAL
                if self._flux_speech_off_task and not self._flux_speech_off_task.done():
                    self._flux_speech_off_task.cancel()
                self._standup_buffer.append(clean_text)
                # Process — if speculative Groq result is cached, handle() uses it
                await self._process_standup_buffer(self._flux_developer)
            else:
                # Interim update — user still speaking
                print(f"[{ts()}] {self.tag} 🎯 Flux interim: \"{text.strip()[:60]}\"")
                # Track latest interim so silence monitor can decide whether to Finalize
                self._flux_last_interim_text = text.strip()

                # Fast interrupt: if re-prompt is playing and user has spoken 2+ words,
                # stop the re-prompt immediately (user is finally answering)
                if (self.standup_flow
                    and getattr(self.standup_flow, '_playing_reprompt', False)
                    and self.speaking
                    and self.audio_playing
                    and not self._partial_interrupted
                    and len(text.strip().split()) >= 2):
                    self._partial_interrupted = True
                    self._partial_interrupt_time = time.time()
                    print(f"[{ts()}] {self.tag} ⚡ Re-prompt interrupted via Flux interim: \"{text.strip()[:40]}\" — stopping audio")
                    try:
                        await self._stop_all_audio()
                    except Exception as e:
                        print(f"[{ts()}] {self.tag} ⚠️  Stop audio failed (non-fatal): {e}")
                    self.interrupt_event.set()
                    if self.current_task and not self.current_task.done():
                        self.current_task.cancel()
                    self.speaking = False
                    self.audio_playing = False

        async def _flux_eager_eot_callback(transcript, confidence):
            """EagerEndOfTurn — start speculative Groq classification.

            Flux fires this when confidence first crosses eager_eot_threshold.
            The transcript here will match the final EndOfTurn transcript (Flux guarantee).
            We pre-compute the Groq classify+ack so it's ready when EndOfTurn confirms.
            """
            if not self.standup_flow or self.standup_flow.is_done:
                return
            project_key = self.jira.project if self.jira and self.jira.enabled else "SCRUM"
            clean = _convert_spoken_ticket_refs(transcript.strip(), project_key)
            self._speculative_text = clean
            # Cancel any previous speculative task
            if self._speculative_task and not self._speculative_task.done():
                self._speculative_task.cancel()
            # Fire-and-forget: pre-compute Groq classification
            self._speculative_task = asyncio.create_task(
                self._run_speculative_classify(clean))
            print(f"[{ts()}] {self.tag} ⚡ EagerEOT (conf={confidence:.2f}) — speculative Groq started")

        async def _flux_turn_resumed_callback():
            """TurnResumed — user kept speaking. Cancel speculative processing."""
            print(f"[{ts()}] {self.tag} 🔄 TurnResumed — cancelling speculative Groq")
            if self._speculative_task and not self._speculative_task.done():
                self._speculative_task.cancel()
            self._speculative_text = ""
            if self.standup_flow:
                self.standup_flow.clear_cached_result()

        async def _flux_end_of_turn_callback(confidence):
            """Called when Flux fires EndOfTurn with confidence score."""
            print(f"[{ts()}] {self.tag} 🎯 Flux EndOfTurn (confidence={confidence:.2f})")

        # Build keyword list for Flux (fixes "Scrub" → "SCRUM" transcription)
        keywords = ["AnavClouds", "Salesforce", "Sam"]
        if self.jira and self.jira.enabled and self.jira.project:
            keywords.append(self.jira.project)

        async def _run_flux():
            try:
                await stream_deepgram(
                    audio_queue=self._stt_queue,
                    transcript_callback=_flux_transcript_callback,
                    api_key=DEEPGRAM_API_KEY,
                    model="flux-general-en",
                    sample_rate=16000,
                    keywords=keywords,
                    end_of_turn_callback=_flux_end_of_turn_callback,
                    eager_eot_callback=_flux_eager_eot_callback,
                    turn_resumed_callback=_flux_turn_resumed_callback,
                    eot_threshold=0.65,
                    eager_eot_threshold=0.35,
                    eot_timeout_ms=1500,
                )
            except Exception as e:
                print(f"[{ts()}] {self.tag} ⚠️  Flux STT error: {e}")
            finally:
                self._flux_enabled = False
                print(f"[{ts()}] {self.tag} 🎯 Flux STT ended")

        self._stt_task = asyncio.create_task(_run_flux())

    async def _run_speculative_classify(self, text: str):
        """Pre-compute Groq classify+ack during EagerEndOfTurn→EndOfTurn window.

        Called as fire-and-forget task. If completed before EndOfTurn fires,
        the cached result is used by standup_flow.handle() — saving ~200-300ms.
        If TurnResumed fires first, this task is cancelled and cache cleared.
        """
        try:
            if not self.standup_flow or self.standup_flow.is_done:
                return
            result = await self.standup_flow.pre_classify(text)
            # Only cache if the transcript still matches (not cancelled/replaced)
            if result and self._speculative_text == text:
                self.standup_flow.set_cached_result(result, text)
                print(f"[{ts()}] {self.tag} ⚡ Speculative Groq cached: \"{result[:50]}\"")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Non-fatal — EndOfTurn will fall back to normal Groq call
            print(f"[{ts()}] {self.tag} ⚠️  Speculative classify failed (non-fatal): {e}")

    async def _stop_flux_stt(self):
        """Stop Flux STT connection and clean up speculative state."""
        # Cancel speculative task
        if self._speculative_task and not self._speculative_task.done():
            self._speculative_task.cancel()
        self._speculative_task = None
        self._speculative_text = ""
        # Close Flux connection
        if self._stt_queue:
            await self._stt_queue.put(None)  # Sentinel to close Deepgram
        if self._stt_task and not self._stt_task.done():
            self._stt_task.cancel()
            try:
                await self._stt_task
            except (asyncio.CancelledError, Exception):
                pass
        # Cancel speech_off debounce timer if pending
        if self._flux_speech_off_task and not self._flux_speech_off_task.done():
            self._flux_speech_off_task.cancel()
        self._flux_speech_off_task = None
        self._stt_task = None
        self._stt_queue = None
        self._flux_enabled = False
        self._flux_audio_buf = b""
        self._flux_last_interim_text = ""

    async def _flux_debounce_finalize(self):
        """Wait for debounce window, then convert Flux's pending interim text to FINAL.

        Called when Recall fires speech_off for the standup participant. If user resumes
        speaking within debounce window, speech_on cancels this task.

        This handles the case where Flux's native silence-based EOT won't fire because
        the user muted their mic (no audio = no silence detection). speech_off tells us
        the user stopped producing audio regardless of mute state.
        """
        try:
            debounce_s = self._FLUX_SPEECH_OFF_DEBOUNCE_MS / 1000.0
            await asyncio.sleep(debounce_s)

            # Debounce window passed — user genuinely stopped. Promote interim to FINAL.
            if not self._flux_enabled or not self.standup_flow or self.standup_flow.is_done:
                return
            interim = self._flux_last_interim_text
            if not interim:
                return  # no pending interim — nothing to finalize

            project_key = self.jira.project if self.jira and self.jira.enabled else "SCRUM"
            clean_text = _convert_spoken_ticket_refs(interim, project_key)
            print(f"[{ts()}] {self.tag} 🔇 speech_off debounce ({self._FLUX_SPEECH_OFF_DEBOUNCE_MS}ms) — promoting interim to FINAL: \"{clean_text[:60]}\"")

            # Clear interim and any speculative state — same as Flux FINAL handler
            self._flux_last_interim_text = ""
            self.partial_text = ""
            self.partial_speaker = ""

            # Feed into the same processing path as Flux's own FINAL
            self._standup_buffer.append(clean_text)
            await self._process_standup_buffer(self.standup_flow.developer)
        except asyncio.CancelledError:
            # speech_on fired within debounce — user resumed, no action needed
            pass
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  speech_off debounce error: {e}")

    async def _process_standup_buffer(self, speaker: str):
        """Process standup buffer immediately (called by Flux EndOfTurn or timer fallback)."""
        if not self._standup_buffer or not self.standup_flow or self.standup_flow.is_done:
            return

        # Wait for Sam to finish speaking (max 2s)
        for _ in range(20):
            if not self.speaking:
                break
            await asyncio.sleep(0.1)

        # Wait for previous handle() to finish (max 15s)
        for _ in range(150):
            if not self.standup_flow._processing:
                break
            await asyncio.sleep(0.1)

        if not self._standup_buffer:
            return

        full_text = " ".join(self._standup_buffer)
        self._standup_buffer.clear()

        # Pre-convert spoken ticket references
        project_key = self.jira.project if self.jira and self.jira.enabled else "SCRUM"
        full_text = _convert_spoken_ticket_refs(full_text, project_key)

        print(f"[{ts()}] {self.tag} 📋 Standup input: {full_text[:80]}")

        # Clear any stale interrupt
        self.interrupt_event.clear()

        self.generation += 1
        self.speaking = True
        try:
            still_active = await self.standup_flow.handle(full_text, speaker, self.generation)
            if not still_active:
                await self._finish_standup()
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Standup error: {e}")
        finally:
            self.speaking = False

    async def _finish_standup(self):
        """Save basic standup + leave immediately + background extract/Jira."""
        if self._standup_finished:
            return
        if not self.standup_flow:
            return
        self._standup_finished = True

        # Stop Flux STT
        await self._stop_flux_stt()

        # 1. Save basic standup data (raw answers, before extraction)
        result = self.standup_flow.get_result()
        result["session_id"] = self.session_id
        result["user"] = self.username
        result["mode"] = "standup"
        session_store.save_standup(result)
        print(f"[{ts()}] {self.tag} ✅ Standup saved (basic)")

        # 2. Leave meeting immediately (2 second pause for last audio)
        if self.standup_flow.is_done:
            asyncio.create_task(self._auto_leave_after_standup())

        # 3. Background: Azure extraction + Jira (fire and forget)
        if self.standup_flow.data.get("completed"):
            asyncio.create_task(self._background_standup_work())

    async def _auto_leave_after_standup(self):
        """Leave meeting 2 seconds after standup completes."""
        if self._auto_left:
            return
        self._auto_left = True
        try:
            await asyncio.sleep(2.0)
            print(f"[{ts()}] {self.tag} 🚪 Auto-leaving after standup")
            if self.bot_id:
                import httpx
                RECALL_REGION = os.environ.get("RECALLAI_REGION", "ap-northeast-1")
                RECALL_API_BASE = f"https://{RECALL_REGION}.recall.ai/api/v1"
                headers = {
                    "Authorization": f"Token {os.environ['RECALLAI_API_KEY']}",
                    "Content-Type": "application/json",
                }
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(f"{RECALL_API_BASE}/bot/{self.bot_id}/leave_call/", headers=headers)
                print(f"[{ts()}] {self.tag} ✅ Bot left meeting")
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Auto-leave failed: {e}")

    async def _background_standup_work(self):
        """Background: Azure extraction + Jira comments + transitions.
        Runs AFTER bot leaves. User is already gone."""
        try:
            # Wait a moment for bot to leave cleanly
            await asyncio.sleep(3.0)

            print(f"[{ts()}] {self.tag} 🔧 Background: processing standup data...")
            await self.standup_flow.background_finalize()

            # Re-save with enriched data (summaries, jira_ids, status_updates)
            result = self.standup_flow.get_result()
            result["session_id"] = self.session_id
            result["user"] = self.username
            result["mode"] = "standup"
            session_store.save_standup(result)
            print(f"[{ts()}] {self.tag} ✅ Standup re-saved (enriched)")

        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Background standup work failed: {e}")

    async def _flush_standup_buffer(self, speaker: str):
        """Simple silence-based flush: 1.2s timer restarts on new text.
        Fast path for tickets/blockers/confirmations. No fillers, no EOT."""
        try:
            # ── Step 1: Wait 1.2 seconds of silence ──
            await asyncio.sleep(1.2)

            if not self._standup_buffer:
                return

            # ── Step 2: Wait for Sam to finish speaking (max 2s) ──
            for _ in range(20):
                if not self.speaking:
                    break
                await asyncio.sleep(0.1)

            # ── Step 3: Wait for previous handle() to finish (max 15s) ──
            for _ in range(150):
                if not self.standup_flow._processing:
                    break
                await asyncio.sleep(0.1)

            if not self._standup_buffer:
                return

            # ── Step 4: Process entire buffer ──
            full_text = " ".join(self._standup_buffer)
            self._standup_buffer.clear()

            # Pre-convert spoken ticket references
            project_key = self.jira.project if self.jira and self.jira.enabled else "SCRUM"
            full_text = _convert_spoken_ticket_refs(full_text, project_key)

            print(f"[{ts()}] {self.tag} 📋 Standup input: {full_text[:80]}")

            # Clear any stale interrupt
            self.interrupt_event.clear()

            self.generation += 1
            self.speaking = True
            try:
                still_active = await self.standup_flow.handle(full_text, speaker, self.generation)
                if not still_active:
                    await self._finish_standup()
            except Exception as e:
                print(f"[{ts()}] {self.tag} ⚠️  Standup error: {e}")
            finally:
                self.speaking = False

            # After processing, check if more text arrived during processing
            if self._standup_buffer and self.standup_flow and not self.standup_flow.is_done:
                self._standup_timer = asyncio.create_task(self._flush_standup_buffer(speaker))

        except asyncio.CancelledError:
            pass

    def _log_sam(self, text):
        self.convo_history.append(f"Sam: {text}")
        self.agent.log_exchange("Sam", text)
        print(f"[{ts()}] {self.tag} 🗣️ Sam: {text[:100]}")

    async def _play_interrupt_ack(self):
        if not self.server._interrupt_ack_audio:
            return
        self.interrupt_event.clear()
        self.generation += 1
        self.speaking = True
        self.playing_ack = True
        try:
            text, audio = random.choice(self.server._interrupt_ack_audio)
            # For acks, always use fallback (pre-baked MP3, instant)
            await asyncio.sleep(0.5)  # Wait for Recall to fully stop previous audio
            b64 = base64.b64encode(audio).decode("utf-8")
            await self.speaker._inject_into_meeting(b64)
            self.audio_playing = True
            play_dur = get_duration_ms(audio)
            try:
                await asyncio.wait_for(self.interrupt_event.wait(), timeout=play_dur / 1000)
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Ack error: {e}")
        finally:
            self.speaking = False
            self.audio_playing = False
            self.playing_ack = False

    # ── Background search ─────────────────────────────────────────────────────

    async def _search_and_speak(self, text, context, my_gen):
        summary = await self.agent.search_and_summarize(text, context)
        sentences = self.agent._split_sentences(summary)
        for sent in sentences:
            if self.interrupt_event.is_set() or my_gen != self.generation:
                return ""
            await self._speak(sent, "search", my_gen)
        return " ".join(sentences)

    # ── Unified Research Flow (Feature 4) ─────────────────────────────────────

    async def _unified_research_flow(self, user_text: str, context: str, my_gen: int,
                                       filler_duration: float, filler_relay_start: float):
        """SerpAPI-direct research path with unified LLM planning.

        Flow:
          1. Generate unified research plan (Groq 70b, ~600-900ms)
          2. Based on plan:
             - jira_status type → bail out, use legacy path (needs synthesis)
             - Otherwise → gather tickets (cached + fresh if needed)
          3. Build project context (feature descriptions)
          4. Call SerpAPI-direct with persona + context + TTS hint
          5. Clean output + split sentences
          6. Stream to TTS via same pipelined path as legacy

        Returns:
          (all_sentences, interrupted) tuple if successful,
          None if any step failed (caller falls back to legacy synthesis).

        Safety: any failure at any step returns None → legacy fallback kicks in.
        """
        import time as _t

        # Step 1: Unified research planner (ONE Groq 70b call)
        plan = await self.agent.generate_unified_research_plan(
            user_text=user_text,
            context=context,
            ticket_cache=self._ticket_cache,
            memory_header=self._memory_header,
        )

        if not plan:
            # Planner failed — fall back to legacy
            return None

        # Step 2: Check if this is a Jira-status query (needs synthesis, not SerpAPI-direct)
        qtype = plan.get("question_type", "general")
        if qtype == "jira_status":
            # Pure Jira query — legacy path handles these better (Jira action + synthesis)
            print(f"[{ts()}] {self.tag} 🎯 Unified plan: jira_status → deferring to legacy")
            return None

        # HALLUCINATION GUARD: internal_org questions (who is our CEO, how many
        # employees, etc.) must NEVER go through SerpAPI-direct. Google AI Mode
        # has no reliable internal company data and will confidently make things
        # up. Defer to legacy Azure synthesis, which can honestly say "I don't
        # have that specific information" instead.
        if qtype == "internal_org":
            print(f"[{ts()}] {self.tag} 🛡️  Unified plan: internal_org → deferring to legacy (hallucination guard)")
            return None

        # Decision: Only use SerpAPI-direct when the question has project relevance.
        # General knowledge questions ("Who built the Taj Mahal?") don't benefit from
        # our project context wrapper and often return no AI Overview. Let Azure
        # synthesis handle them via the legacy path.
        cached_refs = plan.get("relevant_cached_tickets", [])
        needs_fresh = plan.get("needs_fresh_jira", False)
        features = plan.get("serpapi_context_features", [])

        has_project_relevance = bool(cached_refs) or needs_fresh or (
            qtype in ("feasibility", "tech_switch", "best_practices")
        )

        if qtype == "general" and not has_project_relevance:
            print(f"[{ts()}] {self.tag} 🎯 Unified plan: general knowledge (no project relevance) → deferring to legacy")
            return None

        if self.interrupt_event.is_set() or my_gen != self.generation:
            return None

        # Step 3: Gather tickets (cached + fresh if needed)
        relevant_tickets = []
        cached_keys = set(plan.get("relevant_cached_tickets", []))
        if cached_keys:
            for t in self._ticket_cache:
                if t.get("key") in cached_keys:
                    relevant_tickets.append(t)

        # Fresh Jira fetch if planner says cache is missing relevant work
        if plan.get("needs_fresh_jira") and plan.get("jira_search_terms"):
            if self.jira and self.jira.enabled:
                try:
                    fresh = await asyncio.wait_for(
                        self.jira.search_text(plan["jira_search_terms"], max_results=5),
                        timeout=2.5,  # budget cap — filler covers this
                    )
                    if fresh:
                        # Add to ticket_cache and relevant_tickets, avoiding duplicates
                        existing_keys = {t.get("key") for t in self._ticket_cache}
                        for ft in fresh:
                            if ft.get("key") and ft["key"] not in existing_keys:
                                self._ticket_cache.append(ft)
                                existing_keys.add(ft["key"])
                            if ft not in relevant_tickets:
                                relevant_tickets.append(ft)
                        self._rebuild_jira_context()
                        print(f"[{ts()}] {self.tag} 🎯 Fresh Jira fetch: +{len(fresh)} ticket(s)")
                except asyncio.TimeoutError:
                    print(f"[{ts()}] {self.tag} ⏱ Fresh Jira fetch timeout (>2.5s) — continuing without")
                except Exception as e:
                    print(f"[{ts()}] {self.tag} ⚠️  Fresh Jira fetch failed: {e}")

        if self.interrupt_event.is_set() or my_gen != self.generation:
            return None

        # Step 4: Build project context for SerpAPI
        feature_descriptions = plan.get("serpapi_context_features", [])
        project_context = self.agent._build_project_context_from_tickets(
            tickets=relevant_tickets if relevant_tickets else self._ticket_cache[:10],
            feature_descriptions=feature_descriptions,
        )

        # Length hint based on question type
        length_hint = {
            "feasibility": "2-3 sentences",
            "tech_switch": "3-4 sentences",
            "best_practices": "2-3 sentences",
            "general": "2-3 sentences",
        }.get(qtype, "2-3 sentences")

        # Step 5: SerpAPI-direct call
        response_text = await self.agent.serpapi_direct_research(
            user_text=user_text,
            project_context=project_context,
            conversation=context,
            length=length_hint,
        )

        if not response_text:
            # SerpAPI-direct failed/empty — fall back to legacy
            return None

        if self.interrupt_event.is_set() or my_gen != self.generation:
            return None

        # Step 6: Split into sentences and stream to TTS pipeline
        sentences = self.agent._split_sentences(response_text)
        if not sentences:
            return None

        # Use same pipelined streaming as legacy path for consistent behavior
        research_queue: asyncio.Queue = asyncio.Queue()

        async def _fill_queue():
            try:
                for s in sentences:
                    if self.interrupt_event.is_set() or my_gen != self.generation:
                        break
                    await research_queue.put(s)
            finally:
                await research_queue.put(None)

        filler_stream_task = asyncio.create_task(_fill_queue())

        if self._streaming_mode:
            all_sentences, interrupted = await self._stream_pipelined(
                research_queue, my_gen, cancel_task=filler_stream_task,
                extra_duration=filler_duration,
                relay_start_override=filler_relay_start)
            return (all_sentences, interrupted)
        else:
            # Non-streaming fallback: speak each sentence
            spoken: list = []
            while True:
                try:
                    item = await asyncio.wait_for(research_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    break
                spoken.append(item)
            for sent in spoken:
                if self.interrupt_event.is_set() or my_gen != self.generation:
                    return (spoken, True)
                await self._speak(sent, "research", my_gen)
            return (spoken, False)

    # ── Jira handler ──────────────────────────────────────────────────────────

    async def _handle_jira_read(self, text, context, my_gen):
        try:
            context_block = ""
            if context:
                lines = [l for l in context.strip().split('\n') if l.strip()][-3:]
                if lines:
                    context_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

            t0 = time.time()
            response = await self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=[{"role": "system", "content": JIRA_INTENT_PROMPT.format(project_key=self.jira.project, text=text, context_block=context_block)}, {"role": "user", "content": text}],
                temperature=0.0, max_tokens=30,
            )
            intent = response.choices[0].message.content.strip()
            print(f"[{ts()}] {self.tag} 🎫 Intent: \"{intent}\" ({(time.time()-t0)*1000:.0f}ms)")

            if intent == "MY_TICKETS":
                tickets = await self.jira.get_my_tickets()
                return {"tickets": tickets, "count": len(tickets)}
            elif intent == "SPRINT_STATUS":
                return await self.jira.get_sprint_status()
            elif intent.startswith("TICKET:"):
                ids = _re.findall(r'[A-Z]+-\d+', intent.split(":", 1)[1])
                if not ids:
                    return {"tickets": await self.jira.get_my_tickets(), "count": 0}
                if len(ids) == 1:
                    return await self.jira.get_ticket(ids[0])
                results = []
                for tid in ids[:5]:
                    try:
                        results.append(await self.jira.get_ticket(tid))
                    except Exception:
                        pass
                return {"tickets": results, "count": len(results)}
            elif intent.startswith("TRANSITION:"):
                parts = intent.split(":")
                if len(parts) >= 3:
                    tid, status = parts[1].strip(), parts[2].strip()
                    if not _re.match(r'^[A-Z]+-\d+$', tid):
                        return {"error": f"Invalid ID: {tid}"}
                    try:
                        r = await self.jira.transition_ticket(tid, status)
                        if r.get("action") == "already_done":
                            return {"action": "already_done", "ticket": tid, "message": f"{tid} already at '{r['already_at']}'."}
                        return {"action": "transition", "ticket": tid, "new_status": r["new_status"]}
                    except JiraTransitionError as e:
                        return {"action": "transition_error", "ticket": tid, "error": str(e)}
            elif intent.startswith("SEARCH:"):
                q = intent.split(":", 1)[1].strip()
                tickets = await self.jira.search_text(q, max_results=5)
                return {"tickets": tickets, "count": len(tickets)} if tickets else {"error": f"No tickets for '{q}'."}
            elif intent.startswith("CREATE:"):
                # OPTION C: DEFERRED CREATION
                # Don't create the ticket now. Azure at meeting-end will create
                # high-quality tickets with full transcript context. We just
                # record the user's intent here and return an acknowledgment.
                #
                # Why: mid-meeting creation often produces garbage titles
                # ("Ticket Optimization" when user meant 4 specific tickets).
                # Azure sees full discussion and extracts proper titles + counts.
                summary_hint = intent.split(":", 1)[1].strip()
                if not summary_hint:
                    return {"error": "No summary hint provided"}

                try:
                    # Record the user's intent for Azure to process at meeting-end
                    self._pending_creation_intents.append({
                        "user_said": text,
                        "extracted_summary": summary_hint,
                        "at_time": time.strftime("%H:%M:%S", time.gmtime()),
                    })
                    count = len(self._pending_creation_intents)
                    print(f"[{ts()}] {self.tag} 📝 Creation intent #{count} recorded: "
                          f"\"{summary_hint}\" (will be created post-meeting)")

                    # Return acknowledgment data that the research stream uses
                    # to craft a natural "I'll log that" response. The specific
                    # wording is in RESEARCH_PROMPT via Sam's natural-language LLM.
                    return {
                        "action": "creation_deferred",
                        "user_request": text,
                        "intent_summary": summary_hint,
                        "pending_count": count,
                        "note": (
                            "Ticket will be created after the meeting with full "
                            "context. Acknowledge naturally without inventing a "
                            "ticket ID. Say you'll log it post-meeting."
                        ),
                    }
                except Exception as e:
                    print(f"[{ts()}] {self.tag} ⚠️  Intent recording failed: {e}")
                    return {"action": "create_failed", "error": str(e)}
            else:
                return {"tickets": await self.jira.get_my_tickets(), "count": 0}
        except Exception as e:
            print(f"[{ts()}] {self.tag} ⚠️  Jira error: {e}")
            return {"error": str(e)}

    # ── Main pipeline ─────────────────────────────────────────────────────────

    async def _process(self, text, speaker, t0, generation=0):
        if self.speaking:
            return
        self.speaking = True
        self.interrupt_event.clear()
        my_gen = generation
        is_direct = self._is_direct_address(text)

        try:
            context = "\n".join(self.convo_history)
            t1 = time.time()
            mode = "streaming" if self._streaming_mode else "fallback"
            print(f"[{ts()}] {self.tag} 🔊 Mode: {mode}")

            # ── Trigger: skip for direct address ──────────────────────
            if is_direct:
                print(f"  ⚡ Direct address — trigger skipped")
                should = True
            else:
                trigger_task = asyncio.create_task(
                    self.trigger.should_respond(text, speaker, context,
                                                [e["text"] for e in self.agent.rag._entries[-20:]]))
                should = await trigger_task

            if not should:
                return

            # ── Speculative execution: start router + PM LLM + dynamic filler in parallel ──
            router_task = asyncio.create_task(self.agent._route(text, context))

            # Speculatively start PM LLM (80% of queries go to PM)
            speculative_queue = asyncio.Queue()
            speculative_llm = asyncio.create_task(
                self.agent.stream_sentences_to_queue(
                    text, context, speculative_queue,
                    memory_context=self._memory_full,
                ))

            # Dynamic filler task (only consumed if route=RESEARCH).
            # Runs in parallel with router — by the time route is decided, filler
            # is usually ready too. Falls back to hardcoded FILLERS if not ready,
            # times out, or produces invalid output. Cancelled if route=PM.
            filler_task = None
            if USE_DYNAMIC_FILLERS:
                filler_task = asyncio.create_task(
                    self.agent.generate_dynamic_filler(text))

            route = await router_task
            print(f"[{ts()}] {self.tag} Route: [{route}] ({elapsed(t1)})")

            # ── [RESEARCH] — cancel speculative LLM, parallel research, Azure stream ──
            if route == "RESEARCH":
                speculative_llm.cancel()
                try:
                    await speculative_llm
                except (asyncio.CancelledError, Exception):
                    pass

                # ── Dynamic filler: use if ready + valid, else fall back ──
                self.searching = True
                dynamic_filler_text = None
                if filler_task is not None:
                    if filler_task.done():
                        try:
                            result = filler_task.result()
                            if result:  # None = generation failed/invalid
                                dynamic_filler_text = result
                        except (asyncio.CancelledError, Exception):
                            pass
                    else:
                        # Not ready — don't wait for it (defeats the purpose)
                        filler_task.cancel()
                        try:
                            await filler_task
                        except (asyncio.CancelledError, Exception):
                            pass

                filler = dynamic_filler_text if dynamic_filler_text else random.choice(FILLERS)
                if dynamic_filler_text:
                    print(f"[{ts()}] {self.tag} 🎨 Dynamic filler")
                else:
                    print(f"[{ts()}] {self.tag} 📋 Hardcoded filler")

                filler_relay_start = time.time()
                if self._streaming_mode:
                    filler_duration = await self._stream_and_relay(filler, my_gen)
                    if filler_duration <= 0 or self.interrupt_event.is_set():
                        return
                else:
                    await self._speak(filler, "research-filler", my_gen)
                    filler_duration = 0
                    filler_relay_start = 0
                    if self.interrupt_event.is_set():
                        return

                # Research starts NOW — while filler is still playing
                import time as _time
                t_research = _time.time()

                # ═══════════════════════════════════════════════════════════════
                # UNIFIED RESEARCH FLOW (Feature 4)
                # Toggle: USE_UNIFIED_RESEARCH env var. If False/fails, falls back
                # to legacy parallel Jira+web+Azure flow below.
                # ═══════════════════════════════════════════════════════════════
                used_unified = False
                all_sentences: list = []
                interrupted = False

                if USE_UNIFIED_RESEARCH:
                    try:
                        unified_response = await self._unified_research_flow(
                            user_text=text,
                            context=context,
                            my_gen=my_gen,
                            filler_duration=filler_duration,
                            filler_relay_start=filler_relay_start,
                        )
                        if unified_response is not None:
                            all_sentences, interrupted = unified_response
                            used_unified = True
                            research_ms = (_time.time() - t_research) * 1000
                            print(f"[{ts()}] {self.tag} 🔬 Unified research: {research_ms:.0f}ms")
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print(f"[{ts()}] {self.tag} ⚠️  Unified research failed, falling back: {e}")

                # ── Legacy path: Jira action + web search + Azure synthesis ──
                if not used_unified:
                    async def _jira_action():
                        """Detect and execute Jira operations. Updates cache after writes."""
                        if not self.jira or not self.jira.enabled:
                            return "(no Jira)"
                        try:
                            result = await self._handle_jira_read(text, context, my_gen)
                            if not result:
                                return "(no action needed)"

                            if isinstance(result, dict):
                                action = result.get("action", "")
                                if action == "transition":
                                    tid = result.get("ticket", "")
                                    new_status = result.get("new_status", "")
                                    for t in self._ticket_cache:
                                        if t['key'] == tid:
                                            t['status'] = new_status
                                            break
                                    self._rebuild_jira_context()
                                elif "key" in result and result["key"] not in [t["key"] for t in self._ticket_cache]:
                                    self._update_ticket_cache(result)

                            return json.dumps(result, indent=2, default=str)[:800]
                        except Exception as e:
                            print(f"[{ts()}] {self.tag} ⚠️  Jira action: {e}")
                            return "(Jira action failed)"

                    async def _web_search():
                        try:
                            ticket_hint = self._get_ticket_context_for_search()
                            query = await self.agent._to_english_search_query(text, context, ticket_hint)
                            if query.upper().strip() == "SKIP":
                                print(f"[{ts()}] {self.tag} 🔍 Web search: SKIPPED (not needed)")
                                return "(web search skipped — not relevant for this query)"
                            results = await self.agent._get_web_search().search(query)
                            return results[:800] if results else "(no web results)"
                        except Exception as e:
                            return "(web search failed)"

                    action_task = asyncio.create_task(_jira_action())
                    web_task = asyncio.create_task(_web_search())

                    jira_action = await action_task
                    web_results = await web_task

                    research_ms = (_time.time() - t_research) * 1000
                    print(f"[{ts()}] {self.tag} 🔬 Legacy research: {research_ms:.0f}ms")

                    if self.interrupt_event.is_set() or my_gen != self.generation:
                        return

                    # Stream from Azure 4o-mini (pipelined — no gap between sentences)
                    research_queue = asyncio.Queue()
                    research_stream = asyncio.create_task(
                        self.agent.stream_research_to_queue(
                            user_text=text,
                            jira_context=self._jira_context,
                            related_tickets="(all tickets included in project context above)",
                            web_results=web_results,
                            jira_action=jira_action,
                            conversation=context,
                            azure_extractor=self.azure_extractor,
                            queue=research_queue,
                            memory_context=self._memory_full,
                        )
                    )

                    if self._streaming_mode:
                        all_sentences, interrupted = await self._stream_pipelined(
                            research_queue, my_gen, cancel_task=research_stream,
                            extra_duration=filler_duration,
                            relay_start_override=filler_relay_start)
                    else:
                        all_sentences = []
                        while True:
                            try:
                                item = await asyncio.wait_for(research_queue.get(), timeout=30.0)
                            except asyncio.TimeoutError:
                                break
                            if item is None:
                                break
                            all_sentences.append(item)
                        for sent in all_sentences:
                            await self._speak(sent, "research", my_gen)

                # ── Post-response handling (both paths) ──
                if interrupted and all_sentences:
                    self._log_sam(" ".join(all_sentences) + " [interrupted]")
                    self.trigger.mark_responded()
                    return

                full_text = " ".join(all_sentences)
                if full_text:
                    self._log_sam(full_text)
                    self.trigger.mark_responded()

            # ── [PM] — use speculative LLM (already running!) ──
            else:
                # Cancel dynamic filler task (not used in PM path)
                if filler_task is not None and not filler_task.done():
                    filler_task.cancel()
                    try:
                        await filler_task
                    except (asyncio.CancelledError, Exception):
                        pass
                # LLM has been running since before router finished
                # Sentences may already be in the queue

                if self._streaming_mode:
                    all_sentences, interrupted = await self._stream_pipelined(
                        speculative_queue, my_gen, cancel_task=speculative_llm)
                    if interrupted and all_sentences:
                        self._log_sam(" ".join(all_sentences) + " [interrupted]")
                        self.trigger.mark_responded()
                        return
                else:
                    # Fallback mode: collect all then speak
                    all_sentences = []
                    while True:
                        if self.interrupt_event.is_set() or my_gen != self.generation:
                            speculative_llm.cancel()
                            return
                        try:
                            item = await asyncio.wait_for(speculative_queue.get(), timeout=15.0)
                        except asyncio.TimeoutError:
                            break
                        if item is None:
                            break
                        if item == "__FLUSH__":
                            continue
                        all_sentences.append(item)

                    if all_sentences:
                        if len(all_sentences) == 1:
                            await self._speak(all_sentences[0], "single", my_gen)
                        else:
                            from pydub import AudioSegment as _AS
                            parts = []
                            for s in all_sentences:
                                try:
                                    async with self.server._tts_semaphore:
                                        parts.append(await self.speaker._synthesise(s))
                                except Exception:
                                    pass
                            if parts:
                                combined = parts[0] if len(parts) == 1 else self._combine_audio(parts)
                                b64 = base64.b64encode(combined).decode("utf-8")
                                try:
                                    await self.speaker.stop_audio()
                                except Exception:
                                    pass
                                await self.speaker._inject_into_meeting(b64)
                                self.audio_playing = True
                                dur = get_duration_ms(combined)
                                try:
                                    await asyncio.wait_for(self.interrupt_event.wait(), timeout=dur / 1000)
                                except asyncio.TimeoutError:
                                    pass
                                self.audio_playing = False

                if all_sentences:
                    print(f"[{ts()}] {self.tag} 📊 TOTAL: {elapsed(t0)}")
                    self._log_sam(" ".join(all_sentences))
                    self.trigger.mark_responded()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            import traceback
            print(f"[{ts()}] {self.tag} ❌ Error: {e}")
            traceback.print_exc()
        finally:
            self.audio_playing = False
            self.speaking = False
            self.searching = False

    def _combine_audio(self, audio_list):
        from pydub import AudioSegment
        import io
        combined = AudioSegment.empty()
        for ab in audio_list:
            combined += AudioSegment.from_file(io.BytesIO(ab), format="mp3")
        output = io.BytesIO()
        combined.export(output, format="mp3", bitrate="192k")
        return output.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# WebSocketServer
# ══════════════════════════════════════════════════════════════════════════════

class WebSocketServer:
    def __init__(self, port=8000):
        self.port = port
        self.sessions = {}
        self._tts_semaphore = asyncio.Semaphore(4)
        self._interrupt_ack_audio = []
        self.on_session_removed = None  # Callback: fn(session) — called when session is cleaned up
        self.debug_save_audio = os.environ.get("DEBUG_SAVE_AUDIO", "").lower() in ("1", "true", "yes")
        self.app = web.Application()
        self.app.router.add_get("/ws/{session_id}", self.handle_websocket)
        self.app.router.add_get("/audio/{session_id}", self.handle_audio_ws)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "sessions": len(self.sessions)})

    async def handle_websocket(self, request):
        """Recall.ai transcript/events WebSocket."""
        session_id = request.match_info.get("session_id", "")
        session = self.sessions.get(session_id)
        if not session:
            return web.Response(status=404)
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        print(f"[{ts()}] {session.tag} ✅ Recall.ai WebSocket connected")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await session.handle_event(msg.data)
                    except Exception as e:
                        print(f"[{ts()}] {session.tag} ⚠️  Event error: {e}")
                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
        except Exception as e:
            print(f"[{ts()}] {session.tag} WS error: {e}")
        finally:
            print(f"[{ts()}] {session.tag} WebSocket disconnected")
            await self.remove_session(session_id)
        return ws

    async def handle_audio_ws(self, request):
        """Output Media audio page WebSocket — receives PCM chunks to play."""
        session_id = request.match_info.get("session_id", "")
        session = self.sessions.get(session_id)
        if not session:
            return web.Response(status=404, text="Session not found")
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        session.audio_ws = ws
        print(f"[{ts()}] {session.tag} 🔊 Audio page connected (streaming mode ON)")
        try:
            async for msg in ws:
                pass  # Audio page doesn't send us data
        except Exception:
            pass
        finally:
            session.audio_ws = None
            print(f"[{ts()}] {session.tag} 🔇 Audio page disconnected (fallback mode)")
        return ws

    def create_session(self, session_id, bot_id):
        session = BotSession(session_id, bot_id, self)
        self.sessions[session_id] = session
        print(f"[{ts()}] 📦 Session created: {session_id[:12]}")
        return session

    async def remove_session(self, session_id):
        session = self.sessions.pop(session_id, None)
        if session:
            await session.cleanup()
            # Notify server.py to clean up active_bots
            if self.on_session_removed and session.username:
                try:
                    self.on_session_removed(session)
                except Exception as e:
                    print(f"[{ts()}] ⚠️  on_session_removed callback failed: {e}")
            print(f"[{ts()}] 🗑️  Session removed: {session_id[:12]}")

    async def start(self):
        print(f"[{ts()}] Pre-baking interrupt ack audio...")
        temp = CartesiaSpeaker(bot_id=None)
        await temp.warmup()
        for phrase in _INTERRUPT_ACKS:
            try:
                async with self._tts_semaphore:
                    audio = await temp._synthesise(phrase)
                self._interrupt_ack_audio.append((phrase, audio))
            except Exception as e:
                print(f"[{ts()}] ⚠️  Pre-bake failed: {e}")

        await temp.close()
        print(f"[{ts()}] ✅ {len(self._interrupt_ack_audio)} acks pre-baked")

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        print(f"[{ts()}] WebSocket ready on ws://0.0.0.0:{self.port}/ws/{{session_id}}")
        print(f"[{ts()}] Audio relay on ws://0.0.0.0:{self.port}/audio/{{session_id}}")
        print(f"[{ts()}] Health: http://localhost:{self.port}/health\n")