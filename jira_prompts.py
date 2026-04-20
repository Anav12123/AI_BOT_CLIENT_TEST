"""
jira_prompts.py — Jira integration prompts + Azure GPT-4o mini extraction

During meeting: JIRA_INTENT_PROMPT (Groq) classifies what Jira op to run
                JIRA_RESPONSE_PROMPT (Groq) formats Jira data for voice
Post-meeting:   EXTRACTION_PROMPT (Azure GPT-4o mini) extracts action items
"""

import os
import json
import httpx


# ── Prompt for Jira read responses (Groq, during meeting) ────────────────────

JIRA_RESPONSE_PROMPT = """You are Sam, a PM on a live voice call. You just looked up Jira ticket info.

Jira data:
{jira_data}

Rules:
- Speak the info naturally, like telling a coworker.
- If there are MORE than 5 tickets, read the top 5 by priority, then say "Want me to go through the rest?"
- If 5 or fewer, read all of them.
- For each ticket mention: ticket ID, summary, status, and assignee if assigned.
- Don't read raw JSON — summarize it conversationally.
- If a transition was successful, confirm it naturally.
- If a transition failed, explain what moves are available.
- If the ticket is already at the requested status, say so naturally.
- No markdown, no lists — this is voice output.
- Keep it under 30 seconds of speaking time."""


# ── Jira intent classifier (Groq, during meeting) ────────────────────────────

JIRA_INTENT_PROMPT = """You are a Jira intent classifier. Given a user message from a meeting, determine what Jira operation they want.

The Jira project key is: {project_key}

{context_block}User message: "{text}"

Classify into EXACTLY one of these formats:

MY_TICKETS — user wants to see their open/assigned tickets, any tickets in the project, available tickets
SPRINT_STATUS — user wants sprint overview, progress, how many done/in-progress
TICKET:ID — user wants details about a specific ticket (e.g. TICKET:SCRUM-123). For MULTIPLE tickets use comma: TICKET:SCRUM-15,SCRUM-12
TRANSITION:ID:STATUS — user wants to move a ticket to a new status (e.g. TRANSITION:SCRUM-123:Done). Status must be one of: Done, In Progress, To Do
CREATE:SUMMARY — user wants to CREATE a new ticket. Extract a short summary (e.g. CREATE:Code Review for Payment Module)
SEARCH:QUERY — user wants to find tickets by topic/description (e.g. SEARCH:login page bug)

Rules:
- SPOKEN NUMBERS: speech-to-text converts numbers to words. You MUST convert them back.
  "thirty one" = 31, "twenty six" = 26, "twelve" = 12, "forty five" = 45
  "ticket thirty one" = {project_key}-31
  "scrum twenty six" = {project_key}-26
  "move twelve to done" = TRANSITION:{project_key}-12:Done
  
- GARBLED SPEECH: speech-to-text often garbles "move" commands. Look for the INTENT, not exact words.
  "makes from thirty one to done" = TRANSITION:{project_key}-31:Done
  "get it to done" / "move it to done" = look at conversation for which ticket → TRANSITION:ID:Done
  "ticket number twenty six two done" = TRANSITION:{project_key}-26:Done

- CREATE DETECTION: if the user asks to "create", "make", "add", "open", "raise" a ticket/issue/task:
  "create a ticket for code review" = CREATE:Code Review
  "make a ticket for the login bug" = CREATE:Login Bug Fix
  "can you raise a ticket for this" (after discussing chatbot crash) = CREATE:Chatbot Crash
  "create a ticket for this" (after discussing payment API failure) = CREATE:Payment API Failure
  "log this as a ticket" (after discussing deployment issue) = CREATE:Deployment Issue
  IMPORTANT: For "this"/"that"/"it" references, look at recent conversation to identify the actual topic and use that as the summary. NEVER output the literal word "summary" — that's a placeholder, not a real title.

- If user says "define those tickets", "tell me more about those" — look at RECENT CONVERSATION for ticket IDs that were just mentioned and return TICKET:ID,ID
- If unsure, default to MY_TICKETS

Reply with ONLY the classification. Nothing else."""


# ── Prompt for post-meeting extraction (Azure GPT-4o mini) ───────────────────

EXTRACTION_PROMPT = """You are an AI assistant that extracts action items from meeting transcripts.

Analyze the following meeting transcript and extract all actionable items that need Jira tickets.

For each item, output a JSON object with these fields:
- "type": one of "Bug", "Story", "Task"
- "summary": short title (max 80 chars)
- "description": detailed description with context from the meeting
- "priority": "Highest", "High", "Medium", or "Low"
- "labels": list of relevant labels (see rules below)
- "assignee": person's name if someone was explicitly assigned this task, otherwise null

{pending_intents_block}

What counts as an action item:
- Bugs reported (anything broken, crashing, slow, not working)
- Feature requests (new functionality requested)
- Tasks assigned (someone asked to do something)
- Decisions made that need tracking
- Blockers mentioned

Rules:
- Only extract ACTIONABLE items, not general discussion
- ONLY extract items from what HUMAN participants said — IGNORE everything Sam said (lines starting with "Sam:")
- IGNORE any bot errors, connection failures, DNS issues, or technical problems with Sam himself — these are NOT action items
- If the user asked to "create a ticket" but the PENDING CREATION INTENTS section above is empty, then extract the underlying work being requested (e.g. the bug being discussed).
  Example: User says "the chatbot is crashing... create a ticket for this" → extract ONE item about the chatbot crash.
- Include meeting context in descriptions (who said what, why it matters)
- If someone was assigned a task, mention them in the description
- If no action items found, return empty list []
- Be specific in summaries — "Login page crash on Chrome Android" not "Bug fix needed"
- Labels must include "client-feedback" and "session-{meeting_date}" for all items
- For bugs also add "bug" label
- For feature requests add "feature-request" label
- For blockers add "blocker" label
- For urgent/critical items add "critical" label

HANDLING PLURAL REQUESTS:
If a pending intent says something like "create a ticket for each" or "tickets for all those issues",
look at the preceding discussion and create ONE ticket per distinct issue discussed.
Example: if user discussed query optimization, caching, data modeling, and design tweaks, then
said "create a ticket for each", output FOUR tickets — one per topic — not a single generic ticket.

Respond with ONLY a JSON array. No markdown, no explanation, no backticks.

Meeting date: {meeting_date}

Meeting transcript:
{transcript}

Extract action items as JSON array:"""


# ── Prompt for conversation-scoped session summary (Feature 4 Memory) ─────────

SESSION_SUMMARY_PROMPT = """You are summarizing a meeting for long-term memory.
Output STRUCTURED DATA that will be loaded in future meetings with the same group.

Attendees: {attendees}
Meeting date: {meeting_date}

Meeting transcript:
{transcript}

Output a JSON object with EXACTLY these fields:
{{
  "topics": ["..."],
  "decisions": ["..."],
  "commitments": [{{"who": "...", "what": "...", "when": "..."}}],
  "open_items": ["..."],
  "tickets_referenced": [{{"key": "SCRUM-123", "status_at_time": "In Progress"}}],
  "user_concerns": ["..."],
  "summary_text": "2-3 sentence narrative summary"
}}

Rules for accuracy (CRITICAL — this memory feeds future meetings):
- "decisions": ONLY items where the group actually DECIDED something. Options discussed but not decided = NOT a decision.
- "commitments": ONLY explicit commitments with a clear owner. "Someone should look at this" = NOT a commitment.
- "open_items": Things explicitly left unresolved. Not everything discussed.
- "tickets_referenced": Only Jira ticket keys actually mentioned (like SCRUM-162). Include status if mentioned in discussion.
- "user_concerns": What the user seemed worried about or focused on.
- "summary_text": Factual, not speculative. What actually happened.

CRITICAL — DO NOT INVENT:
- Do not add decisions that weren't made
- Do not attribute commitments to people who didn't make them
- Do not infer ticket statuses not mentioned in the transcript
- If the meeting was short/trivial and had nothing substantive, return empty lists
- If something was discussed but unclear, OMIT it rather than guess

If the meeting has no substantive content, return:
{{"topics": [], "decisions": [], "commitments": [], "open_items": [],
  "tickets_referenced": [], "user_concerns": [], "summary_text": "Brief meeting, no notable discussion."}}

Respond with ONLY the JSON object. No markdown, no explanation, no backticks."""


# ── Azure GPT-4o mini client ─────────────────────────────────────────────────

class AzureExtractor:
    """Post-meeting action item extractor using Azure OpenAI GPT-4o mini."""

    def __init__(self):
        self.endpoint   = os.environ.get("AZURE_ENDPOINT", "").strip().rstrip("/")
        self.api_key    = os.environ.get("AZURE_API_KEY", "").strip()
        self.deployment = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o-mini").strip()
        self.api_version = os.environ.get("AZURE_API_VERSION", "2024-02-15-preview").strip()

        if not self.endpoint or not self.api_key:
            print("[Azure] ⚠️  Missing AZURE_ENDPOINT or AZURE_API_KEY — post-meeting extraction disabled")
            self.enabled = False
            return

        self.enabled = True
        self._client = httpx.AsyncClient(timeout=60)
        print(f"[Azure] ✅ Configured: {self.endpoint} (deployment: {self.deployment})")

    async def extract_action_items(self, transcript: str, date_str: str = "",
                                    pending_intents: list | None = None) -> list[dict]:
        """Extract action items from a meeting transcript.

        pending_intents: OPTION C — list of explicit ticket creation intents
        the user requested mid-meeting. These are signals TO CREATE tickets
        (not meta-actions to ignore). Each dict should have:
          - "user_said": the raw utterance
          - "extracted_summary": the classifier's short-hint summary
          - "at_time": timestamp when said

        When pending_intents is non-empty, Azure treats them as explicit user
        requests and creates proper tickets with full context (better titles,
        plural handling, etc.) vs. what the mid-meeting classifier would
        produce from ambiguous speech alone.
        """
        if not self.enabled:
            print("[Azure] ⚠️  Extraction disabled — no Azure credentials")
            return []

        if not transcript or len(transcript.strip()) < 50:
            print("[Azure] ⚠️  Transcript too short — skipping extraction")
            return []

        import time
        if not date_str:
            date_str = time.strftime("%Y-%m-%d", time.gmtime())

        # Build the "pending intents" block for the prompt
        if pending_intents:
            lines = []
            for i, intent in enumerate(pending_intents, 1):
                user_said = intent.get("user_said", "?")
                hint = intent.get("extracted_summary", "?")
                at_time = intent.get("at_time", "?")
                lines.append(
                    f'  {i}. At {at_time}, user said: "{user_said}"\n'
                    f'     (Initial hint: "{hint}")'
                )
            pending_block = (
                "PENDING CREATION INTENTS — the user EXPLICITLY requested these "
                "tickets be created during the meeting. YOU MUST create proper "
                "Jira tickets for each intent below, using the FULL transcript "
                "for context to write accurate titles and descriptions.\n"
                "Do NOT skip any intent. Do NOT treat these as meta-actions.\n\n"
                + "\n".join(lines)
            )
            print(f"[Azure] 📝 Processing {len(pending_intents)} pending creation intent(s)")
        else:
            pending_block = (
                "(No explicit ticket creation intents were recorded during this "
                "meeting. Extract any ACTIONABLE items you find in the transcript "
                "according to the rules below.)"
            )

        prompt = EXTRACTION_PROMPT.format(
            transcript=transcript[:12000],
            meeting_date=date_str,
            pending_intents_block=pending_block,
        )

        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

        try:
            print(f"[Azure] 🤖 Extracting action items ({len(transcript)} chars transcript)...")
            t0 = time.time()

            response = None
            for attempt in range(3):
                try:
                    response = await self._client.post(
                        url,
                        headers={"api-key": self.api_key, "Content-Type": "application/json"},
                        json={
                            "messages": [
                                {"role": "system", "content": "You extract action items from meeting transcripts. Respond with JSON only."},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.2,
                            "max_tokens": 2000,
                        },
                    )
                    break
                except Exception as net_err:
                    if "getaddrinfo" in str(net_err) or "ConnectError" in str(net_err):
                        if attempt < 2:
                            import asyncio
                            wait = 2.0 * (attempt + 1)
                            print(f"[Azure] ⚠️  DNS error (attempt {attempt+1}/3), retrying in {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                    raise

            if response is None:
                print("[Azure] ❌ All retry attempts failed")
                return []

            if response.status_code != 200:
                print(f"[Azure] ❌ API error {response.status_code}: {response.text[:300]}")
                return []

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                print(f"[Azure] ❌ No choices in response: {str(data)[:300]}")
                return []
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                print(f"[Azure] ❌ Empty content in response")
                return []

            ms = (time.time() - t0) * 1000
            print(f"[Azure] ⏱ Extraction done: {ms:.0f}ms")
            print(f"[Azure] 📝 Raw response: {content[:200]}")

            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            items = json.loads(content)
            if not isinstance(items, list):
                print(f"[Azure] ⚠️  Expected list, got {type(items)}")
                return []

            valid_items = []
            for item in items:
                if isinstance(item, dict) and "summary" in item and "type" in item:
                    item.setdefault("priority", "Medium")
                    item.setdefault("description", item["summary"])
                    item.setdefault("labels", [])
                    if "client-feedback" not in item["labels"]:
                        item["labels"].append("client-feedback")
                    session_label = f"session-{date_str}"
                    if session_label not in item["labels"]:
                        item["labels"].append(session_label)
                    valid_items.append(item)

            print(f"[Azure] ✅ Extracted {len(valid_items)} action item(s)")
            for i, item in enumerate(valid_items):
                print(f"[Azure]   {i+1}. [{item['type']}] {item['summary']} ({item['priority']}) labels={item['labels']}")

            return valid_items

        except json.JSONDecodeError as e:
            print(f"[Azure] ❌ JSON parse failed: {e}")
            print(f"[Azure]   Raw content: {content[:200]}")
            return []
        except Exception as e:
            print(f"[Azure] ❌ Extraction failed: {e}")
            return []

    async def extract_session_summary(self, transcript: str,
                                        attendees: list | None = None,
                                        date_str: str = "") -> dict:
        """Generate a structured summary of a meeting for long-term memory.

        Returns a dict with topics/decisions/commitments/open_items/
        tickets_referenced/user_concerns/summary_text. Empty dict on failure.

        Unlike extract_action_items (which creates Jira tickets), this produces
        memory that will be loaded in future meetings with the same attendee group.
        """
        if not self.enabled:
            print("[Azure] ⚠️  Session summary disabled — no Azure credentials")
            return {}

        if not transcript or len(transcript.strip()) < 50:
            print("[Azure] ⚠️  Transcript too short — skipping summary")
            return {}

        import time
        if not date_str:
            date_str = time.strftime("%Y-%m-%d", time.gmtime())

        attendees_str = ", ".join(attendees) if attendees else "unknown"

        prompt = SESSION_SUMMARY_PROMPT.format(
            transcript=transcript[:12000],
            meeting_date=date_str,
            attendees=attendees_str,
        )

        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

        try:
            print(f"[Azure] 🧠 Generating session summary ({len(transcript)} chars, {attendees_str})...")
            t0 = time.time()

            response = None
            for attempt in range(3):
                try:
                    response = await self._client.post(
                        url,
                        headers={"api-key": self.api_key, "Content-Type": "application/json"},
                        json={
                            "messages": [
                                {"role": "system", "content": "You generate structured meeting summaries for AI memory. Respond with JSON only. Never invent facts."},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.1,  # low temperature for factual accuracy
                            "max_tokens": 1500,
                        },
                    )
                    break
                except Exception as net_err:
                    if "getaddrinfo" in str(net_err) or "ConnectError" in str(net_err):
                        if attempt < 2:
                            import asyncio
                            await asyncio.sleep(1.0 * (attempt + 1))
                            continue
                    raise

            if response is None:
                print("[Azure] ❌ Summary request failed")
                return {}

            elapsed_ms = (time.time() - t0) * 1000
            print(f"[Azure] ⏱ Summary generated: {elapsed_ms:.0f}ms")

            if response.status_code != 200:
                print(f"[Azure] ❌ HTTP {response.status_code}: {response.text[:200]}")
                return {}

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(l for l in lines if not l.strip().startswith("```"))

            summary = json.loads(content)

            # Validate structure
            if not isinstance(summary, dict):
                print(f"[Azure] ❌ Summary not a dict: {type(summary)}")
                return {}

            # Ensure all expected fields exist (empty defaults)
            for field in ["topics", "decisions", "commitments", "open_items",
                          "tickets_referenced", "user_concerns"]:
                if field not in summary or not isinstance(summary[field], list):
                    summary[field] = []

            if "summary_text" not in summary or not isinstance(summary["summary_text"], str):
                summary["summary_text"] = ""

            print(f"[Azure] ✅ Summary: {len(summary['topics'])} topics, "
                  f"{len(summary['decisions'])} decisions, "
                  f"{len(summary['commitments'])} commitments, "
                  f"{len(summary['open_items'])} open items")

            return summary

        except json.JSONDecodeError as e:
            print(f"[Azure] ❌ Summary JSON parse failed: {e}")
            return {}
        except Exception as e:
            print(f"[Azure] ❌ Summary generation failed: {e}")
            return {}

    async def close(self):
        if self.enabled:
            await self._client.aclose()