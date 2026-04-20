"""
JiraClient.py — Jira Cloud REST API wrapper

Features:
  - Read: get ticket, search, my tickets, sprint status, projects, sprints
  - Write: create ticket, add comment, transition (with already-done check)
  - Smart: find related tickets, resolve ticket IDs from speech
  - Resilient: DNS retry (3 attempts), rate limit handling, error classification
"""

import os
import re
import httpx
from typing import Optional


class JiraAuthError(Exception):
    pass

class JiraPermissionError(Exception):
    pass

class JiraNotFoundError(Exception):
    pass

class JiraTransitionError(Exception):
    pass


class JiraClient:
    def __init__(self):
        self.base_url = os.environ.get("JIRA_BASE_URL", "").strip().rstrip("/")
        self.email    = os.environ.get("JIRA_EMAIL", "").strip()
        self.token    = os.environ.get("JIRA_API_TOKEN", "").strip()
        self.project  = os.environ.get("JIRA_DEFAULT_PROJECT", "PROJ").strip()

        if not all([self.base_url, self.email, self.token]):
            print("[Jira] ⚠️  Missing JIRA_BASE_URL, JIRA_EMAIL, or JIRA_API_TOKEN — Jira disabled")
            self.enabled = False
            return

        self.enabled = True
        self._client = httpx.AsyncClient(
            timeout=15,
            auth=(self.email, self.token),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        print(f"[Jira] ✅ Configured: {self.base_url} (project: {self.project})")

    # ── Core API call with DNS retry ──────────────────────────────────────────

    async def _api(self, method: str, path: str, json_data: dict = None, params: dict = None) -> dict:
        if not self.enabled:
            raise JiraAuthError("Jira is not configured")

        url = f"{self.base_url}/rest/api/3{path}"
        last_err = None

        for attempt in range(3):
            try:
                response = await self._client.request(method, url, json=json_data, params=params)

                if response.status_code == 401:
                    raise JiraAuthError("Invalid Jira credentials — check API token")
                if response.status_code == 403:
                    raise JiraPermissionError("No permission — check API token scope")
                if response.status_code == 404:
                    raise JiraNotFoundError("Not found in Jira")
                if response.status_code == 429:
                    import asyncio
                    print("[Jira] ⚠️  Rate limited — waiting 5s...")
                    await asyncio.sleep(5)
                    response = await self._client.request(method, url, json=json_data, params=params)
                    response.raise_for_status()

                if response.status_code >= 400:
                    print(f"[Jira] API error {response.status_code}: {response.text[:300]}")
                    response.raise_for_status()

                if response.status_code == 204:
                    return {}
                return response.json()

            except (JiraAuthError, JiraPermissionError, JiraNotFoundError, JiraTransitionError):
                raise
            except httpx.TimeoutException:
                raise TimeoutError("Jira is not responding — try again later")
            except Exception as e:
                if isinstance(e, (JiraAuthError, JiraPermissionError, JiraNotFoundError)):
                    raise
                last_err = e
                if "getaddrinfo" in str(e) or "ConnectError" in str(e) or "DNS" in str(e):
                    if attempt < 2:
                        import asyncio
                        wait = 1.0 * (attempt + 1)
                        print(f"[Jira] ⚠️  DNS error (attempt {attempt+1}/3), retrying in {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                print(f"[Jira] Request failed: {e}")
                raise

        raise last_err

    # ── Agile API (boards, sprints) ───────────────────────────────────────────

    async def _agile_api(self, method: str, path: str, params: dict = None, json_data: dict = None) -> dict:
        """Call Jira Agile REST API (different base path)."""
        if not self.enabled:
            raise JiraAuthError("Jira is not configured")
        url = f"{self.base_url}/rest/agile/1.0{path}"
        last_err = None
        for attempt in range(3):
            try:
                response = await self._client.request(method, url, params=params, json=json_data)
                if response.status_code >= 400:
                    if response.status_code == 404:
                        return {"values": []}
                    response.raise_for_status()
                return response.json() if response.text else {}
            except Exception as e:
                last_err = e
                if "getaddrinfo" in str(e) or "ConnectError" in str(e):
                    if attempt < 2:
                        import asyncio
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                raise
        raise last_err

    # ── Connection test ───────────────────────────────────────────────────────

    async def test_connection(self) -> bool:
        if not self.enabled:
            return False
        try:
            data = await self._api("GET", "/myself")
            name = data.get("displayName", "Unknown")
            print(f"[Jira] ✅ Connected as: {name}")
            return True
        except Exception as e:
            print(f"[Jira] ❌ Connection failed: {e}")
            return False

    # ── Project & Sprint discovery ────────────────────────────────────────────

    async def get_projects(self) -> list[dict]:
        """Get all accessible Jira projects."""
        try:
            data = await self._api("GET", "/project", params={"maxResults": 50})
            if isinstance(data, list):
                projects = [{"key": p.get("key", ""), "name": p.get("name", ""), "id": p.get("id", "")} for p in data]
                print(f"[Jira] 📁 Found {len(projects)} project(s)")
                return projects
            return []
        except Exception as e:
            print(f"[Jira] ⚠️  Failed to fetch projects: {e}")
            return []

    async def get_boards(self, project_key: str = None) -> list[dict]:
        """Get Scrum/Kanban boards for a project."""
        try:
            params = {"maxResults": 50}
            if project_key:
                params["projectKeyOrId"] = project_key
            data = await self._agile_api("GET", "/board", params=params)
            boards = [
                {"id": b.get("id", ""), "name": b.get("name", ""), "type": b.get("type", "")}
                for b in data.get("values", [])
            ]
            print(f"[Jira] 📋 Found {len(boards)} board(s)")
            return boards
        except Exception as e:
            print(f"[Jira] ⚠️  Failed to fetch boards: {e}")
            return []

    async def get_sprints(self, board_id: int = None, project_key: str = None) -> list[dict]:
        """Get sprints for a board. If no board_id, finds board from project."""
        try:
            if not board_id and project_key:
                boards = await self.get_boards(project_key)
                if boards:
                    board_id = boards[0]["id"]
            if not board_id:
                boards = await self.get_boards(self.project)
                if boards:
                    board_id = boards[0]["id"]
            if not board_id:
                return []

            data = await self._agile_api("GET", f"/board/{board_id}/sprint", params={"maxResults": 10, "state": "active,future"})
            sprints = [
                {
                    "id": s.get("id", ""),
                    "name": s.get("name", ""),
                    "state": s.get("state", ""),
                    "startDate": s.get("startDate", "")[:10] if s.get("startDate") else "",
                    "endDate": s.get("endDate", "")[:10] if s.get("endDate") else "",
                }
                for s in data.get("values", [])
            ]
            print(f"[Jira] 🏃 Found {len(sprints)} sprint(s) for board {board_id}")
            return sprints
        except Exception as e:
            print(f"[Jira] ⚠️  Failed to fetch sprints: {e}")
            return []

    async def get_active_sprint(self) -> dict | None:
        """Get the currently active sprint. Returns None if no active sprint."""
        try:
            sprints = await self.get_sprints()
            for s in sprints:
                if s.get("state") == "active":
                    return s
            return None
        except Exception:
            return None

    async def move_to_sprint(self, ticket_ids: list[str], sprint_id: int = None) -> bool:
        """Move tickets to a sprint. If no sprint_id, uses active sprint."""
        try:
            if not sprint_id:
                active = await self.get_active_sprint()
                if not active:
                    print("[Jira] ⚠️  No active sprint — skipping sprint assignment")
                    return False
                sprint_id = active["id"]
                sprint_name = active.get("name", sprint_id)
            else:
                sprint_name = sprint_id

            await self._agile_api("POST", f"/sprint/{sprint_id}/issue",
                                   json_data={"issues": ticket_ids})
            for tid in ticket_ids:
                print(f"[Jira] 📋 Assigned {tid} to sprint \"{sprint_name}\"")
            return True
        except Exception as e:
            print(f"[Jira] ⚠️  Sprint assignment failed: {e}")
            return False

    # ── Read operations ───────────────────────────────────────────────────────

    async def get_ticket(self, ticket_id: str) -> dict:
        data = await self._api("GET", f"/issue/{ticket_id}", params={
            "fields": "summary,status,assignee,priority,created,updated,issuetype,description"
        })
        return self._format_ticket(data)

    async def get_my_tickets(self, max_results: int = 10) -> list[dict]:
        jql = f"project = {self.project} AND status != Done ORDER BY assignee ASC, priority DESC, updated DESC"
        return await self.search_jql(jql, max_results)

    async def get_assigned_tickets(self, max_results: int = 10) -> list[dict]:
        jql = f"project = {self.project} AND assignee = currentUser() AND status != Done ORDER BY priority DESC, updated DESC"
        return await self.search_jql(jql, max_results)

    async def get_sprint_tickets(self, max_results: int = 30) -> list[dict]:
        jql = f"project = {self.project} AND sprint in openSprints() ORDER BY status ASC, priority DESC"
        return await self.search_jql(jql, max_results)

    async def get_sprint_status(self) -> dict:
        tickets = await self.get_sprint_tickets(50)
        status_counts = {"Done": 0, "In Progress": 0, "To Do": 0, "Other": 0}
        for t in tickets:
            status = t.get("status", "Other")
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts["Other"] += 1
        return {
            "total": len(tickets),
            "done": status_counts["Done"],
            "in_progress": status_counts["In Progress"],
            "to_do": status_counts["To Do"],
            "other": status_counts["Other"],
        }

    async def search_jql(self, jql: str, max_results: int = 10) -> list[dict]:
        data = await self._api("GET", "/search/jql", params={
            "jql": jql,
            "maxResults": max_results,
            "fields": "summary,status,assignee,priority,issuetype,updated,description",
        })
        issues = data.get("issues", [])
        total = data.get("total", len(issues))
        print(f"[Jira] 🔍 JQL returned {len(issues)} issue(s) (total: {total})")
        return [self._format_ticket(i) for i in issues]

    async def count_tickets(self, jql: str) -> int:
        """Get total count matching a JQL query without fetching full results.

        Used for project-size detection to decide pre-load strategy.
        Returns 0 on error so caller can fall back to simple pre-load.
        """
        try:
            data = await self._api("GET", "/search/jql", params={
                "jql": jql,
                "maxResults": 0,  # fetch metadata only, no issues
                "fields": "summary",  # minimal field
            })
            return data.get("total", 0)
        except Exception as e:
            print(f"[Jira] ⚠️  count_tickets failed: {e}")
            return 0

    async def search_text(self, query: str, max_results: int = 5) -> list[dict]:
        jql = f'project = {self.project} AND text ~ "{query}" ORDER BY updated DESC'
        return await self.search_jql(jql, max_results)

    # ── Smart: find related tickets ───────────────────────────────────────────

    async def find_related_tickets(self, summary: str, max_results: int = 3) -> list[dict]:
        """Search for existing tickets similar to a new one being created."""
        words = [w for w in summary.lower().split() if len(w) > 3 and w not in {
            "client", "request", "feature", "issue", "about", "from", "with",
            "that", "this", "there", "their", "they", "have", "been", "should",
        }]
        if not words:
            return []
        query = " ".join(words[:4])
        try:
            results = await self.search_text(query, max_results)
            if results:
                print(f"[Jira] 🔗 Found {len(results)} related ticket(s) for \"{summary[:40]}\"")
            return results
        except Exception as e:
            print(f"[Jira] ⚠️  Related ticket search failed: {e}")
            return []

    # ── Write operations ──────────────────────────────────────────────────────

    async def create_ticket(self, summary: str, issue_type: str = "Task",
                           priority: str = "Medium", description: str = "",
                           labels: list[str] = None, assignee_id: str = None) -> dict:
        fields = {
            "project": {"key": self.project},
            "summary": summary,
            "issuetype": {"name": issue_type},
        }
        if description:
            fields["description"] = {
                "type": "doc", "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]
            }
        if priority:
            fields["priority"] = {"name": priority}
        if labels:
            fields["labels"] = labels
        if assignee_id:
            fields["assignee"] = {"accountId": assignee_id}

        data = await self._api("POST", "/issue", json_data={"fields": fields})
        ticket_key = data.get("key", "unknown")
        print(f"[Jira] ✅ Created: {ticket_key} — {summary}")
        return {"key": ticket_key, "id": data.get("id"), "summary": summary}

    async def _get_subtask_type_name(self) -> str:
        """Discover the project's subtask issue type name (varies by configuration)."""
        if hasattr(self, '_subtask_type_name') and self._subtask_type_name:
            return self._subtask_type_name
        try:
            data = await self._api("GET", "/issuetype")
            if isinstance(data, list):
                for it in data:
                    if it.get("subtask") is True:
                        name = it.get("name", "")
                        print(f"[Jira] 🔍 Discovered subtask type: \"{name}\"")
                        self._subtask_type_name = name
                        return name
            print(f"[Jira] ⚠️  No subtask type found in project, available types: {[it.get('name') for it in data if isinstance(data, list)]}")
        except Exception as e:
            print(f"[Jira] ⚠️  Failed to discover subtask type: {e}")
        self._subtask_type_name = "Sub-task"  # fallback
        return "Sub-task"

    async def create_subtask(self, parent_key: str, summary: str,
                              description: str = "", priority: str = "Medium",
                              labels: list[str] = None) -> dict:
        """Create a subtask under a parent ticket.
        Auto-discovers the project's subtask issue type name.
        If parent is already a subtask, creates under grandparent instead."""
        # Check if parent is itself a subtask (can't nest subtasks)
        try:
            parent_data = await self._api("GET", f"/issue/{parent_key}", params={
                "fields": "issuetype,parent"
            })
            parent_fields = parent_data.get("fields", {})
            parent_type = parent_fields.get("issuetype", {})
            if parent_type.get("subtask") is True:
                grandparent = parent_fields.get("parent", {})
                gp_key = grandparent.get("key", "")
                if gp_key:
                    print(f"[Jira] 🔄 {parent_key} is a subtask — creating under grandparent {gp_key} instead")
                    parent_key = gp_key
                else:
                    print(f"[Jira] ⚠️  {parent_key} is a subtask with no parent — creating standalone ticket")
                    return await self.create_ticket(summary=summary, issue_type="Task",
                                                     priority=priority, description=description, labels=labels)
        except Exception as e:
            print(f"[Jira] ⚠️  Parent check failed for {parent_key}: {e}")

        # Discover the correct subtask type name for this project
        subtask_type = await self._get_subtask_type_name()

        # Build ADF description with proper formatting
        desc_paragraphs = []
        for line in description.split("\n"):
            line = line.strip()
            if not line:
                continue
            desc_paragraphs.append({
                "type": "paragraph",
                "content": [{"type": "text", "text": line}]
            })
        if not desc_paragraphs:
            desc_paragraphs = [{"type": "paragraph", "content": [{"type": "text", "text": description or "Standup update"}]}]

        fields = {
            "project": {"key": self.project},
            "parent": {"key": parent_key},
            "summary": summary,
            "issuetype": {"name": subtask_type},
            "description": {
                "type": "doc", "version": 1,
                "content": desc_paragraphs,
            },
        }
        if priority:
            fields["priority"] = {"name": priority}
        if labels:
            fields["labels"] = labels

        try:
            data = await self._api("POST", "/issue", json_data={"fields": fields})
            ticket_key = data.get("key", "unknown")
            print(f"[Jira] ✅ Subtask created: {ticket_key} under {parent_key} — {summary}")
            return {"key": ticket_key, "id": data.get("id"), "parent": parent_key, "summary": summary}
        except Exception as e:
            print(f"[Jira] ❌ Subtask creation failed (type: \"{subtask_type}\"): {e}")
            raise

    async def add_comment(self, ticket_id: str, comment: str) -> dict:
        body = {
            "body": {
                "type": "doc", "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment}]}]
            }
        }
        data = await self._api("POST", f"/issue/{ticket_id}/comment", json_data=body)
        print(f"[Jira] ✅ Comment added to {ticket_id}")
        return data

    async def get_transitions(self, ticket_id: str) -> list[dict]:
        data = await self._api("GET", f"/issue/{ticket_id}/transitions")
        return data.get("transitions", [])

    async def transition_ticket(self, ticket_id: str, target_status: str) -> dict:
        ticket = await self.get_ticket(ticket_id)
        current_status = ticket.get("status", "Unknown")

        if current_status.lower() == target_status.lower():
            return {"ticket": ticket_id, "already_at": current_status, "action": "already_done"}

        transitions = await self.get_transitions(ticket_id)
        target_lower = target_status.lower()
        match = None
        for t in transitions:
            if t["name"].lower() == target_lower or t["to"]["name"].lower() == target_lower:
                match = t
                break

        if not match:
            available_names = [f"{t['to']['name']}" for t in transitions]
            raise JiraTransitionError(
                f"Can't move {ticket_id} from '{current_status}' to '{target_status}'. "
                f"Available moves: {', '.join(available_names) if available_names else 'none'}"
            )

        await self._api("POST", f"/issue/{ticket_id}/transitions", json_data={"transition": {"id": match["id"]}})
        new_status = match["to"]["name"]
        print(f"[Jira] ✅ {ticket_id} → {new_status}")
        return {"ticket": ticket_id, "new_status": new_status, "action": "transitioned"}

    # ── Ticket ID resolver ────────────────────────────────────────────────────

    def resolve_ticket_id(self, spoken_text: str) -> Optional[str]:
        text = spoken_text.strip()
        match = re.search(r'\b([A-Z]+-\d+)\b', text)
        if match:
            return match.group(1)
        match = re.search(r'\b(\w+)\s+(\d+)\b', text, re.IGNORECASE)
        if match:
            prefix = match.group(1).upper()
            number = match.group(2)
            if prefix in ("PROJ", "PROJECT", "TICKET", "ISSUE", "TASK", "BUG"):
                return f"{self.project}-{number}"
            if len(prefix) <= 6 and prefix.isalpha():
                return f"{prefix}-{number}"
        match = re.search(r'\b(\d+)\b', text)
        if match:
            return f"{self.project}-{match.group(1)}"
        digit_words = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
        words = text.lower().split()
        consecutive_digits = []
        current_run = []
        for w in words:
            if w in digit_words:
                current_run.append(digit_words[w])
            else:
                if len(current_run) >= 2:
                    consecutive_digits = current_run
                current_run = []
        if len(current_run) >= 2:
            consecutive_digits = current_run
        if consecutive_digits:
            return f"{self.project}-{''.join(consecutive_digits)}"
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_ticket(self, issue: dict) -> dict:
        if not issue:
            return {"key": "?", "summary": "Unknown", "status": "Unknown", "priority": "Unknown",
                    "type": "Unknown", "assignee": "Unassigned", "updated": "", "description": ""}
        fields = issue.get("fields") or {}
        assignee = fields.get("assignee")
        status = fields.get("status") or {}
        priority = fields.get("priority") or {}
        issuetype = fields.get("issuetype") or {}
        # Extract description text (ADF or plain text)
        desc_raw = fields.get("description") or ""
        if isinstance(desc_raw, dict):
            # Atlassian Document Format — extract text content
            desc_parts = []
            for block in desc_raw.get("content", []):
                for item in block.get("content", []):
                    if item.get("type") == "text":
                        desc_parts.append(item.get("text", ""))
            desc_raw = " ".join(desc_parts)
        desc_text = str(desc_raw)[:200]  # Truncate to 200 chars
        return {
            "key": issue.get("key", ""),
            "summary": fields.get("summary", ""),
            "status": status.get("name", "Unknown"),
            "priority": priority.get("name", "Medium"),
            "type": issuetype.get("name", "Task"),
            "assignee": assignee.get("displayName", "Unassigned") if assignee else "Unassigned",
            "updated": (fields.get("updated") or "")[:10],
            "description": desc_text,
        }

    async def search_user(self, display_name: str) -> Optional[str]:
        try:
            data = await self._api("GET", "/user/search", params={"query": display_name, "maxResults": 1})
            if data and isinstance(data, list) and len(data) > 0:
                account_id = data[0].get("accountId")
                found_name = data[0].get("displayName", "?")
                print(f"[Jira] 👤 User lookup: \"{display_name}\" → {found_name} ({account_id})")
                return account_id
            print(f"[Jira] 👤 User not found: \"{display_name}\"")
            return None
        except Exception as e:
            print(f"[Jira] ⚠️  User search failed for \"{display_name}\": {e}")
            return None

    async def close(self):
        if self.enabled:
            await self._client.aclose()