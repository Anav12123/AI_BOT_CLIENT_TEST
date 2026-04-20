"""
server.py — Entry point with JWT auth + Output Media audio + multi-session
"""

import asyncio
import os
import time
import json
import hashlib
import hmac
import base64
import uuid
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

import session_store
_saved = session_store.load_settings()
_env_map = {"jira_url": "JIRA_BASE_URL", "jira_email": "JIRA_EMAIL", "jira_token": "JIRA_API_TOKEN", "jira_project": "JIRA_DEFAULT_PROJECT", "azure_endpoint": "AZURE_ENDPOINT", "azure_key": "AZURE_API_KEY", "azure_deployment": "AZURE_DEPLOYMENT", "simli_api_key": "SIMLI_API_KEY", "simli_face_id": "SIMLI_FACE_ID"}
for _k, _env in _env_map.items():
    _v = _saved.get(_k, "")
    if _v and not os.environ.get(_env):
        os.environ[_env] = _v

from websocket_server import WebSocketServer
from recall_bot import RecallBot

PORT = int(os.environ.get("PORT", 8000))
USE_OUTPUT_MEDIA = os.environ.get("USE_OUTPUT_MEDIA", "true").lower() in ("1", "true", "yes")
SIMLI_API_KEY = os.environ.get("SIMLI_API_KEY", "").strip()
SIMLI_FACE_ID = os.environ.get("SIMLI_FACE_ID", "").strip()

JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production-please")
JWT_EXPIRY = 24 * 3600

USERS = {}
admin_user = os.environ.get("ADMIN_USERNAME", "admin")
admin_pass = os.environ.get("ADMIN_PASSWORD", "admin123")
USERS[admin_user] = admin_pass
for i in range(1, 11):
    name = os.environ.get(f"USER_{i}_NAME", "").strip()
    pwd = os.environ.get(f"USER_{i}_PASS", "").strip()
    if name and pwd:
        USERS[name] = pwd
print(f"[Auth] {len(USERS)} user(s) configured")

def _b64url_encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

def _b64url_decode(s):
    s += "=" * (4 - len(s) % 4) if len(s) % 4 else ""
    return base64.urlsafe_b64decode(s)

def jwt_encode(payload):
    h = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = _b64url_encode(json.dumps(payload).encode())
    sig = hmac.new(JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"

def jwt_decode(token):
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h, p, s = parts
        expected = hmac.new(JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(s)):
            return None
        payload = json.loads(_b64url_decode(p))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None

def _get_user(request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return jwt_decode(auth[7:])

active_bots = {}
active_server = None
_start_time = time.time()


async def handle_login(request):
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if username not in USERS or USERS[username] != password:
        return web.json_response({"error": "Invalid credentials"}, status=401)
    token = jwt_encode({"sub": username, "iat": int(time.time()), "exp": int(time.time()) + JWT_EXPIRY})
    return web.json_response({"token": token, "username": username})


async def handle_start(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    meeting_url = data.get("meeting_url", "").strip()
    if not meeting_url:
        return web.json_response({"error": "meeting_url required"}, status=400)

    mode = data.get("mode", "client_call")
    username = user["sub"]

    if username in active_bots:
        try:
            old = active_bots[username]
            await old["bot"].leave()
            await active_server.remove_session(old["session_id"])
        except Exception:
            pass
        active_bots.pop(username, None)

    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "") or os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
    tunnel = os.environ.get("TUNNEL_URL", "").strip().rstrip("/")
    session_id = str(uuid.uuid4())

    if domain:
        base_url = f"https://{domain}"
        ws_base = f"wss://{domain}"
    elif tunnel:
        base_url = tunnel
        ws_base = tunnel.replace("https://", "wss://").replace("http://", "ws://")
    else:
        return web.json_response({"error": "No public URL configured."}, status=400)

    ws_url = f"{ws_base}/ws/{session_id}"
    audio_page_url = None
    if USE_OUTPUT_MEDIA:
        audio_page_url = f"{base_url}/audio-page?session={session_id}"
        # Simli avatar: check saved settings first, then env vars
        saved = session_store.load_settings()
        simli_key = saved.get("simli_api_key", "") or SIMLI_API_KEY
        simli_face = saved.get("simli_face_id", "") or SIMLI_FACE_ID
        simli_on = saved.get("simli_enabled", False)
        print(f"[Server] 🎭 Simli check: enabled={simli_on} ({type(simli_on).__name__}), key={'✅' if simli_key else '❌'}, face={'✅' if simli_face else '❌'}")
        if simli_on and simli_key and simli_face:
            audio_page_url += f"&simli_key={simli_key}&face_id={simli_face}"

    print(f"[Server] {username} → deploying Sam to {meeting_url}")
    print(f"[Server] Session: {session_id[:12]}, WS: {ws_url}")
    if audio_page_url:
        print(f"[Server] Audio page: {audio_page_url}")

    try:
        session = active_server.create_session(session_id, bot_id="pending")
        session.username = username
        session.meeting_url = meeting_url
        session.mode = mode
        session.started_at = time.time()
        await session.setup()

        bot = RecallBot()
        bot_id = await bot.join(
            meeting_url, ws_url,
            audio_page_url=audio_page_url,
            use_output_media=USE_OUTPUT_MEDIA,
            mode=mode,
        )
        session.bot_id = bot_id
        session.speaker.bot_id = bot_id

        active_bots[username] = {
            "bot": bot, "bot_id": bot_id, "session_id": session_id,
            "meeting_url": meeting_url, "started_at": time.time(),
        }
        return web.json_response({"status": "joined", "bot_id": bot_id, "session_id": session_id,
                                   "streaming": USE_OUTPUT_MEDIA and audio_page_url is not None})
    except Exception as e:
        await active_server.remove_session(session_id)
        print(f"[Server] Join failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_stop(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    username = user["sub"]
    if username not in active_bots:
        return web.json_response({"status": "no active bot"})
    info = active_bots[username]
    try:
        await info["bot"].leave()
    except Exception:
        pass
    try:
        await active_server.remove_session(info["session_id"])
    except Exception:
        pass
    active_bots.pop(username, None)
    return web.json_response({"status": "left"})


async def handle_status(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    info = active_bots.get(user["sub"])
    if info:
        session = active_server.sessions.get(info["session_id"])
        streaming = session._streaming_mode if session else False
        return web.json_response({
            "active": True, "bot_id": info["bot_id"], "session_id": info["session_id"],
            "meeting_url": info["meeting_url"], "uptime_seconds": int(time.time() - info["started_at"]),
            "streaming": streaming,
        })
    return web.json_response({"active": False})


async def handle_health(request):
    return web.json_response({"status": "ok", "active_bots": len(active_bots), "uptime": int(time.time() - _start_time)})


async def handle_sessions(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    return web.json_response({"sessions": session_store.get_sessions(limit=50, user=user["sub"])})


async def handle_session_detail(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    detail = session_store.get_session_detail(request.match_info.get("session_id", ""))
    if not detail or detail.get("user") != user["sub"]:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response(detail)


async def handle_settings_get(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    s = session_store.load_settings()
    return web.json_response({
        "jira": {"configured": bool(s.get("jira_url") and s.get("jira_email")), "url": s.get("jira_url", ""), "email": s.get("jira_email", ""), "project": s.get("jira_project", ""), "sprint": s.get("jira_sprint", "")},
        "azure": {"configured": bool(s.get("azure_endpoint")), "endpoint": s.get("azure_endpoint", ""), "deployment": s.get("azure_deployment", "")},
        "simli": {"enabled": s.get("simli_enabled", False), "face_id": s.get("simli_face_id", "")},
        "output_media": USE_OUTPUT_MEDIA,
    })


async def handle_settings_save(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    current = session_store.load_settings()
    for key in ["jira_url", "jira_email", "jira_token", "jira_project", "jira_sprint",
                 "azure_endpoint", "azure_key", "azure_deployment",
                 "simli_api_key", "simli_face_id"]:
        if key in data and data[key]:
            current[key] = data[key].strip()
    # Handle boolean toggle for simli_enabled
    if "simli_enabled" in data:
        current["simli_enabled"] = bool(data["simli_enabled"])
    print(f"[Settings] 🎭 Saving simli_enabled={current.get('simli_enabled')} (from_request={'simli_enabled' in data}, raw={data.get('simli_enabled')})")
    session_store.save_settings(current)
    return web.json_response({"ok": True})


async def handle_jira_test(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        from JiraClient import JiraClient
        jira = JiraClient()
        if not jira.enabled:
            return web.json_response({"ok": False, "error": "Not configured"})
        ok = await jira.test_connection()
        await jira.close()
        return web.json_response({"ok": ok, "message": f"Connected to {jira.base_url}" if ok else "Failed"})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


async def handle_jira_projects(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        from JiraClient import JiraClient
        jira = JiraClient()
        projects = await jira.get_projects() if jira.enabled else []
        await jira.close()
        return web.json_response({"projects": projects})
    except Exception as e:
        return web.json_response({"projects": [], "error": str(e)})


async def handle_jira_sprints(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    try:
        from JiraClient import JiraClient
        jira = JiraClient()
        sprints = await jira.get_sprints(project_key=request.query.get("project", jira.project)) if jira.enabled else []
        await jira.close()
        return web.json_response({"sprints": sprints})
    except Exception as e:
        return web.json_response({"sprints": [], "error": str(e)})


async def handle_pending_get(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    pending = session_store.get_pending_tickets()
    return web.json_response({"pending": pending, "count": len(pending)})


async def handle_pending_sync(request):
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    pending = session_store.get_pending_tickets()
    if not pending:
        return web.json_response({"synced": 0})
    from JiraClient import JiraClient
    jira = JiraClient()
    synced = 0
    for item in pending:
        try:
            await jira.create_ticket(summary=item.get("summary", ""), issue_type=item.get("type", "Task"), priority=item.get("priority", "Medium"), description=item.get("description", ""), labels=item.get("labels", []))
            synced += 1
        except Exception:
            break
    await jira.close()
    if synced > 0:
        session_store.clear_pending_tickets()
    return web.json_response({"synced": synced})


async def handle_audio_page(request):
    """Serve the Output Media audio page."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_page.html")
    if os.path.exists(html_path):
        resp = web.FileResponse(html_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return web.Response(text="audio_page.html not found", status=404)


async def handle_standups_today(request):
    """Get all team standups for today (PM dashboard)."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    date = request.query.get("date", None)
    standups = session_store.get_team_standups(date=date)
    blocker_count = sum(s.get("blocker_count", 0) for s in standups)
    completed = sum(1 for s in standups if s.get("completed"))
    return web.json_response({
        "date": date or time.strftime("%Y-%m-%d", time.gmtime()),
        "standups": standups,
        "total": len(standups),
        "completed": completed,
        "blocker_count": blocker_count,
    })


async def handle_standup_detail(request):
    """Get full standup detail for a specific developer."""
    user = _get_user(request)
    if not user:
        return web.json_response({"error": "Unauthorized"}, status=401)
    developer = request.match_info.get("developer", "")
    date = request.query.get("date", None)
    detail = session_store.get_standup_detail(developer, date)
    if not detail:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response(detail)


async def handle_index(request):
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        resp = web.FileResponse(html_path)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp
    return web.Response(text="index.html not found", status=404)


async def main():
    global active_server
    server = WebSocketServer(port=PORT)
    active_server = server

    # When a session is removed (bot left, kicked, disconnected), clean up active_bots
    def on_session_removed(session):
        username = session.username
        if username and username in active_bots:
            active_bots.pop(username, None)
            print(f"[Server] 🔄 Bot status reset for {username} (session ended)")

    server.on_session_removed = on_session_removed

    routes = [
        ("POST", "/auth/login", handle_login),
        ("POST", "/start", handle_start),
        ("POST", "/stop", handle_stop),
        ("GET", "/status", handle_status),
        ("GET", "/api/health", handle_health),
        ("GET", "/api/sessions", handle_sessions),
        ("GET", "/api/sessions/{session_id}", handle_session_detail),
        ("GET", "/api/settings", handle_settings_get),
        ("POST", "/api/settings/save", handle_settings_save),
        ("POST", "/api/settings/jira/test", handle_jira_test),
        ("GET", "/api/jira/projects", handle_jira_projects),
        ("GET", "/api/jira/sprints", handle_jira_sprints),
        ("GET", "/api/pending", handle_pending_get),
        ("POST", "/api/pending/sync", handle_pending_sync),
        ("GET", "/api/standups", handle_standups_today),
        ("GET", "/api/standups/{developer}", handle_standup_detail),
        ("GET", "/audio-page", handle_audio_page),
        ("GET", "/", handle_index),
    ]
    for method, path, handler in routes:
        if method == "GET":
            server.app.router.add_get(path, handler)
        elif method == "POST":
            server.app.router.add_post(path, handler)

    await server.start()

    mode = "Output Media (streaming)" if USE_OUTPUT_MEDIA else "output_audio API (fallback)"
    print(f"[Server] Running on port {PORT}")
    print(f"[Server] Frontend: http://localhost:{PORT}/")
    print(f"[Server] Audio mode: {mode}")
    simli_status = f"✅ Enabled (face: {SIMLI_FACE_ID})" if SIMLI_API_KEY and SIMLI_FACE_ID else "⚠️  Disabled (no SIMLI_API_KEY or SIMLI_FACE_ID)"
    print(f"[Server] Simli avatar: {simli_status}")
    print(f"[Server] Credentials: {admin_user} / {'*' * len(admin_pass)}")

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

if __name__ == "__main__":
    asyncio.run(main())