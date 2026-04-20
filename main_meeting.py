"""
main_meeting.py
Autonomous PM Agent — joins Google Meet via Recall.ai WebSocket.
Usage:
    python main_meeting.py <meeting_url> <wss_url>

Example:
    python main_meeting.py https://meet.google.com/abc-defg-hij wss://c2dtnz2w-8000.inc1.devtunnels.ms/ws
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load .env BEFORE any other imports — modules read env vars at import time
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from recall_bot import RecallBot
from websocket_server import WebSocketServer


async def main(meeting_url: str, ws_base_url: str):
    # Ensure wss:// pointing to /ws
    ws_url = ws_base_url.replace("https://", "wss://").replace("http://", "ws://")
    if not ws_url.endswith("/ws"):
        ws_url = ws_url.rstrip("/") + "/ws"

    print(f"[Config] WebSocket URL: {ws_url}")

    bot    = RecallBot()
    server = WebSocketServer(port=8000, bot_id=None)

    await server.start()

    bot_id = await bot.join(meeting_url, ws_url)
    server.speaker.bot_id = bot_id

    # Debug: check bot status after a few seconds to verify transcription is working
    await asyncio.sleep(5)
    try:
        status = await bot.get_status()
        print(f"[Debug] Bot status: {status.get('status_changes', [{}])[-1].get('code', 'unknown')}")
        # Check for transcription errors
        recording = status.get('recording', {})
        transcript_status = recording.get('transcript', {})
        if transcript_status:
            print(f"[Debug] Transcript config: {transcript_status}")
    except Exception as e:
        print(f"[Debug] Status check failed: {e}")

    print("\nPM Agent (Sam) is live in the meeting!")
    print("Everyone in the meeting will hear Sam's responses.")
    print("Press Ctrl+C to remove the bot and exit.\n")

    # Windows-compatible exit — just wait for KeyboardInterrupt
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\n[Shutting down] Removing bot from meeting...")
        try:
            await bot.leave()
        except Exception as e:
            print(f"[Shutdown] Bot leave failed (network issue): {e}")
        try:
            await server.speaker.close()
        except Exception:
            pass
        print("Done. Goodbye!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main_meeting.py <meeting_url> <wss_url>")
        print()
        print("Example:")
        print("  python main_meeting.py https://meet.google.com/abc-defg-hij wss://c2dtnz2w-8000.inc1.devtunnels.ms/ws")
        sys.exit(1)

    try:
        asyncio.run(main(sys.argv[1], sys.argv[2]))
    except KeyboardInterrupt:
        pass