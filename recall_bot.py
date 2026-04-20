"""
recall_bot.py — Recall.ai bot with Output Media + Deepgram Nova-3

Audio delivery: Output Media (webpage streams PCM via AudioWorklet)
Fallback: automatic_audio_output (MP3 injection) if output_media disabled
"""

import os
import httpx

RECALL_REGION   = os.environ.get("RECALLAI_REGION", "ap-northeast-1")
RECALL_API_BASE = f"https://{RECALL_REGION}.recall.ai/api/v1"

SILENT_MP3_B64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjI5LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADkADMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzM//////////////////////////////////////////////////////////////////8AAAAATGF2YzU4LjU0AAAAAAAAAAAAAAAAJAAAAAAAAAAAkFCGaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


class RecallBot:
    def __init__(self):
        self.api_key = os.environ["RECALLAI_API_KEY"]
        self.bot_id: str | None = None
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type":  "application/json",
            "accept":        "application/json",
        }

    async def join(self, meeting_url: str, websocket_url: str,
                   audio_page_url: str = None, use_output_media: bool = True,
                   mode: str = "client") -> str:
        """Join a meeting. If audio_page_url provided and use_output_media=True,
        uses Output Media for streaming audio. Otherwise falls back to output_audio API.

        mode: 'standup' or 'client'. In client mode, enables Perfect Diarization
        (per-participant separate streams) for better speaker attribution during
        overlapping speech. Costs ~1.8x transcription credits for client calls.
        """

        platform = "Google Meet" if "meet.google" in meeting_url else \
                   "Teams" if "teams.microsoft" in meeting_url else \
                   "Zoom" if "zoom.us" in meeting_url else "Unknown"
        print(f"[Recall.ai] Joining {platform} meeting...")

        payload = {
            "meeting_url": meeting_url,
            "bot_name":    "Sam",
            "recording_config": {
                "transcript": {
                    "provider": {
                        "deepgram_streaming": {
                            "language":    "en",
                            "model":       "nova-3",
                            "endpointing": 150,
                            "keyterm": ["AnavClouds", "AnavClouds Software Solutions", "Salesforce", "Sam"]
                        }
                    }
                },
                "audio_mixed_raw": {},
                "audio_separate_raw": {},
                "realtime_endpoints": [{
                    "type":   "websocket",
                    "url":    websocket_url,
                    "events": [
                        "transcript.data",
                        "transcript.partial_data",
                        "participant_events.speech_on",
                        "participant_events.speech_off",
                        "participant_events.join",
                        "participant_events.leave",
                        "audio_mixed_raw.data",
                        "audio_separate_raw.data",
                    ]
                }]
            },
            "automatic_leave": {
                "everyone_left_timeout": {"timeout": 10, "activate_after": 1},
                "waiting_room_timeout": 300,
                "noone_joined_timeout": 300,
                "silence_detection": {"timeout": 600, "activate_after": 600},
                "bot_detection": {
                    "using_participant_names": {
                        "matches": [
                            "notetaker", "recorder", "otter", "fireflies",
                            "grain", "fathom", "copilot", "read.ai", "tl;dv",
                            "fellow", "assistant", "meeting bot", "ai bot"
                        ],
                        "timeout": 10,
                        "activate_after": 300
                    },
                    "using_participant_events": {
                        "timeout": 600,
                        "activate_after": 600
                    }
                }
            }
        }

        # ── Perfect Diarization (client mode only) ────────────────────
        # Transcribes each participant's audio on a separate stream, giving
        # accurate speaker attribution even during overlapping speech. Not
        # useful for standup mode (single speaker). Costs ~1.8x Deepgram credits.
        if mode == "client":
            payload["recording_config"]["transcript"]["diarization"] = {
                "use_separate_streams_when_available": True
            }
            print(f"[Recall.ai] 🎙️ Perfect Diarization enabled (client mode)")

        # ── Audio delivery mode ──────────────────────────────────────
        if use_output_media and audio_page_url:
            print(f"[Recall.ai] 🔊 Output Media mode: {audio_page_url}")
            payload["output_media"] = {
                "camera": {
                    "kind": "webpage",
                    "config": {"url": audio_page_url}
                }
            }
            # Use 4-core bot for better audio processing
            payload["variant"] = {
                "google_meet": "web_4_core",
                "zoom": "web_4_core",
                "microsoft_teams": "web_4_core",
            }
            # Still need automatic_audio_output for the bot to be allowed to output audio
            # The silent MP3 enables the audio output capability
            payload["automatic_audio_output"] = {
                "in_call_recording": {
                    "data": {"kind": "mp3", "b64_data": SILENT_MP3_B64}
                }
            }
        else:
            print(f"[Recall.ai] 🔊 Fallback mode: output_audio API")
            payload["automatic_audio_output"] = {
                "in_call_recording": {
                    "data": {"kind": "mp3", "b64_data": SILENT_MP3_B64}
                }
            }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{RECALL_API_BASE}/bot/",
                headers=self.headers,
                json=payload,
            )
            if resp.status_code != 201:
                print(f"[Recall.ai] Error: {resp.text}")
            resp.raise_for_status()
            data = resp.json()

        self.bot_id = data["id"]
        print(f"[Recall.ai] Bot joined! ID: {self.bot_id}")
        return self.bot_id

    async def leave(self):
        if not self.bot_id:
            return
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(
                        f"{RECALL_API_BASE}/bot/{self.bot_id}/leave_call/",
                        headers=self.headers,
                    )
                print("[Recall.ai] Bot left the meeting.")
                self.bot_id = None
                return
            except Exception as e:
                if attempt < 2:
                    print(f"[Recall.ai] Leave failed (attempt {attempt+1}/3): {e}")
                    await asyncio.sleep(1)
                else:
                    print(f"[Recall.ai] Leave failed after 3 attempts: {e}")
                    self.bot_id = None

    async def get_status(self) -> dict:
        if not self.bot_id:
            return {}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{RECALL_API_BASE}/bot/{self.bot_id}/",
                headers=self.headers,
            )
            r.raise_for_status()
            return r.json()