"""
Streaming STT clients.

- Deepgram Flux — Deepgram SDK, v2/listen, model-integrated end-of-turn
- Deepgram Nova — Raw WebSocket, v1/listen, endpointing-based
"""

import asyncio
import json
import logging
from typing import Callable, Awaitable, Optional

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

_DG_V1_WS_URL = "wss://api.deepgram.com/v1/listen"
_KEEPALIVE_INTERVAL = 8


def _is_flux_model(model: str) -> bool:
    return model.startswith("flux")


# ═══════════════════════════════════════════════════════════════════════════
# Flux via Deepgram SDK (v2 API) — connection-managed + full param control
# ═══════════════════════════════════════════════════════════════════════════

async def _stream_flux_sdk(
    audio_queue: asyncio.Queue,
    transcript_callback: Callable,
    api_key: str,
    sample_rate: int = 16000,
    keywords: Optional[list] = None,
    start_of_turn_callback: Optional[Callable] = None,
    end_of_turn_callback: Optional[Callable] = None,
    eager_eot_callback: Optional[Callable] = None,
    turn_resumed_callback: Optional[Callable] = None,
    eot_threshold: float = 0.65,
    eager_eot_threshold: Optional[float] = 0.35,
    eot_timeout_ms: int = 1500,
) -> None:
    """Stream audio to Deepgram Flux using the official SDK.

    SDK handles connection lifecycle (keepalive, SSL, reconnection).
    Flux-specific params passed as kwargs → SDK maps to URL query params.
    """
    from deepgram import AsyncDeepgramClient
    from deepgram.core.events import EventType

    client = AsyncDeepgramClient(api_key=api_key)

    # Build SDK connection kwargs — all become URL query params
    connect_kwargs = {
        "model": "flux-general-en",
        "encoding": "linear16",
        "sample_rate": str(sample_rate),
        "eot_threshold": str(eot_threshold),
        "eot_timeout_ms": str(eot_timeout_ms),
    }
    if eager_eot_threshold is not None:
        connect_kwargs["eager_eot_threshold"] = str(eager_eot_threshold)
    if keywords:
        connect_kwargs["keyterm"] = keywords

    logger.info(
        "Flux SDK: connecting | eot=%.2f eager=%s timeout=%dms | %d keyterm(s)",
        eot_threshold,
        f"{eager_eot_threshold:.2f}" if eager_eot_threshold else "off",
        eot_timeout_ms,
        len(keywords) if keywords else 0,
    )

    async with client.listen.v2.connect(**connect_kwargs) as connection:

        _done = asyncio.Event()

        async def on_message(message):
            msg_type = getattr(message, "type", "")

            if msg_type == "Connected":
                logger.info("Flux SDK: Connected")
                return

            if msg_type == "TurnInfo":
                event = getattr(message, "event", "")
                transcript = (getattr(message, "transcript", "") or "").strip()
                turn_index = getattr(message, "turn_index", 0)
                eot_confidence = getattr(message, "end_of_turn_confidence", 0.0)

                if event == "StartOfTurn":
                    logger.info("Flux SDK: StartOfTurn | turn=%d", turn_index)
                    if start_of_turn_callback:
                        try:
                            await start_of_turn_callback()
                        except Exception as e:
                            logger.error("start_of_turn_callback: %s", e)

                elif event == "Update":
                    if transcript:
                        await transcript_callback(transcript, False, None)

                elif event == "EagerEndOfTurn":
                    if transcript:
                        logger.info("Flux SDK: EagerEOT conf=%.2f text=%r", eot_confidence, transcript[:60])
                        # Send as interim (not final) — EndOfTurn will confirm
                        await transcript_callback(transcript, False, None)
                        # Fire speculative callback for pre-computation
                        if eager_eot_callback:
                            try:
                                await eager_eot_callback(transcript, eot_confidence)
                            except Exception as e:
                                logger.error("eager_eot_callback: %s", e)

                elif event == "TurnResumed":
                    logger.info("Flux SDK: TurnResumed | turn=%d", turn_index)
                    if transcript:
                        await transcript_callback(transcript, False, None)
                    # Cancel any speculative processing
                    if turn_resumed_callback:
                        try:
                            await turn_resumed_callback()
                        except Exception as e:
                            logger.error("turn_resumed_callback: %s", e)

                elif event == "EndOfTurn":
                    logger.info("Flux SDK: EndOfTurn conf=%.2f text=%r", eot_confidence, transcript[:60])
                    if transcript:
                        await transcript_callback(transcript, True, None)
                    if end_of_turn_callback:
                        try:
                            await end_of_turn_callback(eot_confidence)
                        except Exception as e:
                            logger.error("end_of_turn_callback: %s", e)
                return

            # Fallback for unexpected message formats
            if hasattr(message, 'transcript') and message.transcript:
                t = message.transcript.strip()
                if t:
                    is_final = getattr(message, 'is_final', False) or getattr(message, 'speech_final', False)
                    await transcript_callback(t, is_final, None)

        def on_close(_):
            logger.info("Flux SDK: closed")
            _done.set()

        def on_error(error):
            logger.error("Flux SDK: error: %s", error)

        connection.on(EventType.OPEN, lambda _: logger.info("Flux SDK: opened"))
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.CLOSE, on_close)
        connection.on(EventType.ERROR, on_error)

        listen_task = asyncio.create_task(connection.start_listening())

        # Feed audio from queue
        try:
            while not _done.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if chunk is None:
                    break
                try:
                    await connection._send(chunk)
                except Exception as e:
                    logger.warning("Flux SDK: send error: %s", e)
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Flux SDK: feed error: %s", e)

        try:
            listen_task.cancel()
            await asyncio.sleep(0.1)
        except Exception:
            pass

    logger.info("Flux SDK: session ended")


# ═══════════════════════════════════════════════════════════════════════════
# Nova via raw WebSocket (v1 API)
# ═══════════════════════════════════════════════════════════════════════════

async def _stream_nova_ws(
    audio_queue: asyncio.Queue,
    transcript_callback: Callable,
    api_key: str,
    model: str = "nova-3",
    language: str = "en",
    endpointing: int = 300,
    sample_rate: int = 16000,
    speech_started_callback: Optional[Callable] = None,
    keywords: Optional[list] = None,
) -> None:
    """Stream audio to Deepgram Nova via raw WebSocket (v1 API)."""
    url = (
        f"{_DG_V1_WS_URL}"
        f"?model={model}&encoding=linear16&sample_rate={sample_rate}"
        f"&channels=1&interim_results=true&smart_format=true"
        f"&punctuate=true&filler_words=false"
        f"&endpointing={endpointing}&vad_events=true"
        f"&utterance_end_ms=1000&language={language}"
    )
    if keywords:
        kw_param = "keyterm" if model.startswith("nova-3") else "keywords"
        url += "".join(f"&{kw_param}={kw}" for kw in keywords)
    headers = {"Authorization": f"Token {api_key}"}

    for attempt in range(3):
      try:
        async with websockets.connect(url, additional_headers=headers) as dg_ws:
            logger.info("Nova STT connected | model=%s | endpointing=%dms", model, endpointing)
            close_event = asyncio.Event()

            async def send_audio():
                try:
                    while True:
                        chunk = await audio_queue.get()
                        if chunk is None:
                            try:
                                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                            except ConnectionClosed:
                                pass
                            close_event.set()
                            break
                        try:
                            await dg_ws.send(chunk)
                        except ConnectionClosed:
                            break
                except asyncio.CancelledError:
                    pass

            async def keepalive():
                try:
                    while True:
                        await asyncio.sleep(_KEEPALIVE_INTERVAL)
                        await dg_ws.send(json.dumps({"type": "KeepAlive"}))
                except (ConnectionClosed, asyncio.CancelledError):
                    pass

            async def recv():
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(dg_ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            await dg_ws.close()
                            break
                        data = json.loads(raw)
                        mt = data.get("type")
                        if mt == "Results":
                            alts = data.get("channel", {}).get("alternatives", [])
                            if not alts:
                                continue
                            text = alts[0].get("transcript", "").strip()
                            is_final = data.get("is_final", False) or data.get("speech_final", False)
                            if text:
                                await transcript_callback(text, is_final, None)
                            if close_event.is_set() and is_final:
                                await dg_ws.close()
                                return
                        elif mt == "SpeechStarted" and speech_started_callback:
                            try:
                                await speech_started_callback()
                            except Exception:
                                pass
                        elif mt == "UtteranceEnd" and close_event.is_set():
                            await dg_ws.close()
                            return
                except ConnectionClosed:
                    pass
                except Exception as e:
                    logger.error("Nova recv error: %s", e)

            ka = asyncio.create_task(keepalive())
            try:
                await asyncio.gather(send_audio(), recv())
            finally:
                ka.cancel()
        return
      except (ConnectionClosed, OSError) as e:
        if attempt < 2:
            logger.warning("Nova disconnected (attempt %d/3): %s", attempt + 1, e)
            await asyncio.sleep(0.5 * (attempt + 1))
      except Exception as e:
        logger.error("Nova error: %s", e)
        raise


# ═══════════════════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

async def stream_deepgram(
    audio_queue: asyncio.Queue,
    transcript_callback: Callable[[str, bool], Awaitable[None]],
    api_key: str,
    model: str = "nova-2",
    language: str = "en",
    endpointing: int = 300,
    sample_rate: int = 16000,
    speech_started_callback: Optional[Callable[[], Awaitable[None]]] = None,
    keywords: Optional[list] = None,
    enable_sentiment: bool = False,
    detect_language: bool = False,
    language_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    start_of_turn_callback: Optional[Callable[[], Awaitable[None]]] = None,
    end_of_turn_callback: Optional[Callable[[float], Awaitable[None]]] = None,
    # Flux-specific params (ignored for Nova)
    eager_eot_callback: Optional[Callable] = None,
    turn_resumed_callback: Optional[Callable] = None,
    eot_threshold: float = 0.65,
    eager_eot_threshold: Optional[float] = 0.35,
    eot_timeout_ms: int = 1500,
) -> None:
    """Route to Flux SDK or Nova raw WebSocket based on model."""
    if _is_flux_model(model):
        logger.info("Routing to Flux SDK (v2)")
        await _stream_flux_sdk(
            audio_queue=audio_queue,
            transcript_callback=transcript_callback,
            api_key=api_key,
            sample_rate=sample_rate,
            keywords=keywords,
            start_of_turn_callback=start_of_turn_callback,
            end_of_turn_callback=end_of_turn_callback,
            eager_eot_callback=eager_eot_callback,
            turn_resumed_callback=turn_resumed_callback,
            eot_threshold=eot_threshold,
            eager_eot_threshold=eager_eot_threshold,
            eot_timeout_ms=eot_timeout_ms,
        )
    else:
        logger.info("Routing to Nova WebSocket (v1)")
        await _stream_nova_ws(
            audio_queue=audio_queue,
            transcript_callback=transcript_callback,
            api_key=api_key,
            model=model,
            language=language,
            endpointing=endpointing,
            sample_rate=sample_rate,
            speech_started_callback=speech_started_callback,
            keywords=keywords,
        )
