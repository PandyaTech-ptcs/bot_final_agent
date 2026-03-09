"""
Multi-Agent Pest Control Voice Bot — Main Server
================================================

Phone-number routing:
  - Numbers listed in PCO_PHONE_NUMBERS  →  PCO Pest Control agent
  - Numbers listed in ECO_PHONE_NUMBERS  →  ECO Pest Control agent

Environment variables (see .env.example):
  PCO_PHONE_NUMBERS   comma-separated Twilio numbers for PCO
  ECO_PHONE_NUMBERS   comma-separated Twilio numbers for ECO
  PCO_AGENT_NUMBER    human-agent transfer number for PCO
  ECO_AGENT_NUMBER    human-agent transfer number for ECO
  PCO_ELEVENLABS_VOICE_ID  ElevenLabs voice for PCO
  ECO_ELEVENLABS_VOICE_ID  ElevenLabs voice for ECO
  (plus Deepgram, Bedrock, Twilio, ElevenLabs keys)

Run:
    uv run main.py
"""

import os
import sys
import asyncio
import json

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from loguru import logger
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_agent import run_pest_control_bot, AgentConfig
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.runner.types import RunnerArguments
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from twilio.rest import Client

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────
#  Data file path (shared by both agents)
# ─────────────────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pest_control_data.json")

# ─────────────────────────────────────────────────────────────
#  Phone → agent routing tables
# ─────────────────────────────────────────────────────────────
def _parse_numbers(env_var: str) -> set[str]:
    """Parse a comma-separated env variable into a normalised set of phone numbers."""
    raw = os.getenv(env_var, "")
    return {n.strip() for n in raw.split(",") if n.strip()}

PCO_NUMBERS: set[str] = _parse_numbers("PCO_PHONE_NUMBERS")
ECO_NUMBERS: set[str] = _parse_numbers("ECO_PHONE_NUMBERS")

logger.info(f"PCO numbers: {PCO_NUMBERS}")
logger.info(f"ECO numbers: {ECO_NUMBERS}")

# ─────────────────────────────────────────────────────────────
#  Agent configs (Loaded from JSON)
# ─────────────────────────────────────────────────────────────
AGENTS_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents_config.json")

def _load_agents_config():
    try:
        with open(AGENTS_CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {AGENTS_CONFIG_FILE}: {e}")
        return {}

_agents_data = _load_agents_config()
_pco_data = _agents_data.get("pco", {})
_eco_data = _agents_data.get("eco", {})

PCO_CONFIG = AgentConfig(
    company_name=_pco_data.get("company_name", "PCO Pest Control"),
    elevenlabs_voice_id=_pco_data.get("voice_id", os.getenv("PCO_ELEVENLABS_VOICE_ID", "g6xIsTj2HwM6VR4iXFCw")),
    agent_transfer_number=_pco_data.get("agent_transfer_number", os.getenv("PCO_AGENT_NUMBER", "+15553334444")),
    data_file=DATA_FILE,
    system_prompt=_pco_data.get("system_prompt", ""),
)

ECO_CONFIG = AgentConfig(
    company_name=_eco_data.get("company_name", "ECO Pest Control"),
    elevenlabs_voice_id=_eco_data.get("voice_id", os.getenv("ECO_ELEVENLABS_VOICE_ID", "g6xIsTj2HwM6VR4iXFCw")),
    agent_transfer_number=_eco_data.get("agent_transfer_number", os.getenv("ECO_AGENT_NUMBER", "+15555556666")),
    data_file=DATA_FILE,
    system_prompt=_eco_data.get("system_prompt", ""),
)

def resolve_agent_config(to_number: str) -> AgentConfig:
    """
    Return the correct AgentConfig based on the Twilio 'To' number.
    Falls back to ECO Pest Control if the number is not mapped.
    """
    if to_number in PCO_NUMBERS:
        logger.info(f"Routing {to_number} → PCO Pest Control")
        return PCO_CONFIG
    elif to_number in ECO_NUMBERS:
        logger.info(f"Routing {to_number} → ECO Pest Control")
        return ECO_CONFIG
    else:
        logger.warning(f"Unknown To number '{to_number}' — defaulting to ECO Pest Control")
        return ECO_CONFIG


# ─────────────────────────────────────────────────────────────
#  WebSocket log broadcaster (live call monitoring)
# ─────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass


manager = ConnectionManager()

# ─────────────────────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Pest Control Voice Bot — Multi-Agent")


# ── Live monitoring page ──
@app.get("/")
async def index():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐛 Pest Control — Live Monitor</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a14;
    color: #e2e8f0;
    padding: 28px;
    min-height: 100vh;
  }
  header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }
  header h1 { font-size: 1.35rem; font-weight: 700; color: #f8fafc; }
  .badge {
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 99px;
    text-transform: uppercase;
    letter-spacing: .05em;
  }
  .badge-pco { background: #1e3a5f; color: #60a5fa; }
  .badge-eco { background: #14352a; color: #4ade80; }
  .sub { color: #64748b; font-size: 0.82rem; margin-bottom: 20px; }
  .dot {
    display: inline-block;
    width: 9px; height: 9px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
  #logs {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 18px;
    height: 74vh;
    overflow-y: auto;
    font-size: 0.875rem;
    line-height: 1.75;
    font-family: 'Inter', monospace;
  }
  .user { color: #60a5fa; }
  .bot  { color: #f59e0b; }
  .sys  { color: #475569; font-style: italic; }
  .agent-pco { color: #818cf8; font-weight: 600; }
  .agent-eco { color: #34d399; font-weight: 600; }
  #logs::-webkit-scrollbar { width: 6px; }
  #logs::-webkit-scrollbar-track { background: transparent; }
  #logs::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
</head>
<body>
  <header>
    <h1><span class="dot"></span>🐛 Pest Control — Live Call Monitor</h1>
    <span class="badge badge-pco">PCO</span>
    <span class="badge badge-eco">ECO</span>
  </header>
  <p class="sub">Real-time conversation log &nbsp;|&nbsp; <span style="color:#60a5fa">Blue = Customer</span> &nbsp;|&nbsp; <span style="color:#f59e0b">Amber = Bot</span></p>
  <div id="logs"><div class="sys">Waiting for incoming calls...</div></div>
  <script>
    var ws = new WebSocket("ws://" + window.location.host + "/logs_ws");
    ws.onmessage = function(e) {
      var logs = document.getElementById('logs');
      var msg = e.data;
      var cls = "sys";
      if (msg.startsWith("User:"))       cls = "user";
      else if (msg.startsWith("Bot:"))   cls = "bot";
      else if (msg.includes("[PCO]"))    cls = "agent-pco";
      else if (msg.includes("[ECO]"))    cls = "agent-eco";
      var el = document.createElement('div');
      el.textContent = msg;
      el.className = cls;
      logs.appendChild(el);
      logs.scrollTop = logs.scrollHeight;
    };
    ws.onclose = function() {
      var el = document.createElement('div');
      el.textContent = "⚠️ Connection lost. Refresh to reconnect.";
      el.className = "sys";
      document.getElementById('logs').appendChild(el);
    };
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.websocket("/logs_ws")
async def logs_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────────
#  TWILIO WEBHOOK — returns TwiML to connect the call
# ─────────────────────────────────────────────────────────────
@app.post("/bot")
async def bot_entry(request: Request):
    host = request.headers.get("host", "localhost:8765")
    form_data = await request.form()
    to_number  = form_data.get("To", "")
    call_sid   = form_data.get("CallSid", "")

    ws_url = f"wss://{host}/ws"
    logger.info(f"Incoming call → To: {to_number}  CallSid: {call_sid}  WS: {ws_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="to_number" value="{to_number}" />
      <Parameter name="call_sid" value="{call_sid}" />
    </Stream>
  </Connect>
</Response>"""

    return HTMLResponse(content=twiml, media_type="application/xml")


# ─────────────────────────────────────────────────────────────
#  TWILIO WEBSOCKET — handles the live audio stream
# ─────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio WebSocket connected")

    stream_sid = None
    call_sid   = None
    to_number  = None

    # ── Wait for the Twilio "start" event ──
    try:
        while True:
            msg   = await websocket.receive_json()
            event = msg.get("event")

            if event == "connected":
                logger.info("Twilio media stream: connected")
                continue

            elif event == "start":
                stream_sid = msg["start"]["streamSid"]
                params      = msg["start"].get("customParameters", {})
                to_number   = params.get("to_number", "")
                call_sid    = params.get("call_sid", "")
                logger.info(f"Stream started — sid={stream_sid}  to={to_number}  call_sid={call_sid}")
                break

            elif event == "stop":
                logger.info("Stream stopped before start")
                await websocket.close()
                return

    except Exception as e:
        logger.error(f"Error waiting for start event: {e}")
        await websocket.close()
        return

    if not stream_sid:
        logger.error("No streamSid — closing")
        await websocket.close()
        return

    # ── Resolve which agent to use ──
    agent_config = resolve_agent_config(to_number)
    await manager.broadcast(f"[{agent_config.company_name.split()[0].upper()}] New call connected → {to_number}")

    # ── Transport ──
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_out_sample_rate=8000,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(confidence=0.8)),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(
                stream_sid,
                params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
            ),
        ),
    )

    runner_args = RunnerArguments()

    # ── Log callback ──
    async def broadcast_log(text: str):
        await manager.broadcast(text)

    # ── Transfer callback (Twilio API) ──
    async def transfer_call(target_number: str):
        logger.info(f"Transferring call {call_sid} → {target_number}")
        try:
            twilio_sid   = os.getenv("TWILIO_ACCOUNT_SID")
            twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
            if not twilio_sid or not twilio_token:
                logger.error("Twilio credentials missing in .env")
                return
            client = Client(twilio_sid, twilio_token)
            twiml  = f"<Response><Dial>{target_number}</Dial></Response>"
            client.calls(call_sid).update(twiml=twiml)
            logger.info(f"Transfer TwiML sent → {target_number}")
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Transfer failed: {e}")

    # ── Launch bot ──
    try:
        await run_pest_control_bot(
            transport=transport,
            runner_args=runner_args,
            config=agent_config,
            handle_sigint=False,
            log_callback=broadcast_log,
            transfer_callback=transfer_call,
        )
    except Exception as e:
        logger.error(f"Bot error: {e}")
        await websocket.close()


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8765))
    uvicorn.run(app, host="0.0.0.0", port=port)
