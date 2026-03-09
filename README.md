# 🐛 Pest Control Voice Bot — Multi-Agent (PCO + ECO)

A production-ready AI voice phone bot built with **Pipecat**, supporting two
independently branded pest control companies on the same server. Each incoming
Twilio call is routed to the correct agent based on the dialed phone number.

---

## Tech Stack

| Layer | Service |
|---|---|
| **STT** | Deepgram |
| **LLM** | AWS Bedrock (Claude 3 Haiku / Sonnet) |
| **TTS** | ElevenLabs (`eleven_v3` model) |
| **Voice Transport** | Twilio Media Streams via WebSocket |
| **Framework** | [Pipecat](https://github.com/pipecat-ai/pipecat) |

---

## Project Structure

```
eco/
├── main.py                  # FastAPI server + phone-number routing
├── base_agent.py            # Shared bot engine (both agents use this)
├── conversation_logger.py   # Real-time transcript logger
├── data/
│   └── pest_control_data.json   # Customers, appointments, pricing, bookings
├── .env.example             # Env variable template
└── pyproject.toml           # Python dependencies
```

---

## Agent Routing Logic

```
Incoming Call (Twilio)
        │
        ▼
  /bot  webhook  ─── reads "To" number ──►  PCO_PHONE_NUMBERS?  ─►  PCO Pest Control Agent
                                                                 ─►  ECO Pest Control Agent  (default)
```

Each agent has its own:
- **Company name** (shown in greeting & closing)
- **ElevenLabs voice ID**
- **Human-agent transfer number**

They share:
- Identical conversation flow & prompt
- The same Deepgram / Bedrock / ElevenLabs keys
- The same `data/pest_control_data.json` file
- The same tool set (lookup, zip check, booking, transfer…)

---

## Setup

### 1. Install dependencies

```bash
pip install uv          # if not already installed
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in all keys
```

Key variables:

```env
# Which Twilio numbers route to which agent
PCO_PHONE_NUMBERS=+15551234567,+15559999999
ECO_PHONE_NUMBERS=+15550001111

# Human-agent transfer targets
PCO_AGENT_NUMBER=+15553334444
ECO_AGENT_NUMBER=+15555556666

# ElevenLabs voice per agent (can be the same)
PCO_ELEVENLABS_VOICE_ID=g6xIsTj2HwM6VR4iXFCw
ECO_ELEVENLABS_VOICE_ID=g6xIsTj2HwM6VR4iXFCw
```

### 3. Run the server

```bash
uv run main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8765
```

### 4. Expose to the internet (for Twilio)

```bash
ngrok http 8765
```

Set your Twilio phone number's **Voice Webhook** to:
```
https://<your-ngrok-id>.ngrok.io/bot   [HTTP POST]
```

---

## Live Monitoring

Visit `http://localhost:8765/` in a browser while a call is active to see the
real-time conversation transcript with colour-coded turns.

---

## Agent Transfer (Handoff)

When the caller requests a human, or when the bot cannot handle a request,
it calls the `transfer_to_agent` tool. The server then:

1. Calls Twilio's REST API with `<Dial>` TwiML
2. The call is bridged to `PCO_AGENT_NUMBER` or `ECO_AGENT_NUMBER`
   (whichever matches the agent that answered)

---

## Data File

`data/pest_control_data.json` is used as a local database for:
- **customers** — account lookup by phone
- **appointments** — next scheduled visit
- **service_areas** — zip codes served
- **services** — list of pest control services
- **pricing** — plan prices by service/property/plan type
- **bookings** — saved when `book_service` is called
- **office_notifications** — saved when `notify_office` is called

---

## Customising

To add a **third company** (e.g. "XYZ Pest Control"):

1. Add `XYZ_PHONE_NUMBERS` and `XYZ_AGENT_NUMBER` to `.env`
2. Create an `XYZ_CONFIG = AgentConfig(...)` in `main.py`
3. Add an `elif` branch in `resolve_agent_config()`

No other changes needed — the bot engine in `base_agent.py` handles everything.
