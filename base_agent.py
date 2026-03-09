#
# base_agent.py
# Shared bot runner for pest control agents.
# Both PCO and ECO use this — the only difference is the COMPANY_NAME
# and the agent transfer number, passed via AgentConfig.
#

import os
import json
import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
logger.info("✅ Silero VAD model loaded")

from pipecat.frames.frames import LLMRunFrame, TextFrame, LLMFullResponseEndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
    LLMUserAggregator,
    LLMAssistantAggregator,
)
from pipecat.processors.aggregators.sentence import SentenceAggregator
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.runner.types import RunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.transports.base_transport import BaseTransport

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────
#  Agent configuration dataclass
# ─────────────────────────────────────────────────────────────
@dataclass
class AgentConfig:
    """
    Holds all per-agent settings.
    Pass a filled AgentConfig to run_pest_control_bot() to start a specific agent.
    """
    company_name: str               # e.g. "PCO Pest Control" or "ECO Pest Control"
    elevenlabs_voice_id: str        # ElevenLabs voice ID for this agent
    agent_transfer_number: str      # Phone number to transfer to a live human
    data_file: str                  # Absolute path to the JSON data file


# ─────────────────────────────────────────────────────────────
#  JSON data helpers
# ─────────────────────────────────────────────────────────────
def _load_data(data_file: str) -> dict:
    with open(data_file, "r") as f:
        return json.load(f)

def _save_data(data_file: str, data: dict):
    with open(data_file, "w") as f:
        json.dump(data, f, indent=2)

def _normalize_phone(phone: str) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


# ─────────────────────────────────────────────────────────────
#  JSON API helpers
# ─────────────────────────────────────────────────────────────
def _json_lookup_account(data_file: str, phone_number: str) -> dict:
    data = _load_data(data_file)
    norm = _normalize_phone(phone_number)
    for c in data["customers"]:
        if _normalize_phone(c["phone_number"]) == norm:
            return {
                "found": True,
                "customer_id": c["customer_id"],
                "customer_name": c["customer_name"],
                "email": c["email"],
                "phone": c["phone_number"],
            }
    return {"found": False}

def _json_get_next_appointment(data_file: str, phone_number: str) -> dict:
    data = _load_data(data_file)
    norm = _normalize_phone(phone_number)
    for a in data["appointments"]:
        if _normalize_phone(a["phone_number"]) == norm:
            return {
                "found": True,
                "appointment_date": a["appointment_date"],
                "start_time": a["start_time"],
                "end_time": a["end_time"],
                "service_type": a["service_type"],
            }
    return {"found": False}

def _json_check_zip(data_file: str, zipcode: str) -> dict:
    data = _load_data(data_file)
    serviced = zipcode.strip() in data.get("service_areas", [])
    return {"serviced": serviced, "zipcode": zipcode.strip()}

def _json_list_services(data_file: str) -> dict:
    data = _load_data(data_file)
    return {"services": data.get("services", [])}

def _json_get_plan_price(data_file: str, service_type: str, property_type: str, plan_type: str) -> dict:
    data = _load_data(data_file)
    pricing = data.get("pricing", {})
    svc = pricing.get(service_type)
    if not svc:
        return {"error": f"Service '{service_type}' not found", "found": False}
    prop = svc.get(property_type)
    if not prop:
        return {"error": f"Property type '{property_type}' not found", "found": False}
    plan = prop.get(plan_type)
    if not plan:
        available = list(prop.keys())
        plan = prop.get(available[0])
        plan_type = available[0]
    return {"found": True, "plan_type": plan_type, **plan}

def _json_book_service(data_file: str, booking_data: dict) -> dict:
    data = _load_data(data_file)
    booking_id = f"BK-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:4].upper()}"
    booking_data["booking_id"] = booking_id
    booking_data["created_at"] = datetime.now().isoformat()
    data.setdefault("bookings", []).append(booking_data)
    _save_data(data_file, data)
    logger.info(f"✅ Booking saved: {booking_id}")
    return {"success": True, "booking_id": booking_id, "message": "Booking confirmed"}

def _json_notify_office(data_file: str, phone_number: str, message: str) -> dict:
    data = _load_data(data_file)
    notification = {
        "id": str(uuid.uuid4())[:8],
        "phone_number": phone_number,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }
    data.setdefault("office_notifications", []).append(notification)
    _save_data(data_file, data)
    logger.info(f"📋 Office notified: {message}")
    return {"success": True, "notification_id": notification["id"]}


# ─────────────────────────────────────────────────────────────
#  Passthrough aggregator so TTS can hear assistant text
# ─────────────────────────────────────────────────────────────
class PassthroughAssistantAggregator(LLMAssistantAggregator):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


# ─────────────────────────────────────────────────────────────
#  System prompt builder
# ─────────────────────────────────────────────────────────────
def build_system_prompt(company_name: str) -> str:
    return f"""You are an AI voice assistant for {company_name}. You speak like a warm, friendly human representative — casual, natural, and genuinely helpful. Every response should feel like a real conversation, not a scripted interview. Never rush through questions. React to what the caller says before moving forward.
AUDIO EXPRESSION RULES
Dynamically integrate audio tags into every response to make speech expressive and engaging.
Audio tags must be enclosed in square brackets (e.g., [happy], [sighs]).
Tags must only describe something auditory related to the voice. Never use tags for music, sound effects, or physical actions.
Place tags immediately before or after the dialogue segment they modify.
You may add emphasis by capitalizing certain words, adding ellipses, exclamation marks, or question marks where naturally appropriate — but never alter, add, or remove any intended response words.
Available tags include but are not limited to:
Emotional directions: [happy], [warm], [excited], [surprised], [thoughtful], [empathetic], [whisper]
Non-verbal sounds: [laughing], [chuckles], [sighs], [clears throat], [short pause], [long pause], [exhales sharply], [inhales deeply]
CORE RULES
Ask only ONE question at a time and genuinely wait for the response before continuing. Never repeat information the caller has already shared. Always react naturally to what the caller says — acknowledge it, then smoothly transition to the next thing. Keep responses short and conversational — 1 to 3 sentences max. Stay on pest control topics only. If asked about something unrelated, say: "[apologetic] Oh, I wish I could help with that! I'm really only set up for pest control questions — want me to get a live person on the line for you?" Never sound like you're reading from a script. Flow naturally.
LOOKUP DATA (Internal — never read aloud)
Account Database: Phone: 555-867-5309 → Name: Sarah Mitchell, Appointment: March 15, 2025, 9:00 AM – 12:00 PM, Service: General Pest Control Phone: 555-234-7890 → Name: James Ortega, No upcoming appointments Any other number → Account not found
Zip Code Database: 90210 → Serviced 30301 → Serviced 73301 → Not serviced Any other zip → Not serviced (re-ask once, then offer specialist)
Pricing: Single-family All Season Plan: $149 initial visit, then $89/quarter Single-family One-Time Service: $199 (30-day guarantee) Multi-unit / Condo / Apartment One-Time Service: $249 (30-day guarantee)
CONVERSATION FLOW
IMPORTANT: This is a guide, not a rigid script. Adapt naturally based on what the caller says. React first, then move forward.
GREETING
Say something like: "[happy] Hey there, thanks so much for calling {company_name}! [warm] Have you called us before, or is this your first time reaching out?"
2A. RETURNING CUSTOMER
React warmly, then naturally ask: "[warm] Oh great, welcome back! [short pause] And what number do we have on file for you?"
If found with appointment: "[excited] Oh perfect, I've got you right here! So [FirstName], looks like you're all set for [Date] somewhere between [StartTime] and [EndTime] for your [ServiceType] — nice! [warm] Is there something you wanted to change, or just checking in?"
If found without appointment: "[thoughtful] Got your account pulled up! Looks like there's nothing scheduled at the moment though. [warm] Were you thinking about booking something, or did you have a question?"
If not found: "[short pause] Hmm, I'm not seeing anything under that number. [warm] Do you happen to have another number it might be under?"
If still not found: "[empathetic] No worries at all — I'm going to flag this for our office team and they'll personally follow up with you. [warm] Is there anything else I can help with in the meantime, or would you rather just speak with someone directly?"
2B. NEW CUSTOMER — SERVICE AREA
React with warmth, then naturally ease into it: "[excited] Oh awesome, welcome! We'd love to help you out. [short pause] Just so I can make sure we cover your area — what zip code are you in?"
If found: "[excited] Oh great news, we definitely service your area! [happy] You're in good hands."
If not found first try: "[thoughtful] Hmm, let me just double check that — could you read that zip code to me one more time?"
If still not found: "[empathetic] I'm having a little trouble confirming that area on my end. [warm] I can pull in one of our specialists who can check this manually for you — would that work?"
If yes: "[happy] Perfect, connecting you right now!"
If no: "[warm] Totally understandable. Unfortunately it does look like we might not cover that area, but thank you SO much for calling — we really appreciate it!"
PROPERTY TYPE
Transition naturally: "[warm] So just so I can point you in the right direction — are we talking about a house, or more of an apartment or condo situation?"
Internally note single_family or multi_family — never say this out loud.
COLLECTING INFORMATION — CONVERSATIONALLY
Do NOT fire questions back to back. After each answer, briefly acknowledge it naturally before asking the next one. Rotate acknowledgment phrases so it never feels repetitive. Examples: "Perfect!", "Got it!", "Oh okay!", "Nice, thank you!", "That's helpful to know!"
Collect in this natural order, one at a time:
Name — "[warm] And who am I speaking with today?"
After they answer: "[happy] [Name], great to meet you! [short pause] And what's a good number to reach you on?"
Pest issue — "[curious] So what's been going on — what kind of pest situation are you dealing with?"
React to what they say before moving on. If it sounds bad, show empathy: "[empathetic] Oh wow, yeah that's definitely something we can take care of!"
Square footage — "[thoughtful] Roughly how big is the place — do you know about how many square feet?"
If they're unsure: "[warm] No worries, even a rough ballpark totally works!"
SERVICE AND PRICING — CONVERSATIONAL
Present options naturally, not like a price list.
If single-family home: "[warm] So for a house like yours, what most of our customers go with is our All Season Plan. [thoughtful] Basically we come out for the first visit, then check back in three weeks later, and after that it's quarterly visits to keep everything under control year-round. [happy] And if anything comes back between visits, we come back out — no extra charge. The first visit runs $149, then it's just $89 every quarter after that. [warm] Does that kind of ongoing coverage sound like what you're looking for?"
If YES: naturally continue into booking. If NO: "[warm] Totally fair! We also do a one-time treatment if you just want to tackle this and see how it goes — that's $199 and comes with a 30-day guarantee. [happy] Want to go that route?"
If multi-unit / apartment / condo: "[warm] So for your type of property, we do a one-time full treatment — interior and exterior — and it comes with a 30-day guarantee. That's $249. [thoughtful] Does that work for what you're looking for?"
BOOKING DETAILS — CONVERSATIONALLY
Collect naturally one at a time, acknowledging each answer:
Address — "[warm] Perfect! And what's the address we'd be coming out to?"
City — "[short pause] And what city is that in?"
Preferred day — "[thoughtful] What day or days work best for you generally?"
Time preference — "[warm] And would morning or afternoon be better for you?"
Email — "[happy] Almost there! What email should we send your service report to?"
Contact preference — "[warm] And would you prefer we reach out by call or text for any updates?"
CONFIRMATION
Don't read it robotically — make it sound like you're just running through it naturally:
"[thoughtful] Okay, let me just make sure I've got everything right here. [short pause] So we've got [Name] at [Address] in [City], zip [Zip] — we're coming out for [Pest Type], going with the [Plan Name] at [Price]. [warm] You're looking at [Day] in the [Morning/Afternoon], reports going to [Email], and we'll keep in touch by [call/text]. [happy] Does all that sound good to you?"
After confirmation: "[excited] Perfect! Let me just get that submitted for you. [long pause] Okay, all done! [happy] Our scheduling team has everything and they'll be reaching out soon to lock in your exact time."
AGENT TRANSFER
Never make it feel like a handoff — make it feel like a favor:
"[warm] Absolutely, let me get you over to one of our people right now — they'll be able to sort this out for you. Just one moment!"
CLOSING
End warmly and genuinely: "[happy] It was so nice chatting with you! [warm] If you ever need anything at all, don't hesitate to call us back. Thanks for choosing {company_name} — hope you have an amazing rest of your day!"
"""


# ─────────────────────────────────────────────────────────────
#  MAIN BOT RUNNER
# ─────────────────────────────────────────────────────────────
async def run_pest_control_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    config: AgentConfig,
    handle_sigint: bool = True,
    log_callback=None,
    transfer_callback=None,
):
    logger.info(f"Starting {config.company_name} bot")

    # ── AI Services ──
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=config.elevenlabs_voice_id,
        model_id="eleven_v3",
        params=ElevenLabsTTSService.InputParams(
            stability=0.2,
            similarity_boost=0.5,
            style=1.0,
            use_speaker_boost=True,
            apply_text_normalization="off",
        ),
    )

    llm = AWSBedrockLLMService(
        model=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    system_prompt = build_system_prompt(config.company_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "(Call Started)"},
    ]

    # ─────────────────────────────────────────
    #  TOOL IMPLEMENTATIONS
    # ─────────────────────────────────────────

    async def lookup_account(params: FunctionCallParams):
        phone = params.arguments.get("phone_number", "").strip()
        logger.info(f"[lookup_account] phone={phone}")
        result = _json_lookup_account(config.data_file, phone)
        logger.info(f"[lookup_account] result={result}")
        await params.result_callback(result)

    async def get_next_appointment_by_phone(params: FunctionCallParams):
        phone = params.arguments.get("phone_number", "").strip()
        logger.info(f"[get_next_appointment_by_phone] phone={phone}")
        result = _json_get_next_appointment(config.data_file, phone)
        logger.info(f"[get_next_appointment_by_phone] result={result}")
        await params.result_callback(result)

    async def check_zip(params: FunctionCallParams):
        zipcode = params.arguments.get("zipcode", "").strip()
        logger.info(f"[check_zip] zipcode={zipcode}")
        result = _json_check_zip(config.data_file, zipcode)
        logger.info(f"[check_zip] result={result}")
        await params.result_callback(result)

    async def list_services(params: FunctionCallParams):
        logger.info("[list_services] fetching services")
        result = _json_list_services(config.data_file)
        logger.info(f"[list_services] {len(result.get('services', []))} services")
        await params.result_callback(result)

    async def get_plan_price_details(params: FunctionCallParams):
        service_type  = params.arguments.get("service_type", "")
        property_type = params.arguments.get("property_type", "single_family")
        plan_type     = params.arguments.get("plan_type", "recurring")
        logger.info(f"[get_plan_price_details] service={service_type}, property={property_type}, plan={plan_type}")
        result = _json_get_plan_price(config.data_file, service_type, property_type, plan_type)
        logger.info(f"[get_plan_price_details] result={result}")
        await params.result_callback(result)

    async def book_service(params: FunctionCallParams):
        booking_data = {
            "company":            config.company_name,
            "name":               params.arguments.get("name", ""),
            "phone":              params.arguments.get("phone", ""),
            "email":              params.arguments.get("email", ""),
            "address":            params.arguments.get("address", ""),
            "city":               params.arguments.get("city", ""),
            "zipcode":            params.arguments.get("zipcode", ""),
            "pest_type":          params.arguments.get("pest_type", ""),
            "service_type":       params.arguments.get("service_type", ""),
            "property_type":      params.arguments.get("property_type", "single_family"),
            "plan_name":          params.arguments.get("plan_name", ""),
            "price":              params.arguments.get("price", ""),
            "preferred_date":     params.arguments.get("preferred_date", ""),
            "preferred_time":     params.arguments.get("preferred_time", ""),
            "contact_preference": params.arguments.get("contact_preference", ""),
            "square_footage":     params.arguments.get("square_footage", ""),
        }
        logger.info(f"[book_service] saving booking for {config.company_name}: {booking_data}")
        result = _json_book_service(config.data_file, booking_data)
        logger.info(f"[book_service] result={result}")
        await params.result_callback(result)

    async def notify_office(params: FunctionCallParams):
        message = params.arguments.get("message", "")
        phone   = params.arguments.get("phone_number", "")
        logger.info(f"[notify_office] phone={phone}, message={message}")
        result = _json_notify_office(config.data_file, phone, message)
        await params.result_callback(result)

    async def transfer_to_agent(params: FunctionCallParams):
        logger.info(f"[transfer_to_agent] Transferring to live agent at {config.agent_transfer_number}")
        if transfer_callback:
            await transfer_callback(config.agent_transfer_number)
            await params.result_callback({
                "status": "transfer_initiated",
                "agent_number": config.agent_transfer_number,
            })
        elif hasattr(transport, "start_dialout"):
            await transport.start_dialout({"phoneNumber": config.agent_transfer_number})
            await params.result_callback({
                "status": "transfer_initiated",
                "agent_number": config.agent_transfer_number,
            })
        else:
            logger.error("Transport does not support dial-out and no transfer_callback provided.")
            await params.result_callback({
                "status": "error",
                "message": "Cannot transfer on this transport.",
            })

    # ─────────────────────────────────────────
    #  TOOL SCHEMAS
    # ─────────────────────────────────────────

    tool_lookup_account = FunctionSchema(
        name="lookup_account",
        description="Look up a customer account by phone number.",
        properties={
            "phone_number": {"type": "string", "description": "Customer phone number, digits only or with country code."}
        },
        required=["phone_number"],
    )

    tool_get_next_appointment = FunctionSchema(
        name="get_next_appointment_by_phone",
        description="Get the next upcoming appointment for a customer by phone number.",
        properties={
            "phone_number": {"type": "string", "description": "Customer phone number used for account lookup."}
        },
        required=["phone_number"],
    )

    tool_check_zip = FunctionSchema(
        name="check_zip",
        description="Check whether a given zipcode is within the service area.",
        properties={
            "zipcode": {"type": "string", "description": "5-digit US zipcode to check."}
        },
        required=["zipcode"],
    )

    tool_list_services = FunctionSchema(
        name="list_services",
        description="List all available pest control services. MUST be called before any service selection logic.",
        properties={},
        required=[],
    )

    tool_get_plan_price = FunctionSchema(
        name="get_plan_price_details",
        description=(
            "Get pricing details for a pest control plan. "
            "MUST include both service_type (service id from list_services) and property_type. "
            "Use plan_type='recurring' for subscription plans and plan_type='one_time' for one-time service."
        ),
        properties={
            "service_type":  {"type": "string", "description": "Service ID returned from list_services."},
            "property_type": {"type": "string", "enum": ["single_family", "multi_family"], "description": "Property type."},
            "plan_type":     {"type": "string", "enum": ["recurring", "one_time"], "description": "Type of plan."},
        },
        required=["service_type", "property_type", "plan_type"],
    )

    tool_book_service = FunctionSchema(
        name="book_service",
        description="Book a pest control service after customer confirms all details.",
        properties={
            "name":               {"type": "string", "description": "Customer full name."},
            "phone":              {"type": "string", "description": "Customer phone number."},
            "email":              {"type": "string", "description": "Customer email."},
            "address":            {"type": "string", "description": "Street address."},
            "city":               {"type": "string", "description": "City."},
            "zipcode":            {"type": "string", "description": "5-digit zipcode."},
            "pest_type":          {"type": "string", "description": "Type of pest problem."},
            "service_type":       {"type": "string", "description": "Service ID from list_services."},
            "property_type":      {"type": "string", "enum": ["single_family", "multi_family"]},
            "plan_name":          {"type": "string", "description": "Name of the selected plan."},
            "price":              {"type": "string", "description": "Confirmed price."},
            "preferred_date":     {"type": "string", "description": "Preferred service date."},
            "preferred_time":     {"type": "string", "enum": ["AM", "PM"], "description": "Preferred time slot."},
            "contact_preference": {"type": "string", "enum": ["call", "text"], "description": "Preferred contact method."},
            "square_footage":     {"type": "string", "description": "Approximate home square footage."},
        },
        required=["name", "phone", "email", "address", "city", "zipcode", "pest_type",
                  "service_type", "property_type", "plan_name", "price", "preferred_date", "preferred_time"],
    )

    tool_notify_office = FunctionSchema(
        name="notify_office",
        description="Send a notification to the office when unable to look up a customer account or handle a request.",
        properties={
            "phone_number": {"type": "string", "description": "Customer phone number if available."},
            "message":      {"type": "string", "description": "Description of the issue or request."},
        },
        required=["message"],
    )

    tool_transfer_to_agent = FunctionSchema(
        name="transfer_to_agent",
        description="Transfer the call to a live human agent. Call this whenever the customer requests an agent or when you cannot help.",
        properties={},
        required=[],
    )

    # ── Register tools ──
    tools = ToolsSchema(standard_tools=[
        tool_lookup_account,
        tool_get_next_appointment,
        tool_check_zip,
        tool_list_services,
        tool_get_plan_price,
        tool_book_service,
        tool_notify_office,
        tool_transfer_to_agent,
    ])

    context = LLMContext(messages, tools=tools)

    llm.register_function("lookup_account",              lookup_account)
    llm.register_function("get_next_appointment_by_phone", get_next_appointment_by_phone)
    llm.register_function("check_zip",                   check_zip)
    llm.register_function("list_services",               list_services)
    llm.register_function("get_plan_price_details",      get_plan_price_details)
    llm.register_function("book_service",                book_service)
    llm.register_function("notify_office",               notify_office)
    llm.register_function("transfer_to_agent",           transfer_to_agent)

    # ── Aggregators ──
    user_aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.4)
            )
        ),
    )
    assistant_aggregator = PassthroughAssistantAggregator(context)

    # ── Pipeline stages ──
    conv_logger = None
    if log_callback:
        from conversation_logger import ConversationLogger
        conv_logger = ConversationLogger(on_log=log_callback)

    pipeline_stages = [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        assistant_aggregator,
    ]
    if conv_logger:
        pipeline_stages.append(conv_logger)
    pipeline_stages += [
        SentenceAggregator(),
        tts,
        transport.output(),
    ]

    pipeline = Pipeline(pipeline_stages)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected — starting {config.company_name} conversation")
        await asyncio.sleep(1.0)
        greeting = (
            f"Oh, hi there! Thanks for calling {config.company_name}. "
            f"Are you a current customer, or are you looking for some help for the first time?"
        )
        logger.info(f"Queuing initial greeting: {greeting}")
        messages.append({"role": "assistant", "content": greeting})
        await task.queue_frames([TextFrame(greeting), LLMFullResponseEndFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)
