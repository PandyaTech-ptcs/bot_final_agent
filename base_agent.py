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
import httpx
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
    system_prompt: str = ""         # (Optional) Custom system prompt template
    company_id: int = 0             # Internal company ID from lookup API


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
def build_system_prompt(config: AgentConfig) -> str:
    # Use custom prompt if provided, else fallback to a minimal default
    template = config.system_prompt if config.system_prompt else f"You are an AI voice assistant for {config.company_name}."
    
    # Try to safely format the template with {company_name} if it exists in the string
    try:
        if "{company_name}" in template:
            return template.format(company_name=config.company_name)
    except Exception as e:
        logger.warning(f"Could not format system prompt: {e}")
        
    return template


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
    conversation_id=None,
    to_number=None,
    from_number=None,
):
    logger.info(f"Starting {config.company_name} bot (ID: {conversation_id}, To: {to_number}, From: {from_number})")

    # ── Fetch Caller Details (at call start) ──
    external_data = {"found": False}
    api_base = os.getenv("COMPANY_LOOKUP_API_URL", "").replace("/api/get-company-by-number", "")
    if api_base and from_number:
        try:
            # Twilio numbers sometimes have +, remove it for consistency
            clean_from = from_number.replace("+", "").strip()
            url = f"{api_base}/api/lookup-account"
            async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
                logger.info(f"Checking for existing customer at {url} (ID: {config.company_id}, From: {clean_from})")
                resp = await client.post(url, json={"company_id": config.company_id, "phone": clean_from})
                if resp.status_code == 200:
                    external_data = resp.json()
                    # Mark as found if success is true or it contains data
                    if external_data.get("success"):
                        external_data["found"] = True
                        logger.info(f"✅ Caller recognized: {external_data.get('customer_name')}")
                    else:
                        logger.info("Caller not recognized")
                else:
                    logger.warning(f"Lookup API error: {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to lookup caller: {e}")

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

    system_prompt = build_system_prompt(config)

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if external_data.get("found"):
        customer_name = external_data.get("customer_name") or external_data.get("name", "Customer")
        messages.append({
            "role": "system",
            "content": (
                f"CALLER RECOGNIZED: The caller is {customer_name}. "
                f"Account Data: {json.dumps(external_data)}. "
                "CRITICAL: Do NOT ask if they are a customer. Do NOT ask for their phone number. "
                "The user has already been greeted by name. Directly ask how you can help."
            )
        })
    else:
        # Instructions for unknown caller
        messages.append({
            "role": "system",
            "content": (
                "CALLER UNKNOWN: If the caller says they are a RETURNING/EXISTING customer calling from a different number, "
                "you MUST ask for their phone number so you can look up their account using the 'lookup_account' tool. "
                "If they say they are a NEW customer, you MUST start the flow by asking for their Zip Code."
            )
        })

    messages.append({"role": "user", "content": "(Call Started)"})

    # ─────────────────────────────────────────
    #  TOOL IMPLEMENTATIONS
    # ─────────────────────────────────────────

    async def lookup_account(params: FunctionCallParams):
        phone = params.arguments.get("phone_number", "").strip()
        logger.info(f"[lookup_account] phone={phone}, company_id={config.company_id}")

        # Priority: External API
        api_base = os.getenv("COMPANY_LOOKUP_API_URL", "").replace("/api/get-company-by-number", "")
        if api_base:
            url = f"{api_base}/api/lookup-account"
            try:
                async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
                    resp = await client.post(url, json={"company_id": config.company_id, "phone": phone})
                    if resp.status_code == 200:
                        result = resp.json()
                        logger.info(f"[lookup_account] External API result: {result}")
                        await params.result_callback(result)
                        return
                    else:
                        logger.warning(f"[lookup_account] API returned {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"[lookup_account] External API failed: {e}")

        # Fallback: Local JSON
        result = _json_lookup_account(config.data_file, phone)
        logger.info(f"[lookup_account] local result={result}")
        await params.result_callback(result)

    async def get_next_appointment_by_phone(params: FunctionCallParams):
        phone = params.arguments.get("phone_number", "").strip()
        logger.info(f"[get_next_appointment_by_phone] phone={phone}")
        result = _json_get_next_appointment(config.data_file, phone)
        logger.info(f"[get_next_appointment_by_phone] result={result}")
        await params.result_callback(result)

    async def check_zip(params: FunctionCallParams):
        zipcode = params.arguments.get("zipcode", "").strip()
        logger.info(f"[check_zip] zipcode={zipcode}, company_id={config.company_id}")

        # Priority: External API
        api_base = os.getenv("COMPANY_LOOKUP_API_URL", "").replace("/api/get-company-by-number", "")
        if api_base:
            url = f"{api_base}/api/is-zip-serviced"
            try:
                async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
                    resp = await client.post(url, json={"company_id": config.company_id, "zip_code": zipcode})
                    if resp.status_code == 200:
                        result = resp.json()
                        logger.info(f"[check_zip] External API result: {result}")
                        await params.result_callback(result)
                        return
                    else:
                        logger.warning(f"[check_zip] API returned {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"[check_zip] External API failed: {e}")

        # Fallback: Local JSON
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

    # ── Pipeline stages (always include logger to capture history) ──
    from conversation_logger import ConversationLogger
    metadata = {
        "company_name": config.company_name,
        "to_number": to_number,
        "from_number": from_number,
    }
    conv_logger = ConversationLogger(
        on_log=log_callback,
        conversation_id=conversation_id,
        metadata=metadata
    )

    pipeline_stages = [
        transport.input(),
        stt,
        user_aggregator,
        llm,
        assistant_aggregator,
        conv_logger,
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
        
        # Personalized greeting if recognized
        customer_name = external_data.get("customer_name") or external_data.get("name")
        if customer_name:
            greeting = f"Hi {customer_name}! [warm] Welcome back to {config.company_name}. I have your account pulled up. How can I help you today?"
        else:
            greeting = (
                f"Oh, hi there! Thanks for calling {config.company_name}. [warm] "
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
    try:
        await runner.run(task)
    finally:
        # Save the conversation history to JSON before exiting
        filepath = conv_logger.save_to_json()
        logger.info(f"💾 Conversation stored in {filepath}")
