import json
import os
from datetime import datetime
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import TranscriptionFrame, TextFrame, LLMFullResponseEndFrame


class ConversationLogger(FrameProcessor):
    """
    Logs user transcription and bot responses.
    Optionally calls on_log(text) so callers can broadcast to a websocket monitor.
    Now stores conversation history and saves to JSON.
    """

    def __init__(self, on_log=None, conversation_id=None, metadata=None):
        super().__init__()
        self._buffer = ""
        self._on_log = on_log
        self._conversation_id = conversation_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self._history = []
        self._metadata = metadata or {}

    async def _log(self, role: str, text: str):
        entry = {
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
        self._history.append(entry)
        
        log_text = f"{role.capitalize()}: {text}"
        print(log_text)
        if self._on_log:
            await self._on_log(log_text)

    def save_to_json(self, directory="conversations"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filename = f"conversation_{self._conversation_id}.json"
        filepath = os.path.join(directory, filename)
        
        data = {
            "conversation_id": self._conversation_id,
            "metadata": self._metadata,
            "start_time": self._history[0]["timestamp"] if self._history else datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "history": self._history
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        return filepath

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._log("user", frame.text)

        elif isinstance(frame, TextFrame):
            self._buffer += frame.text

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._buffer:
                await self._log("bot", self._buffer)
                self._buffer = ""

        await self.push_frame(frame, direction)
