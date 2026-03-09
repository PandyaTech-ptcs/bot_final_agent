from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import TranscriptionFrame, TextFrame, LLMFullResponseEndFrame


class ConversationLogger(FrameProcessor):
    """
    Logs user transcription and bot responses.
    Optionally calls on_log(text) so callers can broadcast to a websocket monitor.
    """

    def __init__(self, on_log=None):
        super().__init__()
        self._buffer = ""
        self._on_log = on_log

    async def _log(self, text: str):
        print(text)
        if self._on_log:
            await self._on_log(text)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self._log(f"User: {frame.text}")

        elif isinstance(frame, TextFrame):
            self._buffer += frame.text

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._buffer:
                await self._log(f"Bot: {self._buffer}")
                self._buffer = ""

        await self.push_frame(frame, direction)
