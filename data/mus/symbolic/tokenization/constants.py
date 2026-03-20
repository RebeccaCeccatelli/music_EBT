from enum import Enum

class DatasetType(str, Enum):
    CUSTOM = "custom"
    LAKH = "lakh"
    MAESTRO = "maestro"
    GIGA_MIDI = "gigaMIDI"

    @classmethod
    def list(cls):
        return [c.value for c in cls]