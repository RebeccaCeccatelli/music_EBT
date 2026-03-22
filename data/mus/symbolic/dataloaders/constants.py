from enum import Enum

class DatasetType(str, Enum):
    CUSTOM = "custom"
    LAKH = "lakh"
    MAESTRO = "maestro"
    GIGA_MIDI = "gigamidi" 
    
    @property
    def folder_name(self):
        mapping = {
            DatasetType.LAKH: "lakh-midi-clean",
            DatasetType.GIGA_MIDI: "giga-midi",
        }
        return mapping.get(self, self.value)

    @classmethod
    def list(cls):
        return [c.value for c in cls]