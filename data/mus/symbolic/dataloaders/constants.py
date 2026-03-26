from enum import Enum

class DatasetType(str, Enum):
    CUSTOM = "custom"
    GIGA_MIDI = "gigamidi" 
    
    @property
    def folder_name(self):
        mapping = {
            DatasetType.GIGA_MIDI: "giga-midi",
        }
        return mapping.get(self, self.value)

    @classmethod
    def list(cls):
        return [c.value for c in cls]