import random
from typing import Tuple


class Region:
    def __init__(self, id:int, padding:int,  bbox: Tuple[int, int, int, int]):
        self.id: int = id
        self.bbox: Tuple[int, int, int, int] = bbox
        self.padding = padding

        self.paddedWidth = self.bbox[3] - self.bbox[1]
        self.paddedHeight = self.bbox[2] - self.bbox[0]

        self.width = self.paddedWidth - 2*padding
        self.height = self.paddedHeight - 2*padding

    def randomPointInRegion(self):
        return random.randint(0, self.bbox[2] - self.bbox[0] - 1), random.randint(0, self.bbox[3] - self.bbox[1] - 1)

    def __str__(self):
        return f"Region(id: {self.id}, bbox: {self.bbox})"

