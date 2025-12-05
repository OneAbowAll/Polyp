import random
from typing import Tuple


class Region:
    def __init__(self, id:int,  bbox: Tuple[int, int, int, int]):
        self.id: int = id
        self.bbox: Tuple[int, int, int, int] = bbox

        self.width = self.bbox[3] - self.bbox[1]
        self.height = self.bbox[2] - self.bbox[0]

    def randomPointInRegion(self):
        return random.randint(0, self.bbox[2] - self.bbox[0] - 1), random.randint(0, self.bbox[3] - self.bbox[1] - 1)

    def __str__(self):
        return f"Region(id: {self.id}, bbox: {self.bbox})"

