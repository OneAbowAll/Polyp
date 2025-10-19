import glm
class Box3:
    """ A class representing a 3D bounding box. """

    def __init__(self, *args):
        """ 
        Overloaded constructors:
        - Box3() -> Creates an empty box.
        - Box3(s: float) -> Creates a box of size `s` centered at (0, 0, 0).
        - Box3(min: glm.vec3, max: glm.vec3) -> Creates a box with given min/max points.
        """
        if len(args) == 1 and isinstance(args[0], (int, float)):  # Box3(float s)
            s = args[0]
            self.min = glm.vec3(-s / 2.0)
            self.max = glm.vec3(s / 2.0)
        elif len(args) == 2:  # Box3(glm.vec3 min, glm.vec3 max)
            self.min = glm.vec3(args[0])
            self.max = glm.vec3(args[1])
        else:  # Default constructor (empty box)
            self.min = glm.vec3(1.0)
            self.max = glm.vec3(-1.0)

    def add(self, p):
        """ Expands the bounding box to include a point or another Box3. """
        if isinstance(p, Box3):  # If `p` is another Box3, add both corners
            self.add(p.min)
            self.add(p.max)
        else:  # Otherwise, `p` is a vector (point)
            p = glm.vec3(p)
            if self.is_empty():
                self.min = self.max = p
            else:
                self.min = glm.vec3(min(self.min.x, p.x), min(self.min.y, p.y), min(self.min.z, p.z))
                self.max = glm.vec3(max(self.max.x, p.x), max(self.max.y, p.y), max(self.max.z, p.z))

    def is_empty(self):
        """ Checks if the bounding box is empty. """
        return self.min.x > self.max.x

    def diagonal(self):
        """ Returns the length of the diagonal of the bounding box. """
        return glm.length(self.max - self.min)

    def center(self):
        """ Returns the center of the bounding box. """
        return (self.min + self.max) * 0.5

    def p(self, i):
        """ Returns the i-th corner of the bounding box (0 to 7). """
        return glm.vec3(
            self.min.x if (i % 2 == 0) else self.max.x,
            self.min.y if ((i // 2) % 2 == 0) else self.max.y,
            self.min.z if ((i // 4) == 0) else self.max.z
        )
    def dim_z(self):
        return self.max.z - self.min.z