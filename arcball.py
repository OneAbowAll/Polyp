import glm


class ArcballCamera:
    def __init__(self, screen_width, screen_height) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.center = glm.vec3(0)
        self.distance = 2.0
        self.radius = 1.0

        self.last_point = glm.vec3(0)
        self.rotation = glm.mat4(1.0)

        self.is_dragging = False

    def mouse_pressed(self, screen_x, screen_y):
        self.last_point = self.get_arcball_vector(screen_x, screen_y)
        self.is_dragging = True

    def mouse_release(self):
        self.is_dragging = False
    
    def mouse_move(self, screen_x, screen_y):
        if(not self.is_dragging):
            return
        
        P0 = self.last_point
        P1 = self.get_arcball_vector(screen_x, screen_y)

        angle = glm.acos(glm.min(1.0, glm.dot(P0, P1)))

        rotation_axis = glm.cross(P0, P1)
        if glm.length(rotation_axis) < 1e-6:
            return
        
        delta_rot = glm.rotate(glm.mat4(1.0), angle, rotation_axis)
        self.rotation = delta_rot * self.rotation
        self.last_point = P1

    def get_arcball_vector(self, screen_x, screen_y):
        """
        Get a normalized vector from the center of the virtual ball O to a
        point P on the virtual ball surface, such that P is aligned on
        screen's (X,Y) coordinates.  If (X,Y) is too far away from the
        sphere, return the nearest point on the virtual ball surface.
        """

        P = glm.vec3(
            1.0 * screen_x /self.screen_width*2 - 1.0,
            1.0 * screen_y/self.screen_height*2 - 1.0,
            0
        )

        P.y = - P.y
        
        OP_squared = P.x * P.x + P.y * P.y
        if OP_squared <= 1.0 :
            P.z = glm.sqrt(1 - OP_squared)
        else:
            P = glm.normalize(P) #Trovami il punto piu' vicino

        return P
    
    def view_matrix(self):
        return self.rotation