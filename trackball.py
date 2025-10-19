import glm
from OpenGL.GL import *

class Trackball:
    def __init__(self):
        self.is_trackball_dragged = False
        self.changed = False
        self.p0 = glm.vec3(0.0)
        self.p1 = glm.vec3(0.0)
        self.rotation_matrix = glm.mat4(1.0)
        self.scaling_matrix = glm.mat4(1.0)
        self.translation_matrix = glm.mat4(1.0)
        self.old_tb_matrix = glm.mat4(1.0)
        self.scaling_factor = 1.0
        self.center = glm.vec3(0.0)
        self.radius = 2.0
        self.reset()

    def reset(self):
        self.scaling_factor = 1.0
        self.scaling_matrix = glm.mat4(1.0)
        self.rotation_matrix = glm.mat4(1.0)
        self.translation_matrix = glm.mat4(1.0)

    def set_center_radius(self, c, r):
        self.old_tb_matrix = self.matrix()
        self.reset()
        self.center = glm.vec3(c)
        self.radius = r
        self.translation_matrix = glm.translate(glm.mat4(1.0), self.center)

    def reset_center(self,c): #FIX THIS FOR NON 0,0,0 CENTER
        self.old_tb_matrix =  glm.translate(glm.mat4(1.0), -glm.vec3(c))*self.matrix()
        self.reset()


    def viewport_to_ray(self, proj, pX, pY):
        vp = glGetIntegerv(GL_VIEWPORT)
        proj_inv = glm.inverse(proj)
        p0 = glm.vec4(
            -1.0 + (float(pX) / vp[2]) * 2.0,
            -1.0 + ((vp[3] - float(pY)) / vp[3]) * 2.0,
            -1.0, 1.0
        )
        p1 = glm.vec4(p0.x, p0.y, 1.0, 1.0)
        p0 = proj_inv * p0
        p0 /= p0.w
        p1 = proj_inv * p1
        p1 /= p1.w
        d = glm.normalize(p1 - p0)
        return p0, d

    def cursor_sphere_intersection(self, proj, view, xpos, ypos):
        view_frame = glm.inverse(view)
        o, d = self.viewport_to_ray(proj, xpos, ypos)
        o = view_frame * o
        d = view_frame * d

        # Simulating ray-sphere intersection
        hit, int_point = self.intersection_ray_sphere(o, d)
        if hit:
            int_point -= self.center
        return hit, int_point

    def intersection_ray_sphere(self, o, d):
        # Ray-sphere intersection logic
        oc = glm.vec3(o) - self.center
        b = 2.0 * glm.dot(glm.vec3(d), oc)
        c = glm.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * c
        if discriminant < 0:
            return False, glm.vec3(0.0)
        t1 = (-b - glm.sqrt(discriminant)) / 2.0
        t2 = (-b + glm.sqrt(discriminant)) / 2.0
        if t1 > 0:
            return True, glm.vec3(o) + t1 * glm.vec3(d)
        else:
            if t2 > 0:
                return True, glm.vec3(o) + t2 * glm.vec3(d)
        return False, glm.vec3(0.0)

    def mouse_move(self, proj, view, xpos, ypos):
        if not self.is_trackball_dragged:
            return

        hit, self.p1 = self.cursor_sphere_intersection(proj, view, xpos, ypos)
        if hit:
            rotation_vector = glm.cross(glm.normalize(self.p0), glm.normalize(self.p1))

            if glm.length(rotation_vector) > 0.01:
                alpha = glm.asin(glm.length(rotation_vector))
                delta_rot = glm.rotate(glm.mat4(1.0), alpha, rotation_vector)
                self.rotation_matrix = delta_rot * self.rotation_matrix
                self.p0 = self.p1

    def mouse_press(self, proj, view, xpos, ypos):
        hit, int_point = self.cursor_sphere_intersection(proj, view, xpos, ypos)
        if hit:
            self.p0 = int_point
            self.is_trackball_dragged = True

    def mouse_release(self):
        self.is_trackball_dragged = False

    def mouse_scroll(self, xoffset, yoffset):
        self.changed = True
        self.scaling_factor *= 1.1 if yoffset > 0 else 0.97
        self.scaling_matrix = glm.scale(glm.mat4(1.0), glm.vec3(self.scaling_factor))

    def matrix(self):
        return (
            self.translation_matrix
            * self.scaling_matrix
            * self.rotation_matrix
            * glm.inverse(self.translation_matrix)
            * self.old_tb_matrix
        )

    def is_moving(self):
        return self.is_trackball_dragged

    def is_changed(self):
        if self.changed or self.is_trackball_dragged:
            self.changed = False
            return True
        return False
