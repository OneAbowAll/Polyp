import pygame
import imgui
from imgui.integrations.pygame import PygameRenderer

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((1280, 720), pygame.DOUBLEBUF | pygame.RESIZABLE | pygame.OPENGL)
pygame.display.set_caption("Pygame + ImGui Menu")

# Initialize OpenGL context (this is necessary for ImGui rendering)
pygame.display.get_surface()  # Ensure OpenGL context is active

# Initialize ImGui
imgui.create_context()
renderer = PygameRenderer()

# Set ImGui's display size manually to avoid assertion error
io = imgui.get_io()
io.display_size = screen.get_size()  # Set the display size to the window size

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        
        # Pass the event to the renderer to process input for ImGui
        renderer.process_event(event)

    # Update ImGui's display size in case the window was resized
    io.display_size = screen.get_size()

    # Start a new ImGui frame
    imgui.new_frame()

    # Create the menu
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):
            if imgui.menu_item("Open", "Ctrl+O", False, True):
                print("Open clicked")
            if imgui.menu_item("Save", "Ctrl+S", False, True):
                print("Save clicked")
            if imgui.menu_item("Exit", "Alt+F4", False, True):
                running = False
            imgui.end_menu()

        if imgui.begin_menu("Edit", True):
            if imgui.menu_item("Undo", "Ctrl+Z", False, True):
                print("Undo clicked")
            if imgui.menu_item("Redo", "Ctrl+Y", False, True):
                print("Redo clicked")
            imgui.end_menu()

        imgui.end_main_menu_bar()

    # Render the frame
    imgui.render()
    renderer.render(imgui.get_draw_data())

    # Update the screen
    pygame.display.flip()

# Cleanup
renderer.shutdown()
pygame.quit()
