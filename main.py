
import glm
import imgui
from imgui.integrations.pygame import PygameRenderer

import pygame

import log

def main():
    glm.silence(4)
    
    #Window context variables
    W, H = 1200, 800
    SCREEN = None
    CLOCK = None

    #Setup PyGame
    pygame.init()
    pygame.display.set_caption("Polyp Detector")
    SCREEN = pygame.display.set_mode((W, H), pygame.OPENGL|pygame.DOUBLEBUF)
    CLOCK = pygame.time.Clock()

    # Initialize ImGui
    imgui.create_context()
    IMGUI_RENDERER = PygameRenderer()
    imgui.get_io().display_size = (W, H)

    running = True
    while running:
        #Handle PyGames events ------------------------
        for event in pygame.event.get():
            IMGUI_RENDERER.process_event(event)

            if event.type == pygame.QUIT:
                running = False;

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    running = False;
        #----------------------------------------------
        
        #Imgui ----------------------------------------
        imgui.new_frame()
        if imgui.begin_main_menu_bar().opened:
            if imgui.begin_menu('Actions', True).opened:
                imgui.text("Test")
                imgui.end_menu()

        imgui.end_main_menu_bar()
        #----------------------------------------------

        #End of frame----------------------------------
        imgui.render()
        IMGUI_RENDERER.render(imgui.get_draw_data())

        pygame.display.flip()
        CLOCK.tick(60)
        #----------------------------------------------

    #Stuff to do before close
    log.print_debug(f"Test {W}:{H}")
    log.print_warning(f"Test {W}:{H}")
    log.print_info(f"Test {W}:{H}")
    return 0;

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()
