import pygame
import pygame_gui

# Function to create a widget to edit integer value
def create_integer_widget(manager, rect, value_range):
    """
    Creates a widget to display and edit an integer value.
    
    :param manager: The pygame_gui UI manager.
    :param rect: The rectangle defining the widget's position and size.
    :param value_range: A list of integers that define the valid range of the value.
    :return: A tuple of the widget elements (UITextEntryLine, increment_button, decrement_button).
    """
    min_value, max_value = min(value_range), max(value_range)
    
    # Initial value set to the minimum value of the range
    initial_value = min_value

    # Create a UITextEntryLine to display and edit the integer value
    value_text = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(rect.x, rect.y, 60, 30),
        manager=manager,

        object_id="#integer_value_entry"
    )

    # Create buttons to increase and decrease the value
    increment_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(rect.x + 70, rect.y, 30, 30),
        text='+',
        manager=manager,
        object_id="#increment_button"
    )

    decrement_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(rect.x - 40, rect.y, 30, 30),
        text='-',
        manager=manager,
        object_id="#decrement_button"
    )

    def update_value(new_value):
        """
        Update the value inside the text entry, ensuring it stays within the valid range.
        """
        if min_value <= new_value <= max_value:
            value_text.set_text(str(new_value))
        else:
            # If out of range, reset to the nearest boundary
            value_text.set_text(str(min(max_value, max(min_value, new_value))))

    return value_text, increment_button, decrement_button, update_value
