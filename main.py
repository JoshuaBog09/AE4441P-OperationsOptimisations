
class Square_Obsticale:
    def __init__(self, x_lower_left, y_lower_left, x_upper_right, y_upper_right):
        self.x_lower_left = x_lower_left
        self.y_lower_left = y_lower_left
        self.x_upper_right = x_upper_right
        self.y_upper_right = y_upper_right


list_of_obsticales = []

# Limits of the map, width is 600px, height is 400px

list_of_obsticales.append(Square_Obsticale(100, 100, 150, 150))
