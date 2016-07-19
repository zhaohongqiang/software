import math

def calc_field_of_view(cam, dimension):
    """
    Calculates the field of view angle in radians of the camera given a
    dimension in mm.
    """
    return 2 * math.atan((dimension / 2) / cam['focal_length_mm'])

def calc_forward_angles(cam, coord):
    """
    Calculates the heading and pitch of the given (x, y) coordinate relative to
    given camera in radians.

    It is "as if" the sensor was smaller. See
    https://en.wikipedia.org/wiki/Angle_of_view for a nice picture.
    """
    pixel_size = cam['sensor_size_wh_mm'][0] / cam['width']
    diff_x = pixel_size * (coord[0] - cam['width'] / 2)
    diff_y = pixel_size * (cam['height'] / 2 - coord[1])
    heading = calc_field_of_view(cam, 2 * diff_x) / 2
    pitch = calc_field_of_view(cam, 2 * diff_y) / 2
    return (heading, pitch)

def calc_downward_angles(cam, coord):
    forward_heading, forward_pitch = calc_forward_angles(cam, coord)
    center_x, center_y = cam['width'] / 2, cam['height'] / 2
    heading = None
    if center_y == coord[1]:
        if coord[0] >= center_x:
            heading = 0
        else:
            heading = math.pi
    else:
        heading = math.atan((coord[0] - center_x) / (center_y - coord[1]))
    if coord[0] < center_x:
        heading += math.pi
    pitch = forward_pitch - math.pi / 2
    return (heading, pitch)
