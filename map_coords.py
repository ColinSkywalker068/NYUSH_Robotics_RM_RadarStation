"""Coordinate helpers for battle and NYUSH test-map modes."""

BATTLEFIELD_WIDTH_CM = 2800
BATTLEFIELD_HEIGHT_CM = 1500

# NYUSH test field: 6.79m x 3.82m.
TESTMAP_REAL_WIDTH_M = 6.79
TESTMAP_REAL_HEIGHT_M = 3.82
TESTMAP_REAL_WIDTH_CM = int(round(TESTMAP_REAL_WIDTH_M * 100))
TESTMAP_REAL_HEIGHT_CM = int(round(TESTMAP_REAL_HEIGHT_M * 100))


def get_reference_field_size(map_mode):
    if map_mode == "testmap":
        return TESTMAP_REAL_WIDTH_CM, TESTMAP_REAL_HEIGHT_CM
    return BATTLEFIELD_WIDTH_CM, BATTLEFIELD_HEIGHT_CM


def _clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def map_to_display_coords(map_x, map_y, map_mode, state, display_w, display_h):
    if map_mode == "testmap":
        return float(map_x), float(map_y)
    if state == "R":
        return float(display_w - map_y), float(map_x)
    return float(map_y), float(display_h - map_x)


def testmap_calibration_to_display_coords(map_x, map_y, calibration_w):
    # `my_map(m).jpg` is `my_map.jpg` rotated 90 degrees counterclockwise.
    return float(map_y), float((calibration_w - 1) - map_x)


def display_to_ref_coords(display_x, display_y, map_mode, display_w, display_h):
    ref_w, ref_h = get_reference_field_size(map_mode)
    if map_mode == "testmap":
        ref_x = round(display_x * ref_w / float(display_w))
        ref_y = round((display_h - display_y) * ref_h / float(display_h))
    else:
        ref_x = round(display_x)
        ref_y = round(display_h - display_y)
    ref_x = _clamp(int(ref_x), 0, ref_w)
    ref_y = _clamp(int(ref_y), 0, ref_h)
    return ref_x, ref_y


def map_to_display_and_ref_coords(map_x, map_y, map_mode, state, display_w, display_h):
    display_x, display_y = map_to_display_coords(map_x, map_y, map_mode, state, display_w, display_h)
    ref_x, ref_y = display_to_ref_coords(display_x, display_y, map_mode, display_w, display_h)
    return (display_x, display_y), (ref_x, ref_y)
