import numpy as np
import scipy

def pre_process_frame(frame, prev_frame):
    phi_frame = np.zeros((84, 84, 3))
    zoom_shape = np.array(phi_frame.shape, dtype=float) / np.array(frame.shape, dtype=float)
    zoom_frame = scipy.ndimage.zoom(frame, zoom_shape)
    r = zoom_frame[:,:,0].astype(float) # red channel
    g = zoom_frame[:,:,1].astype(float) # green channel
    b = zoom_frame[:,:,2].astype(float) # blue channel
    l = (0.2126*r + 0.7152*g + 0.0722*b) # luminescence 

    max_r = max(np.max(prev_frame[:,:,0]), np.max(r))
    max_g = max(np.max(prev_frame[:,:,1]), np.max(g))
    max_b = max(np.max(prev_frame[:,:,2]), np.max(b))

    normalized_color = np.zeros((zoom_frame.shape[0], zoom_frame.shape[1], 4))
#     normalized_color = np.zeros_like(zoom_frame).astype(float)
    normalized_color[:,:,0] = r / max_r
    normalized_color[:,:,1] = g / max_g
    normalized_color[:,:,2] = b / max_b
    normalized_color[:,:,3] = l

    return normalized_color