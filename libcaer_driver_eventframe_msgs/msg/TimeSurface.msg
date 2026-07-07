# Dense event-camera "time surface" representation.
#
# For each (y, x, channel) cell, `data` stores the most recent normalized
# event timestamp that fell into that cell within the temporal window.
# Time is normalized to [0, 1] over the window.
#
# Channels are organized as: channel = 2 * bin_idx + polarity
#   bin_idx in [0, n_bins)   (temporal sub-window)
#   polarity in {0, 1}       (0 = OFF, 1 = ON)
# so channels = 2 * n_bins.

std_msgs/Header header

uint32 height            # camera Y resolution (rows)
uint32 width             # camera X resolution (cols)
uint32 channels          # = 2 * n_bins
uint32 n_bins            # number of temporal bins

# Sensor timestamps (microseconds since libcaer epoch) that bound the
# event batch used to build this representation.
uint64 window_start_us
uint64 window_end_us

# Row-major (H, W, C). Index: data[y * width * channels + x * channels + c]
float32[] data
