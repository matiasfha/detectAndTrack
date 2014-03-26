import numpy as np
from primesense import openni2
class Kinect:
    def __init__(self,oni=None):
        openni2.initialize()
        if oni is not None:
            self.dev = openni2.Device.open_file(oni)
        else:
            self.dev = openni2.Device.open_any()

    def get_depth_stream(self):
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.start()

    def get_color_stream(self):
        self.color_stream = self.dev.create_color_stream()
        self.color_stream.start()


    def depth_to_cv(self):
        frame       = self.depth_stream.read_frame()
        frame_data  = frame.get_buffer_as_uint16()
        depth_image = np.ndarray((frame.height, frame.width),dtype=np.uint16,buffer=frame_data)
        return depth_image

    def color_to_cv(self):
        frame       = self.color_stream.read_frame()
        frame_data  = frame.get_buffer_as_uint8()
        color_image = np.ndarray((frame.height, frame.width,3),dtype=np.uint8, buffer = frame_data)
        return color_image
    def unload(self):
        self.depth_stream.stop()
        self.color_stream.stop()
        openni2.unload()