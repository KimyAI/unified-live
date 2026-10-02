"""Virtual device abstractions. Drivers are separately installed."""


class VirtualCamera:
    def __init__(self, width, height, fps):
        import pyvirtualcam

        self._camera = pyvirtualcam.Camera(
            width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR
        )
        self.name = self._camera.device

    def send(self, frame):
        self._camera.send(frame)

    def close(self):
        self._camera.close()


def open_audio_output(device, sample_rate, blocksize, callback):
    import sounddevice as sd

    if device is None:
        raise ValueError("Choose an explicit audio output (e.g. CABLE Input) before routing audio")
    sd.check_output_settings(device=device, channels=1, dtype="float32", samplerate=sample_rate)
    return sd.OutputStream(
        device=device, samplerate=sample_rate, channels=1, dtype="float32",
        blocksize=blocksize, latency="low", callback=callback,
    )
