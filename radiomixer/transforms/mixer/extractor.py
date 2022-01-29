from radiomixer.io.signal import Signal
from radiomixer.transforms.transform import Transform, TransformType

class ExtractSegment(Transform):
    def __init__(self):
        super().__init__(TransformType.EXTRACTSEGMENT)

    def process(self, signal:Signal):
        signal.name = self._prepend_transform_name(signal.name)
        start, end = signal.parameters['segment'][0], signal.parameters['segment'][1]
        if signal.data.shape[0]==1:
            audio = signal.data.squeeze(0)
        signal.data = audio[start:end].view(1, -1)

        
        return signal
    