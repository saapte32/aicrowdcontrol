import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
#from pool import run
#from Bothavg_old import run,predict
import numpy as np
import cv2
import sys
#import cv2

cap = cv2.VideoCapture('ind1.mp4')
f = []
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == False:
        break
    f.append(frame)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
# print(len(f))
cap.release()
cv2.destroyAllWindows()


# import pdb; pdb.set_trace()
# print(lines)
class imgpre(beam.DoFn):
    def preprocessy(frame):
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(frame, kernel, iterations=1)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(erosion, -1, sharpen_kernel)
        return sharpen

    def process(self, lines):
        for frame in lines:
            yield preprocessy(frame)


pipeline_options = PipelineOptions(sys.argv)
with beam.Pipeline(options=pipeline_options) as pipeline:
    lines = (
        pipeline
        | 'To pipeline' >> beam.Create(f))
# | 'To detect' >> beam.ParDo(Detect())
# | beam.Map(print))

# p = beam.Pipeline(options=pipeline_options)
# lines = p | 'pipe' >> beam.Create(f)
# lines | 'klklkl' >> beam.Map(print)
# violations = lines | 'par' >> beam.ParDo(Detect())
# violations | 'dds' >> beam.Map(print)

# print('+++++++++', lines)

preimage = lines | 'Preprocessing' >> beam.ParDo(imgpre())

print(preimage)
# # x = violations | beam.Map(print)
# # print(x)


# import apache_beam as beam

# class SplitWords(beam.DoFn):
#   def __init__(self, delimiter=','):
#     self.delimiter = delimiter

#   def process(self, text):
#     for word in text.split(self.delimiter):
#       yield word

# with beam.Pipeline() as pipeline:
#   plants = (
#       pipeline
#       | 'Gardening plants' >> beam.Create([
#           '🍓Strawberry,🥕Carrot,🍆Eggplant',
#           '🍅Tomato,🥔Potato',
#       ])
#       | 'Split words' >> beam.ParDo(SplitWords(','))
#       | beam.Map(print))

