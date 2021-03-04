import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
#from pool import run
from detect.Bothavg_old import run,predict

import numpy as np
import cv2
import sys

cap = cv2.VideoCapture('socialchina.mp4')
f = []
while(True):
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
print(len(f))
cap.release()
cv2.destroyAllWindows()


# import pdb; pdb.set_trace()
# print(lines)  
class Detect(beam.DoFn):
	def __init__(self):
		self.lines = lines



	def process(self,lines):
		for frame in lines:
			yield run(frame, 3.6, 1.1, 10)




class MaskDetect(beam.DoFn):
	def __init__(self,lines):
		self.element = lines



	def process(self, element):
		for frame in element:
			yield predict(frame)

pipeline_options = PipelineOptions(sys.argv)
with beam.Pipeline(options=pipeline_options) as pipeline:
	lines = (
		pipeline
		| beam.Create(f))



violations = lines | beam.ParDo(Detect()) | beam.Map(print)
maskviolations = lines | beam.ParDo(MaskDetect()) |beam.Map(print)

#import pdb; pdb.set_trace()
	
	

