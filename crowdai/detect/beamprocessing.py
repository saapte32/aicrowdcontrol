import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
options = PipelineOptions()
p = beam.Pipeline(options=options)

beam.io.fileio.ReadableFile




