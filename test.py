from tensorflow.python.framework import tensor_util
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

for event in my_summary_iterator("/home/tomerweiss/MIMO/summary/selection_test_1e-5_2/events.out.tfevents.1608573265.floria.79308.0"):
    if event.step == 491:
        for value in event.summary.value:
            if value.tag.startswith('Rx_low'):
                print(value.tag, event.step, value.tensor.string_val)