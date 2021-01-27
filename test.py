from tensorflow.python.framework import tensor_util
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

# for event in my_summary_iterator("/home/tomerweiss/MIMO/summary/7mse/learned_1e-5_1e-4_gaussian_mvb_no_trace/events.out.tfevents.1611147471.floria.24173.0"):
for event in my_summary_iterator("/home/tomerweiss/MIMO/summary/7mse/learned_1e-5_1e-4_gaussian/events.out.tfevents.1610642757.floria.62839.0"):
    if event.step %100==0:
        for value in event.summary.value:
            if value.tag.startswith('Rx_low'):
                print(value.tag, event.step, value.tensor.string_val)