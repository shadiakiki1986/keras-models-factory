# stride the data
from skimage.util.shape import view_as_windows # pip install scikit-image
def _load_data_strides(A, n_prev):
    out = view_as_windows(A,window_shape=(n_prev,A.shape[1]),step=1)
    out = out.reshape((out.shape[0],out.shape[2],out.shape[3])) # for some reason need to drop extra dim=1
    return out


#--------------------------------
# Read from tensorboard events file: epochs trained, latest loss, latest val_loss
#
# Parameters: fnd3: path to folder
#
# Example: 
# load_tensorboard_events('/mnt/ec2vol/g2-ml-data/t2e5/tb/p3g7-130-100_lb-4_data-2000-2015/')
# load_tensorboard_events('/mnt/ec2vol/g2-ml-data/t2e5/tb/p3g6-130-100_lb-4/')
from os import listdir
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
def load_tensorboard_latest_data(fnd3:str):
  # get most recent events file in folder
  xxx=listdir(fnd3)
  xxx.sort(reverse=True)
  for fnf0 in xxx:
    print('tb check '+fnf0)
    fnf = fnd3+'/'+fnf0

    # read tensorboard events file
    # The format is Google Protocol Buffer
    # https://developers.google.com/protocol-buffers/docs/pythontutorial
    #
    # The Tensorflow Protocol Files:
    #   Repository: /home/shadi/.local/share/virtualenvs/G2ML/lib/python3.5/site-packages/tensorflow
    #   Files:
    #     - core/util/event_pb2.py
    #     - core/framework/summary_pb2.py
    loader = EventFileLoader(fnf)
    inter = [
        {'step':event.step, 'tag': event.summary.value[0].tag, 'value': event.summary.value[0].simple_value}
        for event in loader.Load()
        if len(event.summary.value)>0
      ][-2:]

    # sanity checks
    if len(inter)==0:
      print("No data in file")
      continue

    if inter[0]['step']!=inter[1]['step']: raise Exception("mixed steps")
    ix_loss = 0 if inter[0]['tag']=='loss' else 1
    if inter[ix_loss]['tag']!='loss': raise Exception("loss not found")
    if inter[1-ix_loss]['tag']!='val_loss': raise Exception("val_loss not found")

    # reformat
    out = {'step': inter[0]['step'], 'loss': inter[ix_loss]['value'], 'val_loss': inter[1-ix_loss]['value']}
    return out

  # if didn't return anything yet
  return None

