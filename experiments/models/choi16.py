from keras.models import Model
from keras.layers import Dense
from keras.applications.music_tagger_crnn import MusicTaggerCRNN

model = MusicTaggerCRNN(weights=None, input_tensor=(96, 1366, 1), include_top=False)
