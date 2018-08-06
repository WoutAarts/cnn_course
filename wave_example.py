import matplotlib.pyplot as plt
import numpy as np
import wave
import pandas as pd
import sys

from scipy.io.wavfile import write

spf = wave.open('/home/wout/Codebase/machine_learning_examples/cnn_class/helloworld.wav', 'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
plt.close()

print(signal)
# print('valuecounts of this array is: ', pd.Series(signal).plot('hist'))
print('numpy signal shape and type', signal.shape, type(signal))


plt.plot(signal)
plt.title('Hello world without echo')
plt.show()

delta = np.array([1.,0.,0.])

noecho = np.convolve(signal,delta)
print('no echo signal shape: ', noecho.shape)

assert(np.abs(noecho[:len(signal)] - signal).sum()<0.00000001)

noecho = noecho.astype(np.int16)

write('noecho.wav', 16000, noecho)

filt = np.zeros(16000)
filt[0] = 1
filt[4000] = 0.6
filt[8000] = 0.3
filt[12000] = 0.2
filt[15999] = 0.1

out = np.convolve(signal, filt)

out = out.astype(np.int16)
write('out.wav', 16000, out)

plt.close()
plt.plot(out)
plt.title('Hello world without echo')
plt.show()
