import numpy as np
import numpy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.signal.windows import hamming

## Read in the audio file: https://ufile.io/0d4f4
# Convert both raw samples and sample rate to floats, then convert stereo to mono.
(fs, y) = wavfile.read("C://Users//drago//Desktop//Projet//Expérience guitare//A5//Son 01.wav")
y = y.astype(float) / 2**16 # normalize to -1 to 1
y = y[:,0] + y[:,1] # stereo to mono
fs = float(fs)


def sliding(x, Nwin, Noverlap=0, f=lambda x: x):
  """Apply a function over overlapping sliding windows with overlap.

  Given an iterator with N elements (a list, a Numpy vector, a range object,
  etc.), subdivide it into Nwin-length chunks, potentially with Noverlap samples
  overlapping between chunks. Optionally apply a function to each such chunk.

  Any chunks at the end of x whose length would be < Nwin are silently ignored.

  Parameters
  ----------
  x : array_like
      Iterator (list, vector, range object, etc.) to operate on.
  Nwin : int
      Length of each chunk (sliding window).
  Noverlap : int, optional
      Amount of overlap between chunks. Noverlap must be < Nwin. 0 means no
      overlap between chunks. Positive Noverlap < Nwin means the last Noverlap
      samples of a chunk will be the first Noverlap samples of the next chunk.
      Negative Noverlap means |Noverlap| samples of the input will be skipped
      between each successive chunk.
  f : function, optional
      A function to apply on each chunk. The default is the identity function
      and will just return the chunks.

  Returns
  -------
  l : list
      A list of chunks, with the function f applied.
  """
  hop = Nwin - Noverlap
  return [f(x[i : i + Nwin])
          for i in range(0, len(x), hop)
          if i + Nwin <= len(x)]


def hps(x, numProd, Nfft=None, fs=1):
  """Harmonic product spectrum of a vector.

  This algorithm can be used for fundamental frequency detection. It evaluates
  the magnitude of the FFT (fast Fourier transform) of the input signal, keeping
  only the positive frequencies. It then element-wise-multiplies this spectrum
  by the same spectrum downsampled by 2, then 3, ..., finally ending after
  numProd downsample-multiply steps.

  Here, "downsampling a vector by N" means keeping only every N samples:
  downsample(v, N) = v[::N].

  Of course, at each step, a vector of data is multiplied by a vector *smaller*
  than it: the algorithm specifies that the extra elements at the end of the
  longer vector be ignored. This implies that the output will be ceil(len(x) /
  numProd) long, so at each step, we only consider this many elements.

  References
  ----------
  See Gareth Middleton, Pitch Detection Algorithms (2003) at
  http://cnx.org/contents/i5AAkZCP@2/Pitch-Detection-Algorithms#idp2614240
  (accessed September 2016).

  Parameters
  ----------
  x : array_like
      Time-samples of data
  numProd : int
      Number of products to evaluate the harmonic product spectrum over.
  Nfft : int, optional
      The length of the FFT. Almost always this is greater than the length of x,
      with x being zero-padded before the FFT. This is helpful for two reasons:
      more zero-padding means more interpolation in the spectrum (a smoother
      spectrum). Also, FFT lengths with low prime factors (i.e., products of 2,
      3, 5, 7) are usually (much) faster than those with high prime factors. The
      default is Nfft = len(x). By way of example, if len(x) = 4001, this
      default might take much more time to run than Nfft = 4096, since 4001 is
      prime while 4096 is a power of 2.
  fs : float, optional
      The sample rate of x, in samples per second (Hz). Used only to format the
      returned vector of frequencies.

  Returns
  -------
  y : array
      Spectrum vector with ceil(Nfft / (2 * numProd)) elements.
  f : array
      Vector of frequencies corresponding to the spectrum in y. Runs from 0 to
      roughly (fs / (2 * numProd)) Hz.
  """
  Nfft = Nfft or x.size
  # Evaluate FFT. f is the frequencies corresponding to the spectrum xf
  f = np.arange(Nfft) / Nfft
  xf = fft.fft(x, Nfft)
  # Keep magnitude of spectrum at positive frequencies
  xf = np.abs(xf[f < 0.5])
  f = f[f < 0.5]
  N = f.size

  # Downsample-multiply
  smallestLength = int(np.ceil(N / numProd))
  y = xf[:smallestLength].copy()
  for i in range(2, numProd + 1):
    y *= xf[::i][:smallestLength]
  f = f[:smallestLength] * fs
  return (y, f)


# Parameters for HPS and the sliding window over data to apply it over
prod = 5
winlen = 8*1024
overlap = winlen // 2
Nfft = int(4 * 2**np.ceil(np.log2(winlen)))

# Run HPS on winlen-long chunks of the data.
hpsArr = np.array(sliding(y, winlen, Noverlap=overlap,
                          f=lambda x: hps(x * hamming(len(x), False),
                                          prod, Nfft)[0]))
# Extract the frequency and time vectors that this array of spectra corresponds to
hpsF = hps(y[:winlen], prod, Nfft)[1] * fs
hpsT = np.arange(hpsArr.shape[0]) / fs * (winlen - overlap)


# Display
db20 = lambda x: np.log10(np.abs(x)) * 20

def extents(f):
  """Convert evenly-spaced vector of sample locations to extents for `imshow`.

  If you want to use Matplotlib's `imshow` to visualize an array and need to
  specify the location of each pixel, you need to adjust the `extent` kwarg to
  `imshow`.

  Example:
  # imshow(data, extent=extents(x) + extents(y))

  See https://gist.github.com/fasiha/eff0763ca25777ec849ffead370dc907 for
  additional details.
  """
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(['dark_background','ggplot'])

plt.close()
plt.imshow(db20(hpsArr), aspect='auto', interpolation='nearest',
           extent=extents(hpsF) + extents(hpsT), cmap='viridis')
plt.xlabel('frequency (Hz)')
plt.ylabel('time (seconds)')
plt.title('Harmonic product spectrum: # products={}, window length={}'.format(prod, winlen))
plt.tight_layout()

plt.xlim([0, 500])
plt.savefig('hps-0-500.png')

plt.xlim([200,240])
plt.savefig('hps-200-240.png')

plt.xlim([0,1000])
plt.savefig('hps-0-1000.png')

# No chunking
def nextpow2(n):
  return int(2 ** np.ceil(np.log2(n)))

prod = 5
[hpsS, hpsF] = hps(y * hamming(len(y)), prod, nextpow2(y.size), fs)
plt.close()
plt.plot(hpsF, db20(hpsS))
plt.ylim([-100, 300])
plt.xlabel('frequency (Hz)')
plt.ylabel('spectrum, dB')
plt.title('Harmonic product spectrum: # products={}'.format(prod))
plt.tight_layout()

plt.gca().set_xticks(np.arange(0, 2000, 55))
plt.xlim([0, 1000])
plt.savefig('hps-alldata-0-1000.png')
plt.gca().set_xticks(np.arange(0, 2000, 110))
plt.xlim([0, 2000])
plt.savefig('hps-alldata-0-2000.png')

print(f"LA SUPER MEGA VALEUR IMPORTANTE EST : freq = {np.round(hpsF[np.argmax(hpsS)],1)}Hz , valeurs = {np.round(max(db20(hpsS)),1)} dB")


# Welch?
from scipy.signal import welch
(fWelch, pWelch) = welch(y, fs=fs, window='hamming', nperseg=8192, noverlap=4096)

plt.close()
plt.plot(fWelch, db20(pWelch))
plt.xlabel('frequency (Hz)')
plt.ylabel('spectrum estimate')
plt.title('Welch spectral estimate, 8192-Hamming window & 50% overlap')

plt.ylim([-200, -50])
plt.xlim([0, 2000])
plt.tight_layout()
plt.gca().set_xticks(np.arange(0, 2000, 110))
plt.savefig('welch.png')


# Blackman-Tukey spectral esimator
from scipy.signal import correlate

def acorrBiased(y):
  """Obtain the biased autocorrelation and its lags
  """
  r = correlate(y, y) / len(y)
  l = np.arange(-(len(y)-1), len(y))
  return r,l

# This is a port of the code accompanying Stoica & Moses' "Spectral Analysis of
# Signals" (Pearson, 2005): http://www2.ece.ohio-state.edu/~randy/SAtext/
def blackmanTukey(y, w, Nfft, fs=1):
  """Evaluate the Blackman-Tukey spectral estimator

  Parameters
  ----------
  y : array_like
      Data
  w : array_like
      Window, of length <= y's
  Nfft : int
      Desired length of the returned power spectral density estimate. Specifies
      the FFT-length.
  fs : number, optional
      Sample rate of y, in samples per second. Used only to scale the returned
      vector of frequencies.

  Returns
  -------
  phi : array
      Power spectral density estimate. Contains ceil(Nfft/2) samples.
  f : array
      Vector of frequencies corresponding to phi.

  References
  ----------
  P. Stoica and R. Moses, *Spectral Analysis of Signals* (Pearson, 2005),
  section 2.5.1. See http://www2.ece.ohio-state.edu/~randy/SAtext/ for original
  Matlab code. See http://user.it.uu.se/~ps/SAS-new.pdf for book contents.
  """
  M = len(w)
  N = len(y)
  if M>N:
    raise ValueError('Window cannot be longer than data')
  r, lags = acorrBiased(y)
  r = r[np.logical_and(lags >= 0, lags < M)]
  rw = r * w
  phi = 2 * fft.fft(rw, Nfft).real - rw[0];
  f = np.arange(Nfft) / Nfft;
  return (phi[f < 0.5], f[f < 0.5] * fs)


btWin = hamming(1024*8)
(pBT, fBT) = blackmanTukey(y, btWin, 16*1024, fs)

plt.close()
plt.plot(fBT, db20(pBT))
plt.xlabel('frequency (Hz)')
plt.ylabel('spectrum estimate')
plt.title('Blackman-Tukey spectral estimate, {}-Hamming window'.format(btWin.size))

plt.gca().set_xticks(np.arange(0, 2000, 110))
plt.xlim([0, 2000])
plt.tight_layout()

plt.savefig('btse.png')