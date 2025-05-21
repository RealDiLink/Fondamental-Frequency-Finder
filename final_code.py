from os import walk
import csv
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import scipy.io.wavfile as wavfile
from scipy.signal.windows import hamming
from scipy.optimize import curve_fit

folder = []

masses_line = {
   "E2" : 0.0059,
   "A2" : 0.0038,
   "D3" : 0.0023,
   "G3" : 0.00095,
   "B3" : 0.00052,
   "E4" : 0.00035
}

tensions = {
   "E2" : 67.24,
   "A2" : 77.16,
   "D3" : 83.21,
   "G3" : 61.24,
   "B3" : 53.21,
   "E4" : 63.82
}

def f(x, a, b):
   return a*x + b

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



for (dirpath, dirnames, filenames) in walk("C://Users//drago//Desktop//Projet//Expérience guitare"):
    folder.extend(dirnames)
    break

for f_names in folder :
  if(f_names in ["A2", "B3", "D3", "E2", "E4", "G3"]):
    for (dirpath, dirnames, filenames) in walk("C://Users//drago//Desktop//Projet//Expérience guitare//" + f_names):
        final_data = []
        freq = []
        longueurs = 1/np.array([0.6477, 0.6133, 0.5772, 0.5454, 0.5157, 0.4880, 0.4622, 0.4380, 0.4154, 0.3944, 0.3747, 0.3563, 0.32385, 0.3053, 0.2879, 0.2713])
        incertitude_longueurs = [np.sqrt(1/x**4*0.005**2) for x in longueurs]
        incertitudes_freq = [np.sqrt((((-1)/(2*x**2))*np.sqrt(tensions[f_names]/masses_line[f_names]))**2)*0.005**2 for x in longueurs]
        for file in filenames :    
            if "Son" in file:
                (fs, y) = wavfile.read("C://Users//drago//Desktop//Projet//Expérience guitare//" + f_names + "//" + file)
                y = y.astype(float) / 2**16 # normalize to -1 to 1
                y = y[:,0] + y[:,1] # stereo to mono
                fs = float(fs)

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

                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                plt.style.use(['dark_background','ggplot'])

                print(hpsT)

                plt.close()
                plt.imshow(db20(hpsArr), aspect='auto', interpolation='nearest',
                                extent=extents(hpsF) + extents(hpsT), cmap='viridis')
                plt.xlabel('frequency (Hz)')
                plt.ylabel('time (seconds)')
                plt.title('Harmonic product spectrum: # products={}, window length={}'.format(prod, winlen))
                plt.tight_layout()

                plt.xlim([0,1000])
                plt.savefig(f"C://Users//drago//Desktop//Projet//Expérience guitare//result//{f_names}-{file}-hps-0-1000.png")

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

                plt.gca().set_xticks(np.arange(0, 2000, 110))
                plt.xlim([0, 2000])
                plt.savefig(f"C://Users//drago//Desktop//Projet//Expérience guitare//result//{f_names}-{file}-hps-alldata-0-2000.png")

                if(f_names == "B3" and file == "Son 02.wav"): #gros chommage ici l'harmonique était supérieur de peu à la freq fonda, on la retire à la main mdrrr
                    hpsF = np.delete(hpsF, np.argmax(db20(hpsS)))
                    hpsS = np.delete(hpsS, np.argmax(db20(hpsS)))


                freq.append(np.round(hpsF[np.argmax(hpsS)],1))
                period = 1/np.round(hpsF[np.argmax(hpsS)],1)
                final_data.append(dict(Corde_number = file.replace('Son ', ' '), Fundamental_Frequency = np.round(hpsF[np.argmax(hpsS)],1), Period = '{:.2e}'.format(period)))

            if len(freq) == 16:
                param, pcov = curve_fit(f, longueurs, freq)
                x_data = np.linspace(min(longueurs), max(longueurs), 100)
                y_data = [f(x, param[0], param[1]) for x in x_data]
                plt.close()
                plt.style.use(['default','fast'])
                plt.errorbar(longueurs, freq, xerr=incertitude_longueurs, yerr=incertitudes_freq, label="Valeurs expérimentales")
                plt.errorbar(x_data, y_data, label=f"Régression linéaire : a = {np.round(param[0],1)}")
                plt.legend()
                plt.xlabel("1/L en m^-1")
                plt.ylabel("Fréquence en Hz")
                plt.title(f"{f_names} Fréquence fondamentale en fonction de 1 sur la longueur")
                plt.savefig(f"C://Users//drago//Desktop//Projet//Expérience guitare//final_result//{f_names}-graph-final.png")

                with open(f"C://Users//drago//Desktop//Projet//Expérience guitare//final_result//{f_names}" + '.csv', 'w', newline='') as csvfile:
                    fieldnames = ['Corde_number', 'Fundamental_Frequency', 'Period']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
                    writer.writeheader()
                    writer.writerows(final_data)



      



