from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# Leer el archivo .wav
samplerate, data = wavfile.read('impulse_amp1.0_rep5.wav')

# samplerate: Frecuencia de muestreo en Hz (número de muestras por segundo)
# data: Array de amplitudes de la señal

# Si la señal es estéreo, usar un solo canal
if len(data.shape) > 1:
    data = data[:, 0]

# Normalizar los datos si es necesario (convertir a flotantes)
data = data / np.max(np.abs(data))

# Eje de tiempo para la señal original
N = len(data)  # Número total de muestras
time = np.linspace(0, N / samplerate, N)  # Eje de tiempo en segundos

# Graficar la señal original
plt.figure(figsize=(12, 6))
plt.plot(time, data, label='Señal original')
plt.title('Señal en el dominio del tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.show()

# Calcular la FFT
frequencies = np.fft.fftfreq(N, d=1/samplerate)  # Eje de frecuencias
spectrum = np.fft.fft(data)  # Transformada de Fourier

# Tomar la magnitud del espectro (frecuencia positiva)
positive_frequencies = frequencies[:N // 2]
magnitude = np.abs(spectrum[:N // 2])

# Graficar el espectro de frecuencias
plt.figure(figsize=(12, 6))
plt.plot(positive_frequencies, magnitude, label='Espectro de frecuencias')
plt.title('Espectro de Frecuencias')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid()
plt.legend()
plt.show()

# ----------------------------------------------------------------

# Leer el archivo .wav
samplerate_bb, data_bb = wavfile.read('impulse_amp1.0_rep5_bb.wav')

# samplerate: Frecuencia de muestreo en Hz (número de muestras por segundo)
# data: Array de amplitudes de la señal

# Si la señal es estéreo, usar un solo canal
if len(data_bb.shape) > 1:
    data_bb = data_bb[:, 0]

# Normalizar los datos si es necesario (convertir a flotantes)
data_bb = data_bb / np.max(np.abs(data_bb))

# Eje de tiempo para la señal original
N_bb = len(data_bb)  # Número total de muestras
time_bb = np.linspace(0, N_bb / samplerate_bb, N_bb)  # Eje de tiempo en segundos

# Graficar la señal original
plt.figure(figsize=(12, 6))
plt.plot(time_bb, data_bb, label='Señal original(BB)')
plt.title('Señal en el dominio del tiempo(BB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid()
plt.legend()
plt.show()

# Calcular la FFT
frequencies_bb = np.fft.fftfreq(N, d=1/samplerate_bb)  # Eje de frecuencias
spectrum_bb = np.fft.fft(data_bb)  # Transformada de Fourier

# Tomar la magnitud del espectro (frecuencia positiva)
positive_frequencies_bb = frequencies_bb[:N // 2]
magnitude_bb = np.abs(spectrum_bb[:N // 2])

# Graficar el espectro de frecuencias
plt.figure(figsize=(12, 6))
plt.plot(positive_frequencies_bb, magnitude_bb, label='Espectro de frecuencias(BB)')
plt.title('Espectro de Frecuencias(BB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid()
plt.legend()
plt.show()
