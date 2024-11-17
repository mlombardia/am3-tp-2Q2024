from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# Leer las señales de entrada y salida en formato .wav
samplerate_in, input_signal = wavfile.read('impulse_amp1.0_rep5.wav')
samplerate_out, output_signal = wavfile.read('impulse_amp1.0_rep5_bb.wav')

# Asegurarse de que las frecuencias de muestreo coincidan
assert samplerate_in == samplerate_out, "Las frecuencias de muestreo deben coincidir"

# Si las señales son estéreo, usar un solo canal
if len(input_signal.shape) > 1:
    input_signal = input_signal[:, 0]
if len(output_signal.shape) > 1:
    output_signal = output_signal[:, 0]

# Asegurar que ambas señales tengan la misma longitud
min_length = min(len(input_signal), len(output_signal))
input_signal = input_signal[:min_length]  # Recortar entrada
output_signal = output_signal[:min_length]  # Recortar salida

# Normalizar las señales
input_signal = input_signal / np.max(np.abs(input_signal))
output_signal = output_signal / np.max(np.abs(output_signal))

# Calcular la FFT de entrada y salida
N = len(input_signal)
frequencies = np.fft.fftfreq(N, d=1/samplerate_in)
input_fft = np.fft.fft(input_signal)
output_fft = np.fft.fft(output_signal)

# Calcular la función de transferencia en el dominio de la frecuencia (H(f) = salida / entrada)
H = output_fft / input_fft  # Dividir espectros

# Obtener la respuesta en magnitud y fase (frecuencias positivas)
positive_frequencies = frequencies[:N // 2]
magnitude = 20 * np.log10(np.abs(H[:N // 2]))  # Magnitud en dB
phase = np.angle(H[:N // 2]) * 180 / np.pi  # Fase en grados

# Graficar el diagrama de Bode
plt.figure(figsize=(12, 8))

# Gráfico de magnitud
plt.subplot(2, 1, 1)
plt.plot(positive_frequencies, magnitude)
plt.title('Diagrama de Bode: Magnitud')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (dB)')
plt.grid()

# Gráfico de fase
plt.subplot(2, 1, 2)
plt.plot(positive_frequencies, phase)
plt.title('Diagrama de Bode: Fase')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Fase (grados)')
plt.grid()

plt.tight_layout()
plt.show()
