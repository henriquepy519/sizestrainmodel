import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def load_ascii_data(file_path):
    data = np.loadtxt(file_path)
    x, y = data[:, 0], data[:, 1]
    return x, y


def plot_xrd_data(x, y, markersize=4, peak_height=50, peak_distance=10, peak_prominence=0.3, peak_width=5):
    x_2theta = x
    peaks, properties = find_peaks(y, height=peak_height, distance=peak_distance, prominence=peak_prominence,
                                   width=peak_width)
    peak_positions = x[peaks]
    peak_values = y[peaks]
    plt.figure(figsize=(10, 6))
    plt.scatter(x_2theta, y, color='black', s=markersize, label='Data')
    plt.scatter(x_2theta[peaks], y[peaks], color='red', s=markersize * 2, label='Peaks')
    plt.title('X-ray Diffraction Pattern of Mica')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True)
    plt.legend()
    plt.show()


def gaussian(x1, amp, mu, sig):
    return amp * np.exp(-(x1 - mu) ** 2 / (2 * sig ** 2))


def lorentzian(x1, amp, mu, sig):
    return amp / (1 + ((x1 - mu) / sig) ** 2)


def mixed_model(x1, amp_g, mu_g, sig_g, amp_l, mu_l, sig_l):
    return gaussian(x1, amp_g, mu_g, sig_g) + lorentzian(x1, amp_l, mu_l, sig_l)


def fwhm_gaussian(sig):
    return 2 * np.sqrt(2 * np.log(2)) * sig


def fwhm_lorentzian(sig):
    return 2 * sig


def fwhm_mixed(sig_g, sig_l):
    return 0.5346 * fwhm_lorentzian(sig_l) + np.sqrt(0.2166 * fwhm_lorentzian(sig_l) ** 2 + fwhm_gaussian(sig_g) ** 2)


def automatic_peak_fitting(x, y):
    peaks, _ = find_peaks(y, height=50, distance=10, prominence=0.3, width=5)
    peak_positions = x[peaks]
    results = []

    for peak in peaks:
        # Parâmetros iniciais para o ajuste
        amp_gauss = y[peak]
        mu_gauss = x[peak]
        sig_gauss = 0.1

        amp_lorentz = y[peak]
        mu_lorentz = x[peak]
        sig_lorentz = 0.1

        initial_guess_peak = [
            amp_gauss, mu_gauss, sig_gauss,  # Parâmetros da gaussiana
            amp_lorentz, mu_lorentz, sig_lorentz  # Parâmetros da lorentziana
        ]

        bounds_gauss = [0, mu_gauss - 10, 0.01]
        bounds_lorentz = [0, mu_lorentz - 10, 0.01]
        upper_bounds_gauss = [amp_gauss * 2, mu_gauss + 10, 0.5]
        upper_bounds_lorentz = [amp_lorentz * 2, mu_lorentz + 10, 0.5]

        bounds = (
            bounds_gauss + bounds_lorentz,
            upper_bounds_gauss + upper_bounds_lorentz
        )

        try:
            popt, pcov = curve_fit(mixed_model, x, y, p0=initial_guess_peak, bounds=bounds)
            results.append((popt, peak))
        except RuntimeError as e:
            print(f"Error fitting peak at {x[peak]}: {e}")
            continue

    return peaks, results


def automatic_peak_fitting1(x1, y1):
    peaks1, _ = find_peaks(y1, height=50, distance=10, prominence=0.3, width=5)
    peak_positions1 = x1[peaks1]
    results1 = []

    for peak in peaks1:
        amp_gauss = y1[peak]
        mu_gauss = x1[peak]
        sig_gauss = 10

        amp_lorentz = y1[peak]
        mu_lorentz = x1[peak]
        sig_lorentz = 10

        initial_guess_peak = [
            amp_gauss, mu_gauss, sig_gauss,
            amp_lorentz, mu_lorentz, sig_lorentz
        ]

        bounds_gauss = [0, mu_gauss - 10, 0.01]
        bounds_lorentz = [0, mu_lorentz - 10, 0.01]
        upper_bounds_gauss = [amp_gauss * 2, mu_gauss + 10, 0.5]
        upper_bounds_lorentz = [amp_lorentz * 2, mu_lorentz + 10, 0.5]

        bounds = (
            bounds_gauss + bounds_lorentz,
            upper_bounds_gauss + upper_bounds_lorentz
        )

        try:
            popt, pcov = curve_fit(mixed_model, x1, y1, p0=initial_guess_peak, bounds=bounds)
            results1.append((popt, peak))
        except RuntimeError as e:
            print(f"Error fitting peak at {x1[peak]}: {e}")
            continue

    return peaks1, results1


def save_peak_parameters(results, x, filename="peak_parameters_Bragg_Brentano_Geometry.txt"):
    with open(filename, 'w') as f:
        f.write("Peak parameters (Gaussian and Lorentzian fits):\n")
        f.write(
            "Peak  |  2θ (degrees)  |  Amp_Gauss  |  Mu_Gauss  |  Sig_Gauss  |  Amp_Lorentz  |  Mu_Lorentz  |  Sig_Lorentz\n")
        f.write("-" * 100 + "\n")
        for i, (popt, peak) in enumerate(results):
            f.write(
                f"{i + 1:4d}  |  {x[peak]:12.4f}  |  {popt[0]:10.4f}  |  {popt[1]:9.4f}  |  {popt[2]:10.4f}  |  {popt[3]:12.4f}  |  {popt[4]:11.4f}  |  {popt[5]:11.4f}\n")


def save_statistical_data(results, x, filename="statistical_data_Bragg_Brentano_Geometry.txt"):
    with open(filename, 'w') as f:
        f.write("Statistical data (Gaussian and Lorentzian fits):\n")
        f.write("Peak  |  2θ (degrees)  |  FWHM_Gauss  |  FWHM_Lorentz  |  FWHM_Mixed\n")
        f.write("-" * 70 + "\n")
        for i, (popt, peak) in enumerate(results):
            f.write(
                f"{i + 1:4d}  |  {x[peak]:12.4f}  |  {fwhm_gaussian(popt[2]):10.4f}  |  {fwhm_lorentzian(popt[5]):10.4f}  |  {fwhm_mixed(popt[2], popt[5]):10.4f}\n")


def save_peak_parameters1(results1, x1, filename="peak_parameters_GIXRD_Geometry.txt"):
    with open(filename, 'w') as f:
        f.write("Peak parameters (Gaussian and Lorentzian fits):\n")
        f.write(
            "Peak  |  2θ (degrees)  |  Amp_Gauss  |  Mu_Gauss  |  Sig_Gauss  |  Amp_Lorentz  |  Mu_Lorentz  |  Sig_Lorentz\n")
        f.write("-" * 100 + "\n")
        for i, (popt, peak1) in enumerate(results1):
            f.write(
                f"{i + 1:4d}  |  {x1[peak1]:12.4f}  |  {popt[0]:10.4f}  |  {popt[1]:9.4f}  |  {popt[2]:10.4f}  |  {popt[3]:12.4f}  |  {popt[4]:11.4f}  |  {popt[5]:11.4f}\n")


def save_statistical_data1(results1, x1, filename="statistical_data_GIXRD_Geometry.txt"):
    with open(filename, 'w') as f:
        f.write("Statistical data (Gaussian and Lorentzian fits):\n")
        f.write("Peak  |  2θ (degrees)  |  FWHM_Gauss  |  FWHM_Lorentz  |  FWHM_Mixed\n")
        f.write("-" * 70 + "\n")
        for i, (popt, peak1) in enumerate(results1):
            f.write(
                f"{i + 1:4d}  |  {x1[peak1]:12.4f}  |  {fwhm_gaussian(popt[2]):10.4f}  |  {fwhm_lorentzian(popt[5]):10.4f}  |  {fwhm_mixed(popt[2], popt[5]):10.4f}\n")


# Carrega os arquivos

ascii_file_path = 'Mica_TT-TT.ASC'
x, y = load_ascii_data(ascii_file_path)

ascii_file_path = 'Mica_GIXRD.ASC'
x1, y1 = load_ascii_data(ascii_file_path)

# Detecção e ajuste
peaks, results = automatic_peak_fitting(x, y)
peaks1, results1 = automatic_peak_fitting(x1, y1)

# Salvar os parâmetros ajustados em um arquivo txt
save_peak_parameters(results, x)
save_peak_parameters1(results1, x1)
# Salvar os dados estatísticos em um arquivo txt
save_statistical_data(results, x)
save_statistical_data1(results1, x1)

# Visualização dos resultados do ajuste

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Dados')
for popt, peak in results:
    # Definir a faixa em torno do pico para visualização
    peak_range = 1.2  # Ajuste conforme necessário para focar nos picos
    mask = (x > x[peak] - peak_range) & (x < x[peak] + peak_range)
    x_fit = x[mask]
    y_fit = mixed_model(x_fit, *popt)
    plt.plot(x_fit, y_fit, '-', color='red', linewidth=2)
plt.xlabel('2θ (degrees)')
plt.ylabel('Intensity (a.u.)')
plt.legend(title='Desconvolução de Picos')
plt.title('Perfil de Ajustado - Geometria Bragg-Brentano')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x1, y1, 'o', label='Dados')
for popt, peak1 in results1:
    # Definir a faixa em torno do pico para visualização
    peak_range1 = 0.8  # Ajuste conforme necessário para focar nos picos
    mask1 = (x1 > x1[peak1] - peak_range1) & (x1 < x1[peak1] + peak_range1)
    x1_fit1 = x1[mask1]
    y1_fit1 = mixed_model(x1_fit1, *popt)
    plt.plot(x1_fit1, y1_fit1, '-', color='red', linewidth=2)
plt.xlabel('2θ (degrees)')
plt.ylabel('Intensity (a.u.)')
plt.legend(title='Desconvolução de Picos')
plt.title('Perfil de Ajuste - GIXRD')
plt.show()

# Visualização de cada pico individualmente, focando no pico
for i, (popt, peak) in enumerate(results):
    # Definir a faixa em torno do pico para visualização
    peak_range = 0.3  # Ajuste conforme necessário para focar nos picos
    mask = (x > x[peak] - peak_range) & (x < x[peak] + peak_range)
    x_peak = x[mask]
    y_peak = y[mask]

    plt.figure(figsize=(10, 6))
    plt.plot(x_peak, y_peak, 'o', label='Dados')
    plt.plot(x_peak, gaussian(x_peak, *popt[:3]), '--', label='Pico Gaussiano')
    plt.plot(x_peak, lorentzian(x_peak, *popt[3:]), '--', label='Pico Lorentziano')
    yfit_peak = mixed_model(x_peak, *popt)
    plt.plot(x_peak, yfit_peak, '-', color='red', linewidth=2, label='Ajuste Total')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(title='Desconvolução de Picos')
    plt.title(f'Ajuste do Pico {i + 1} em {x[peak]:.2f} - Geometria Bragg-Brentano')

    plt.show()

# Visualização de cada pico individualmente, focando no pico
for i, (popt, peak1) in enumerate(results1):
    # Definir a faixa em torno do pico para visualização
    peak_range1 = 0.3  # Ajuste conforme necessário para focar nos picos
    mask1 = (x1 > x1[peak1] - peak_range1) & (x1 < x1[peak1] + peak_range1)
    x1_peak1 = x1[mask1]
    y1_peak1 = y1[mask1]

    plt.figure(figsize=(10, 6))
    plt.plot(x1_peak1, y1_peak1, 'o', label='Dados')
    plt.plot(x1_peak1, gaussian(x1_peak1, *popt[:3]), '--', label='Pico Gaussiano')
    plt.plot(x1_peak1, lorentzian(x1_peak1, *popt[3:]), '--', label='Pico Lorentziano')
    y1fit_peak1 = mixed_model(x1_peak1, *popt)
    plt.plot(x1_peak1, y1fit_peak1, '-', color='red', linewidth=2, label='Ajuste Total')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(title='Desconvolução de Picos')
    plt.title(f'Ajuste do Pico {i + 1} em {x1[peak1]:.2f} graus - GIXRD')
    plt.show()
