import numpy as np
from scipy.integrate import simps
from scipy import signal as signal_processing
import pywt
from scipy.stats import zscore

mean_energy =  lambda x: np.sum(x*x)/x.size

def split(array,nb):
    array = np.array(np.split(array, nb, axis= len(array.shape)-1 ))
    return array.reshape(-1, array.shape[-1])

def bandpower(signal,sf, low, high):
 
    freqs = np.fft.fftfreq(signal.size, 1/sf)
    idx = np.logical_and(freqs >= low, freqs <= high)
    psd = np.abs(np.fft.fft(signal))**2

    return simps(psd[idx],freqs[idx])


def treat_record_fft(signal,power_bands,sf):
    results = { key : 0 for key in power_bands.keys()}

   
    sliced_signals = split(signal,10)
    taped_signal = np.hamming(sliced_signals.shape[1])*sliced_signals
    
    for sub_signal in taped_signal:

        total_powerband = bandpower(sub_signal,sf,-np.inf,np.inf)
        for (name,(low,high)) in power_bands.items():

            # Pas optimal de recalculer la fft à chaque tour de boucle
            results[name] += bandpower(sub_signal,sf, low, high)/total_powerband
    
    results = {key : results[key]/len(taped_signal) for key in power_bands.keys()}

    return  np.array(list(results.values()))


def treat_record_welch(signal,power_bands,sf):
    results = { key : 0 for key in power_bands.keys()}

    win = 4 * sf
    freqs, psd = signal_processing.welch(signal, sf, nperseg=win)
    freq_res = freqs[1] - freqs[0]
    total_power = simps(psd, dx=freq_res)

    for (name,(low,high)) in power_bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        power = simps(psd[idx], dx=freq_res)
        results[name] = power/total_power

    return np.array(list(results.values()))

def treat_record_wpt(signal,wavelet="sym3"):

    subband_p = pywt.WaveletPacket(signal,wavelet,maxlevel=3)
    leaf_subbands=np.array([n.data for n in subband_p.get_leaf_nodes(True)])
    feature_vector = [mean_energy(subband) for subband in leaf_subbands]

    return feature_vector

def treat_record_dwt(signal,wavelet="db2"):

    subband = pywt.wavedec(signal,wavelet,level=10)[1:]
    feature_vector = [mean_energy(subband) for subband in subband]

    return feature_vector

def get_filter_abnormal_values(samples):

    z_scores = zscore(samples)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)

    return filtered_entries
