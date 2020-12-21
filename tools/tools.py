import numpy as np
from scipy.integrate import simps
from scipy import signal as signal_processing
from scipy.signal import butter,lfilter
import pywt
from PyEMD import EMD
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import random as rand
import pyeeg 
from hurst import compute_Hc

mean_energy =  lambda x: np.sum(x*x)/x.size

def split(array,nb):
    array = np.array(np.split(array, nb, axis= len(array.shape)-1 ))
    return array.reshape(-1, array.shape[-1])

def bandpower(signal,sf, low, high):
 
    freqs = np.fft.fftfreq(signal.size, 1/sf)
    idx = np.logical_and(freqs >= low, freqs <= high)
    psd = np.abs(np.fft.fft(signal))**2

    return simps(psd[idx],freqs[idx])


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    #print(b,a)
    y = lfilter(b, a, data)
    return y

def split_in_bands(signal,power_bands,fs,order=2):
    splited_signals = []
    for band in power_bands:
        splited_signals.append(butter_bandpass_filter(signal, band[0],  band[1], fs, order=2))
    return splited_signals

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

def treat_record_wpt_pfd(signal,wavelet="sym3"):

    subband_p = pywt.WaveletPacket(signal,wavelet,maxlevel=3)
    leaf_subbands=np.array([n.data for n in subband_p.get_leaf_nodes(True)])
    feature_vector1 = np.array([mean_energy(subband) for subband in leaf_subbands])
    feature_vector2 = np.array([pyeeg.pfd(subband) for subband in leaf_subbands]) 
    feature_vector = np.concatenate((feature_vector1,feature_vector2),axis=0)

    return feature_vector

def treat_record_wpt(signal,wavelet="sym3"):

    subband_p = pywt.WaveletPacket(signal,wavelet,maxlevel=3)
    leaf_subbands=np.array([n.data for n in subband_p.get_leaf_nodes(True)])
    feature_vector = np.array([mean_energy(subband) for subband in leaf_subbands])

    return feature_vector

def treat_record_EMD(signal):

    emd = EMD()
    IMFs = emd(signal)[:5]
    feature_vector = [mean_energy(subband) for subband in IMFs]

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

""" pick random sample with equal proportion from each class"""

def record_eeg(index,h5):
    data = []

    for key in h5.keys():
        if "eeg" in key: 
            data.append(np.array(h5[key])[index])

    return np.transpose(data,(1, 0, 2))

def record_features(index,h5, features =[]):
    data = []
    
    for key in h5.keys():
        if  len(features)==0 or np.all([feature in key for feature in features]):
            data.append(np.array(h5[key])[index])

    return np.array(data).T

def pick_samples(len_sample,labels):
    len_sample = len_sample//5
    samples_0 =rand.sample([record for record in range(len(labels)) if labels[record] == 0],len_sample)
    samples_1 =rand.sample([record for record in range(len(labels)) if labels[record] == 1],len_sample)
    samples_2 =rand.sample([record for record in range(len(labels)) if labels[record] == 2],len_sample)
    samples_3 =rand.sample([record for record in range(len(labels)) if labels[record] == 3],len_sample)
    samples_4 =rand.sample([record for record in range(len(labels)) if labels[record] == 4],len_sample)
    
    samples_index = np.array([samples_0,samples_1,samples_2,samples_3,samples_4]).ravel()
    
    return samples_index

def pick_train_test(len_sample,labels):
    len_sample = len_sample//5
    samples_0 =rand.sample([record for record in range(len(labels)) if labels[record] == 0],len_sample)
    samples_1 =rand.sample([record for record in range(len(labels)) if labels[record] == 1],len_sample)
    samples_2 =rand.sample([record for record in range(len(labels)) if labels[record] == 2],len_sample)
    samples_3 =rand.sample([record for record in range(len(labels)) if labels[record] == 3],len_sample)
    samples_4 =rand.sample([record for record in range(len(labels)) if labels[record] == 4],len_sample)
    
    samples_index = np.array([samples_0,samples_1,samples_2,samples_3,samples_4]).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(samples_index, labels[samples_index])
    
    return X_train, X_test, y_train, y_test


"""very bad fuction to map a fuction"""

def treat_samples(all_samples,function):
    treated_samples = []
    
    for sample in all_samples:
        treated_samples.append([function(eeg) for eeg in sample])
        
    return np.array(treated_samples)

def treat_samples_map(all_samples,function):

    result = np.array(list(map(lambda channels : list(map(lambda eeg :function(eeg) ,channels)), all_samples)))

    return np.reshape(np.transpose(result,(1,0,2)),(result.shape[1],result.shape[0]*result.shape[2]) )

