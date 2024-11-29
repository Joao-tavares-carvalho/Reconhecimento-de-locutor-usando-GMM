import os
import soundfile as sf
import wavio
import numpy
import python_speech_features as psf
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
import streamlit as st
#from sklearn.preprocessing import StandardScaler
import webrtcvad
import struct
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt


#####variaveis de Audio_sem_silencio
#comprimento da janela
vad_window_length = 10
#taxa de amostragem
Samplerate = 16000
# Número de quadros para calcular a média juntos ao realizar a suavização da média móvel.
# Quanto maior esse valor, maiores devem ser as variações de VAD para não serem suavizadas.
vad_moving_average_width = 2
# Número máximo de quadros silenciosos consecutivos que um segmento pode ter.
vad_max_silence_length = 1    #padrao 6
int16_max = (2 ** 15) - 1



#### script para guardar nome do diretorio num Diretorio_Treno.txt
def Treno(caminho_audio,diretorio_audio_txt):
    #diretorio_audio_txt = "C:/Users/User/Desktop/dados_de_audio/locutor_youtube/treno.txt"
    #caminho_audio = "C:/Users/User/Desktop/dados_de_audio/locutor_youtube/"

    file_paths = open(diretorio_audio_txt ,'w')

    for (dirpath, dirnames, filenames) in os.walk(caminho_audio):
        for f in filenames:

            file_path = os.path.join(dirpath, f)

            file_paths.write(file_path.replace('\\','/') + "\n")

    return file_paths

st.write("---")
st.subheader('COPIAR TODO O DIRETORIO DE UM PASTA E GUARADR EM UM ARQUIVO .txt')
st.write("---")

diretorio_audio_txt = st.text_input('Guardado no diretorio txt', "C:\\Users\\Patron_Zona\\Desktop\\dados_de_audio\\locutor_audio_concatenado\\concatenado.txt")
caminho_audio = st.text_input("Caminho do audio","\\")
iniciar=st.button("iniciar")
if iniciar:
    while True:
        try:
            Treno(caminho_audio,diretorio_audio_txt)
            st.success("concluido!")
            break
        except FileNotFoundError:
            st.error('**Erro no diretorio, por favor sertifique.** ')
            break



def Teste():

    txt_ficheiro = "C:/Users/Patron_Zona/Desktop/RECONHECIMENTO_LOCUTOR_GMM/TFC/Diretorio_Teste_Todo.txt"
    caminho_audio = "C:/Users/Patron_Zona/Desktop/RECONHECIMENTO_LOCUTOR_GMM/TFC/Audio_Teste"
    file_paths = open(txt_ficheiro,'w')

    for (dirpath, dirnames, filenames) in os.walk(caminho_audio):

        for f in filenames:
            file_path = os.path.join(dirpath, f)
            file_paths.write(file_path + "\n")
    return file_paths

#### script para representar duracao de um arquivo  de audio_audio concatenado
def concatenado():

    caminho_audio_guardado = 'C:/Users/Patron_Zona/Desktop/dados_de_audio/locutor_audio_concatenado/concatenado/audio.wav'
    diretorio_audio_txt = "C:/Users/Patron_Zona/Desktop/dados_de_audio/locutor_audio_concatenado/concatenado.txt"

    with open(diretorio_audio_txt) as f:
        arquivos = f.readlines()

    srr = 16000
    todos_dados = np.array([])
    for n_arquivos in arquivos:

        arquivos_dados, sr = sf.read(n_arquivos.strip())
        todos_dados = np.concatenate((todos_dados, arquivos_dados))

    #sf.write( caminho_audio_guardado,todos_dados,srr,1)
    wavio.write(caminho_audio_guardado, todos_dados, 16000, sampwidth=2)


    return


st.subheader('CONCATENAR AUDIO')
st.write("diterorio de audio foi pre configurado e o local tmb na pasta concatenado")
conc = st.button("concatenar")

if conc:
    concatenado()
    st.success("concluido!")




#### script para trocar numero por nome em um arquivo
def mudar_primeiro_numero_por_nome(diretorio, nome):



    for (dirpath, dirnames, filenames) in os.walk(diretorio):


        contador = 0
        for f in filenames:
            contador = contador+1
            file_oldname = os.path.join(dirpath,f)

            file_newname_newfile = os.path.join(dirpath, f.replace(f , nome+"-"+f.split("-")[-2]+"-"+ f.split("-")[-1]))

            os.rename(file_oldname, file_newname_newfile)

            st.write(file_newname_newfile)



st.write("---")
st.subheader('MUDAR PRIMEIRO NOME POR NUMERO')
st.write("---")

diretorio_audio = st.text_input('DERETORIO DE AUDIO', placeholder="\\")
nome = st.text_input("introdusa um nome")
mudar = st.button("mudar")

if mudar:
    mudar_primeiro_numero_por_nome(diretorio_audio,nome)

#### reducao do ruido usando modulo
import noisereduce as nr
#reduced_noise = nr.reduce_noise(y=arquivos_dados, sr=sr)

###### Calculo de coeficientes delta
def delta_mfcc(feat, N):
    if N < 1:
        raise ValueError('N deve ser um inteiro >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')  # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(numpy.arange(-N, N + 1),
                                  padded[t: t + 2 * N + 1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    print('delta_mfcc_terminado')
    return delta_feat

###### scipt para estrair mfcc e coeficientes delta bibliiotecas
def mfcc_audio():

    Arquivos_teste = "C:/Users/Patron_Zona/Desktop/RECONHECIMENTO_LOCUTOR_GMM/TFC/audio_concatenado.txt"
    with open(Arquivos_teste) as f:
        arquivos = f.readlines()

    mfc = []
    for n_arquivos in arquivos:
        arquivos_dados, sr = sf.read(n_arquivos.strip())

        MFCC = psf.mfcc(arquivos_dados, 16000, numcep=13,
                                                winlen=20 / 1000,
                                                winstep=10 / 1000, nfft=1200)

        mfcc_feature = preprocessing.scale(MFCC)

        delta = delta_mfcc(mfcc_feature, 2)
        combined = np.hstack((mfcc_feature, delta))
        return combined
    print('mfcc_audio_terminado')
def extract_features(audio, rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines
    delta to make it 40 dim feature vector"""

    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    #delta = calculate_delta(mfcc_feature)
    #combined = np.hstack((mfcc_feature, delta))
    #return combined
    print('apenas_audio_terminado')

#### script para recortar audio em 1:40 segundos
def recortar_audio():
    from pydub import AudioSegment
    from pydub.utils import make_chunks

    ## bluesfile 30s
    audio = AudioSegment.from_file("C:/Users/Patron_Zona/Downloads/tratamento/audio_sem_silencio/Geremias_correia.wav", "wav")

    size = 110000 ## Los milisegundos de corte

    chunks = make_chunks (audio, size) ## Corta el archivo en trozos de 10 segundos

    for i, chunk in enumerate(chunks):

        chunk_name = 'C:/Users/Patron_Zona/Downloads/tratamento/recortado/Geremias_correia-00-0{}.wav'.format(i)
        print(chunk_name)

        chunk.export(chunk_name, format="wav")

    print('recorte_audio_terminado')
#recortar_audio()


def audio(diretorio_audio):
    arquivos_dados, sr = sf.read(diretorio_audio.strip())
    #st.write(arquivos_dados)
    return arquivos_dados



def recortar_silencio(wav , vad_window_length, sampling_rate):
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)



    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True] , audio_mask



#audio_recortado, audio_mask = recortar_silencio(audio(diretorio_audio) , vad_window_length, Samplerate)

#audio_normal = audio(diretorio_audio)
#time =np.arange(0, len(audio_normal)) /Samplerate

#fig, ax = plt.subplots()
#ax.plot()
#ax.plot(time,audio_normal)
#plt.show()
#st.pyplot(fig)

#plt.figure(figsize=(14,3))
#plt.plot(time,audio_normal)
#plt.plot(audio_mask*0.2)
#plt.xlabel("Tempo")
#plt.ylabel("Amplitude")
#st.pyplot(fig=plt)



con = st.container()
#con2 = st.container
con.write("---")
con.subheader('RECORTE DE SILENCIO')
con.write("---")

diretorio_audio = con.text_input("Ditetorio de audio")
caminho_audio = con.text_input("Diretorio para guardar audio")
#inicia=st.button("butao")

inicia = con.button('INICIAR')
if inicia:
    #audio(diretorio_audio)
    dados = recortar_silencio(audio(diretorio_audio), vad_window_length, Samplerate)
    audio=audio(diretorio_audio)
    wavio.write(caminho_audio, dados, 16000, sampwidth=2)
    #st.write("RECORTADO COM SUCEÇO")

    #reprodizir audio
    st.markdown("**Audio normal**")
    audio_file = open(diretorio_audio, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='wav')

    #reproduzir audio
    st.markdown("**Audio com selencio recortado**")
    audio_file = open(caminho_audio, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='wav')

    fig, ax = plt.subplots()
    ax.plot(audio)
    st.pyplot(fig)






#st.subheader('AUDIO RECORTADO')
#caminho_audi = st.text_input("Arquivo txt de novo audio")
#butao = ("audio recortado")

#agree = st.checkbox('AUDIO')
#if agree:
#    audio_file = open(caminho_audi, 'rb')
#    audio_bytes = audio_file.read()
#    st.audio(audio_bytes, format='wav')