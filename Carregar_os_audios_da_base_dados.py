import soundfile as sf
import numpy as np
#import noisereduce as nr


#### script para usar o arquivo .txt como diretorio para ler o oudio
def dados_audio_teste(diretorio_teste):

    total = 0.0
    #Arquivos_teste = "C:/Users/User/Desktop/RECONHECIMENTO_LOCUTOR_GMM/TFC/Diretorio_Teste_Todo.txt"   #teste

    with open(diretorio_teste) as f:
        arquivos = f.readlines()

    identificador = []
    audio = []
    for n_arquivos in arquivos:

        arquivos_dados, sr = sf.read(n_arquivos.strip())

        #ruido_eleminado = nr.reduce_noise(y=arquivos_dados, sr=sr)

        audio.append(arquivos_dados)
        identificador.append(n_arquivos.split("/")[-1].split("-")[0])
        total+=1

    return audio, identificador, total



def dados_audio_treno(nome_pasta, diretorio_treno):

    #Arquivos_treino = "C:/Users/User/Desktop/RECONHECIMENTO_LOCUTOR_GMM/TFC/Diretorio_Treno_Outro.txt"
    #Arquivos_treino = "C:/Users/User/Desktop/dados_de_audio/locutor_youtube/treno.txt"

    with open(diretorio_treno) as f:
        arquivos = f.readlines()


    todos_dados = np.array([])
    for n_arquivos in arquivos:

        if str(nome_pasta) == (n_arquivos.split("/")[-2]):
            arquivos_dados, sr = sf.read(n_arquivos.strip())
            todos_dados = np.concatenate((todos_dados, arquivos_dados))

    return todos_dados


