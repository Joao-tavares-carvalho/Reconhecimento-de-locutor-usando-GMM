import streamlit as st
import numpy as np
import python_speech_features as psf
import webrtcvad
from B_GMM_algoritimo import classify_probability
import pickle
from Carregar_os_audios_da_base_dados import dados_audio_teste, dados_audio_treno
import pandas as pd
from B_GMM_algoritimo import EM
import os
import sounddevice as sd
import wavio
from datetime import datetime
import matplotlib.pyplot as plt
import glob


#css file
with open("./css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.title('IDENTIFICAÇÃO DE LOCUTORES')

# Variaveis
fSamplerate = 16000
comprimento_do_quadro_ms = 20
comprimento_de_salto_ms = 10
Pre_enfase = 0.97
nfft=512        #1200
sVad = webrtcvad.Vad()
iSeed = 42
#txt_registo_dados = "10_locutores_30_13_N_100.txt"
#txt_registo_dados = "REGISTO_PARA_GRAFICO.txt"
caminho = 'Arquivo_txt_registo_dados'
caminho_arquivo_txt = glob.glob(os.path.join(caminho, "*.txt"))


# funcão para pegar apenas os dados de voz
def estrairApenas_dados_de_voz(fSamplerate, comprimento_do_quadro_ms, pfAllData):
    # Dividir os dados completos em Comprimento do quadro em MS / aqui sta definido 20 ms
    iVadSamplesPerFrame = int(fSamplerate * (comprimento_do_quadro_ms / 1000))
    ppfAllData = np.array(np.split(pfAllData[:int(pfAllData.shape[0] / iVadSamplesPerFrame) * iVadSamplesPerFrame],
                                   int(pfAllData.shape[0] / iVadSamplesPerFrame)))

    # Verifique se o quadro contém fala
    pbAllDataIsSpeech = np.zeros(ppfAllData.shape[0], dtype=bool)
    for iFrame in np.arange(ppfAllData.shape[0]):
        pbAllDataIsSpeech[iFrame] = sVad.is_speech(ppfAllData[iFrame, :], 16000)

    # Reconstruct speech signal with only speech
    pfOnlySpeech = ppfAllData[pbAllDataIsSpeech, :].flatten()

    return pfOnlySpeech


# Devisao dos dados para treinamento
def dados_de_treno_separados(ppfSig, comprimento_do_quadro_ms, Comprimento_dos_dados_de_treno_in_s):
    # Create array for training data
    iNumberOfTrainingFrames = int(Comprimento_dos_dados_de_treno_in_s / (comprimento_do_quadro_ms / 1000 / 2))
    ppfTrainingData = np.zeros((iNumberOfTrainingFrames, ppfSig.shape[1]))

    # Fill training data with mfccs from random frames
    piUsedFrames = []
    for iFrame in np.arange(iNumberOfTrainingFrames):
        # Make sure, we use new random frame

        while True:
            iRand = np.random.randint(0, ppfSig.shape[0] - 1)
            if iRand not in piUsedFrames:
                piUsedFrames.append(iRand)
                break
            else:
                break
        # Copy frame into training data set
        ppfTrainingData[iFrame, :] = ppfSig[iRand, :]

    return ppfTrainingData


def treinar_model(txt_treno, diretorio_mudel):

    with open(txt_treno) as f:
        arquivos = f.readlines()

    nome_duplicado = []
    nome_locutor = []
    for n_arquivos in arquivos:

        #nota_ver este trexo de codigo_modificar para 2 condicoes
        nome = (n_arquivos.split("/")[-2])
        nome_duplicado.append(nome)

    for elemento in nome_duplicado:
        if elemento not in nome_locutor:
            nome_locutor.append(elemento)

    #### Semente para gerar numero aleatorio
    if iSeed:
        np.random.seed(iSeed)

    #### dados de teinamento, todos os palestrante e o modelo de mundo
    pppf_dados_de_treno = []
    for Nome_pasta_txt in nome_locutor:
        with st.spinner('**_Processando dados, espere..._**'):
            todos_Dados = dados_audio_treno(Nome_pasta_txt, txt_treno)

            apenas_fala = estrairApenas_dados_de_voz(fSamplerate, comprimento_do_quadro_ms, todos_Dados)

            ppfMfcc = psf.mfcc(apenas_fala, fSamplerate, numcep=Numero_de_mfcc, winlen=comprimento_do_quadro_ms / 1000,
                               winstep=comprimento_de_salto_ms / 1000, nfft=nfft, preemph =Pre_enfase)

            #ppfMfcc = mfcc_delta(apenas_fala,fSamplerate,Numero_de_mfcc,comprimento_do_quadro_ms,comprimento_de_salto_ms,nfft,Pre_enfase)

            pppf_dados_de_treno.append(dados_de_treno_separados(ppfMfcc, comprimento_do_quadro_ms, Comprimento_dos_dados_de_treno_in_s))

    pppf_dados_de_treno = np.array(pppf_dados_de_treno)


    start = datetime.now()
    list_models = []
    for iModel in np.arange(pppf_dados_de_treno.shape[0]):

        with st.spinner(f'**Treinando o modelo pertence  ao locutor:  _{nome_locutor[iModel]}_**'):
            #### Modelo de treno com dados de treinamento fornecidos com algoritimo EM
            list_models.append(EM(pppf_dados_de_treno[iModel, :, :], Numero_de_gausiano_GMM, Iteracao_maxima_GMM))

        st.success(f'**_Modelo {nome_locutor[iModel]} treinado!_**')


        ## Guardar modelo treinada
        n_arquiv = nome_locutor[iModel]
        with open(diretorio_mudel + n_arquiv + ".gmm", 'wb') as f:
            pickle.dump(list_models, f)
    end = datetime.now()

    tempo = end - start
    st.write(f'Tempo total {tempo}')

    return tempo, nome_locutor

def testar_model(txt_teste, dir_model):

    erro = 0
    ## Carregar o modelo treinado
    #modelo_path = "C:/Users/User/Desktop/RECONHECIMENTO_LOCUTOR_GMM/TFC/Modelo/"
    modelos_path = [os.path.join(dir_model, fname) for fname in os.listdir(dir_model) if fname.endswith('.gmm')]

    model_shape = []
    nome_locutor = []
    for fname in modelos_path:
        #print(fname)
        nome_locutor.append(fname.split("/")[-1].split("\\")[-1].split(".")[0])

        with open(fname, 'rb') as f:
            modelo = pickle.load(f)
            model_shape.append(modelo)
            #print(np.array(modelo).shape)

    while True:
        try:
            model_shape = np.array(model_shape[1][1][1])
            break
        except IndexError:
            break


    todos_dados_desc, identificador, total_loc = dados_audio_teste(txt_teste)
    identificador_copiado = identificador.copy()
    piClassifiedData = []

    for iDados_Desc in np.arange(len(todos_dados_desc)):
        apenas_fala = estrairApenas_dados_de_voz(fSamplerate, comprimento_do_quadro_ms, todos_dados_desc[iDados_Desc])

        MFCC_dados_desconhecidos = psf.mfcc(apenas_fala, fSamplerate, numcep=Numero_de_mfcc,
                                            winlen=comprimento_do_quadro_ms / 1000,
                                            winstep=comprimento_de_salto_ms / 1000, nfft=nfft, preemph = Pre_enfase )


        devisao_dados_loc_desconhecido = dados_de_treno_separados(MFCC_dados_desconhecidos, comprimento_do_quadro_ms,
                                                                  Comprimento_dos_dados_de_test_in_s)


        if MFCC_dados_desconhecidos.shape[1] == model_shape.shape[1]:
            piClassifiedData.append(classify_probability(devisao_dados_loc_desconhecido, modelo))

        elif (MFCC_dados_desconhecidos.shape[1] > model_shape.shape[1] or MFCC_dados_desconhecidos.shape[1] < model_shape.shape[1]):

            st.warning(f'**_Numero de MFCC para teste =  {MFCC_dados_desconhecidos.shape[1]} diferente de treno =  {model_shape.shape[1]} sertifique-se!_**')
            break

    Estimado = []
    Locutores= []
    aceitados_rejeitados = []

    for iIndex, iModel in enumerate(piClassifiedData):

        if identificador_copiado[iIndex] not in nome_locutor:
            identificador_copiado[iIndex] = 'ubm'

        if nome_locutor[iModel] != identificador_copiado[iIndex]:
            rejeitado = "Locutor Errado"
            aceitados_rejeitados.append(rejeitado)
            erro += 1

        else:
             aceiatdo = "Locutor Certo"
             aceitados_rejeitados.append(aceiatdo)

        Estimado.append(str(identificador[iIndex]))
        Locutores.append(nome_locutor[iModel])

    #amarelo porque é uma variavel local
    if MFCC_dados_desconhecidos.shape[1] == model_shape.shape[1]:
        accuracy = ((total_loc - erro) / total_loc) * 100

        numero =[]
        for i,_ in enumerate(Locutores,1):
            numero.append(i)
        data = {'Orador estimado' : Estimado ,'Pertence ao locutor' : Locutores, 'ID': numero, 'Estado': aceitados_rejeitados}
        df = pd.DataFrame(data)
        df.set_index('ID', inplace=True)
        st.write("---")

        acura, tota, err = st.columns([2,2,1])
        with acura:
            st.metric(label="ACCURACY", value=f"{accuracy:.2f}%")
        with tota:
            st.metric(label=f"TOTAL LOCUTORES TESTE:", value=f"{int(total_loc)}") #, delta=f"-Errada {erro}",delta_color="normal")
        with err:
            st.metric(label="ERRADO", value=f"{erro}")

        st.write(f'TOTAL LOCUTORES TRENO: {len(nome_locutor)}')
        #st.markdown("")
        st.info(f'**_{nome_locutor}_**')
        st.write("---")
        st.table(df)


        #txt_ficheiro = "registo_grafico.txt"
        arquivo = open(txt_registo_dados, 'r')  # Abra o arquivo (leitura)
        conteudo = arquivo.readlines()

        conteudo.append("-" * 7 + "\n"
                        "TESTE" + "\n" +
                        "-" * 7 + "\n"
                        f"DIRETORIO DO ARQUIVO DE MODELO: {dir_model}" + "\n"
                        f"DIRETORIO DO ARQUIVO DE TESTE: {txt_teste}" + "\n"
                        f"LISTA DE LOCUTORES TESTADO: {nome_locutor}" + "\n"
                        f"TOTAL LOCUTORES TESTE: {int(total_loc)}" + "\n" 
                        f"ERRADO               : {erro}" + "\n"  
                        f"ACCURACY             : {accuracy:.2f} %" + "\n" +
                        "-" * 30 + "\n\n")

        arquivo = open(txt_registo_dados, 'w')
        arquivo.writelines(conteudo)
        arquivo.close()


    return


st.sidebar.title('PARAMETROS:')
Comprimento_dos_dados_de_treno_in_s = st.sidebar.slider('Comprimento dos dados de treno em segundo',1, 100)
Comprimento_dos_dados_de_test_in_s = st.sidebar.slider('Comprimento dos dados de teste em segundo',1, 100)
Numero_de_mfcc = st.sidebar.slider('Numero de mfcc',6,26,value=13)
Numero_de_gausiano_GMM = st.sidebar.slider('Numero de gausiano (GMM)',1, 64)
Iteracao_maxima_GMM = st.sidebar.slider('Iteracao maxima (GMM)',10,200)

st.subheader('Selecione a operacão que deseja executar')
genre = st.radio(
     "",
     ('GRAVAR AUDIO','TESTE DO MODELO TREINADO','TREINAR E GUARDAR O MODELO'))


if genre == 'TESTE DO MODELO TREINADO':
    st.title('TESTE DO MODELO TREINADO')
    txt_teste = st.text_input("Diretorio do arquivo .txt para testar o modelo", placeholder=".txt")
    dir_model = st.text_input("Diretorio do modelo", "\\")
    txt_registo_dados = st.selectbox("Arquivo txt para guardar dados", caminho_arquivo_txt)

    botao_testar = st.button('INICIAR TESTE')

    if botao_testar:
         while True:
            try:
                with st.spinner('**_Processando dados, espere..._**'):
                    testar_model(txt_teste, dir_model)

                break
            except FileNotFoundError:
                st.error('**Erro no diretorio, por favor sertifique.** ')
                break


elif genre == 'TREINAR E GUARDAR O MODELO':
    #st.write("---")
    st.title('TREINAR E GUARDAR O MODELO')
    st.warning("**Antes de iniciar o treino sertique os 'PARAMETROS'**")
    txt_treno = st.text_input('Diretorio do arquivo txt para treinar e construir o modelo', placeholder=".txt")
    dir_model_guardado = st.text_input("Diretorio para guardar o modelo","\\")
    txt_registo_dados = st.selectbox("Arquivo txt para guardar dados de TRENO", caminho_arquivo_txt)

    botao_treino = st.button('INICIAR TREINO')
    if botao_treino:
        while True:
            try:
                tempo, nome_locutor = treinar_model(txt_treno, dir_model_guardado)

                #txt_ficheiro = "registo_grafico.txt"
                arquivo = open(txt_registo_dados, 'r')
                conteudo = arquivo.readlines()
                conteudo.append("\n\n"+
                                "-"*12+"\n"
                                "NOVO TRENO"+"\n" +
                                "-"*156 + "\n"
                                f"DIRETORIO DO ARQUIVO DE TRENO: {txt_treno}" + "\n"
                                f"DITETORIO DO ARQUIVO DO MODELO GUARDADO: {dir_model_guardado}" + "\n\n"
                                f"Lista de locutores treinados: {nome_locutor}" + "\n"
                                f'Comprimento_dos_dados_de_treno_in_s ={Comprimento_dos_dados_de_treno_in_s}' + "\n"
                                f'Numero_de_mfcc ={Numero_de_mfcc}' + "\n"
                                f'Numero_de_gausiano_GMM ={Numero_de_gausiano_GMM}' + "\n"                             
                                f'Iteracao_maxima_GMM ={Iteracao_maxima_GMM}' + "\n"
                                f'Duração treno ={tempo} s' + "\n\n")


                arquivo = open(txt_registo_dados, 'w')
                arquivo.writelines(conteudo)
                arquivo.close()

                break
            except FileNotFoundError:
                st.error('**Erro no diretorio, por favor sertifique-se.** ')
                break


elif genre == 'GRAVAR AUDIO':

    st.title('GRAVAR AUDIO')
    st.info('**Taxa de amostragem** : 16000   **Formato**  : wav  **Numero de canal**  : 1')

    nome_audio = st.text_input("Nome do audio")
    number = st.number_input('Insira a duracão do audio em segundos', min_value= 1)

    botao_test = st.button('Gravar')

    if botao_test :

        if nome_audio == "":
            st.warning("Escolhe um 'nome do ficheiro de audio'.")

        else:
            Caminho = f"Audio_gravado/{nome_audio}.wav"
            fs = 16000
            sd.default.samplerate = fs
            sd.default.channels = 1
            Gravar_Au = sd.rec(int(number * fs))

            with st.spinner('**Gravando...**'):
                sd.wait(number)
            #st.success('Terminado!')

            wavio.write(Caminho, Gravar_Au, fSamplerate, sampwidth=2)
            audio_file = open(Caminho, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='wav')

            time = np.arange(0, len(Gravar_Au)) / fSamplerate
            fig, ax = plt.subplots()
            ax.plot(time,Gravar_Au)
            plt.xlabel("Tempo")
            plt.ylabel("Amplitude")
            st.pyplot(fig)





