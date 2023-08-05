import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pydub import AudioSegment
import webbrowser
import pyttsx3

engine=pyttsx3.init('sapi5')
voices=engine.getProperty('voices')
print(voices[0].id)
engine.setProperty('voice',voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X=sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
            result=np.hstack((result,mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

emotions={
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'anger',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised',
}

#emotions to observe
observed_emotions=['neutral','calm','happy','sad','anger','fearful','disgust','surprised']
#load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob(r"C:\Users\anand\Desktop\anand 4-2\Audio\Actor_01\\03-01-04-01-01-01-01.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        if emotion in observed_emotions:
            print("file name:",file_name,"	","emotion:",emotion)
            if emotion=="happy":
                webbrowser.open("https://youtu.be/09R8_2nJtjg")
            elif emotion=="anger":
                webbrowser.open("https://youtu.be/r6zIGXun57U")
            elif emotion=="fearful":
                webbrowser.open("https://youtu.be/oNb8mqluP44")
            elif emotion=="disgust":
                webbrowser.open("https://youtu.be/YVkUvmDQ3HY")
            elif emotion=="neutral":
                webbrowser.open("https://youtu.be/sK7riqg2mr4")  
            elif emotion=="sad":
                webbrowser.open("https://youtu.be/RgKAFK5djSk")
            elif emotion=="calm":
                webbrowser.open("https://youtu.be/_sdh5h_zkkk")  
            elif emotion=="surprised":
                webbrowser.open("https://youtu.be/d_HlPboLRL8")      
            else:
                webbrowser.open("https://youtu.be/JGwWNGJdvx8")
            speak(emotion)
            continue
        feature =extract_feature(file,mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x),y, test_size = test_size, random_state=9)


#split the dataset
x_train,x_test,y_train,y_test = load_data(test_size=0.25)

#get the shape of training and testing datasets
print((x_train.shape[0],x_test.shape[0]))

#get the no of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#initialize the MPL classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',max_iter=500)

#train the model
model.fit(x_train,y_train)

#predict the test set
y_pred=model.predict(x_test)

print(y_pred)


