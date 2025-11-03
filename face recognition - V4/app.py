import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import dct, idct  # For DCT and IDCT

app = Flask(__name__)

nimgs = 10
imgBackground = cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

metrics = {}  # Dictionary to store performance metrics


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def ridgelet_dct_encrypt(image):
    """Encrypt an image using Ridgelet-DCT algorithm."""
    # Step 1: Apply DCT to the image
    dct_image = dct(dct(image, axis=0), axis=1)
    
    # Step 2: Apply Ridgelet transform (simulated here as a simple transformation)
    # In a real implementation, Ridgelet transform would be applied here.
    encrypted_image = np.log1p(np.abs(dct_image))  # Simulated encryption
    
    return encrypted_image


def ridgelet_dct_decrypt(encrypted_image):
    """Decrypt an image using Ridgelet-DCT algorithm."""
    # Step 1: Reverse the Ridgelet transform (simulated here)
    decrypted_dct = np.expm1(encrypted_image)  # Simulated decryption
    
    # Step 2: Apply Inverse DCT
    decrypted_image = idct(idct(decrypted_dct, axis=0), axis=1)
    
    return decrypted_image


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            # Decrypt the image before processing
            encrypted_image = np.load(f'static/faces/{user}/{imgname}')
            decrypted_image = ridgelet_dct_decrypt(encrypted_image)
            resized_face = cv2.resize(decrypted_image, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)
    labels = np.array(labels)

    # Split data into training and testing sets
    split = int(0.8 * len(faces))
    X_train, X_test = faces[:split], faces[split:]
    y_train, y_test = labels[:split], labels[split:]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

    # Evaluate the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save metrics for display
    metrics['accuracy'] = accuracy
    metrics['confusion_matrix'] = conf_matrix
    metrics['classification_report'] = classification_report(y_test, y_pred, target_names=userlist)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=userlist, yticklabels=userlist)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')
    plt.close()


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/metrics')
def show_metrics():
    """Display model performance metrics."""
    accuracy = metrics.get('accuracy', 'Model not trained yet.')
    classification_report = metrics.get('classification_report', 'Model not trained yet.')
    return render_template(
        'metrics.html',
        accuracy=accuracy,
        classification_report=classification_report,
        confusion_matrix_img='static/confusion_matrix.png',
    )


@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                face = frame[y:y+h, x:x+w]
                # Encrypt the face image before saving
                encrypted_face = ridgelet_dct_encrypt(face)
                np.save(f'{userimagefolder}/{name}', encrypted_face)  # Save as .npy file
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


if __name__ == '__main__':
    app.run(debug=True)