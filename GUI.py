import cv2
import numpy as np
import face_recognition
import os
import csv
import pandas as pd
from datetime import datetime
import PIL.Image
import PIL.ImageTk
from tkinter import *
from tkinter import messagebox

window =Tk()
window.geometry('601x601')
window.resizable(width=False, height=False)
window.title("PEC Attendance Portal")
window.configure(background='#D0D3D4')
image=PIL.Image.open("banner1.png")
photo=PIL.ImageTk.PhotoImage(image)
lab=Label(image=photo, bg='#D0D3D4')
lab.pack()

fn=StringVar()
entry_name=Entry(window, textvar=fn)
entry_name.place(x=150, y=257)
ln=StringVar()
entry_id=Entry(window, textvar=ln)
entry_id.place(x=455, y=257)
dn=StringVar()
entry_name_del=Entry(window, textvar=dn)
entry_name_del.place(x=150, y=507)


label2=Label(window, text="New User", fg='#717D7E', bg='#D0D3D4', font=("roboto", 20, "bold")).place(x=20, y=200)
label3=Label(window, text="Enter Name :", fg='black', bg='#D0D3D4', font=("roboto", 15)).place(x=20, y=250)
label4=Label(window, text="Enter Roll Number :", fg='black', bg='#D0D3D4', font=("roboto", 15)).place(x=275, y=252)
label5=Label(window, text="Note : To exit the frame window press 'q'", fg='red', bg='#D0D3D4', font=("roboto", 15)).place(x=20, y=100)

#def v(set):
 #   pass

#status=Label(window, textvariable=v, fg='red', bg='#D0D3D4', font=("roboto", 15, "italic")).place(x=20, y=150)
label6=Label(window, text="Take Attendance", fg='#717D7E', bg='#D0D3D4', font=("roboto", 20, "bold")).place(x=20, y=350)
label7=Label(window, text="Delete a users information", fg='#717D7E', bg='#D0D3D4', font=("roboto", 20, "bold")).place(x=20, y=450)
label8=Label(window, text="Enter Id :", fg='black', bg='#D0D3D4', font=("roboto", 15)).place(x=20, y=500)

def exit_window():
    window.destroy()

button1=Button(window, text="Exit", width=5, fg='#fff', bg='red', relief=RAISED, font=("roboto", 15, "bold"), command=exit_window)
button1.place(x=500, y=550)


def insert_user():
    id = ln.get()
    name = fn.get()
    # if (isNumber(Id) and name.isalpha()):
    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sampleNum = sampleNum + 1
            cv2.imwrite("ImagesAttendance\ " + name + "." + id + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('Webcam', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum > 20:
            break
    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", name + " data is successfully collected")
    # res = "Images Saved for ID : " + Id + " Name : " + name
    row = [id, name]
    with open('StudentDetails.csv', 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

button2=Button(window,text="Submit",width=5,fg='#fff',bg='#27AE60',relief=RAISED,font=("roboto",15,"bold"),command=insert_user)
button2.place(x=20,y=300)

def train_image():
    global org1, org2, encodeListKnown
    path = 'ImagesAttendance'
    images = []
    classNames = []
    org1 = []
    org2 = []
    myList = os.listdir(path)
    # print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    # print(classNames)
    for line in classNames:
        entry = line.split('.')
        org1.append(entry[1])
        org2.append(entry[0])
    # print(org1)
    # print(org2)

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
    encodeListKnown = findEncodings(images)
    print(encodeListKnown)
    # v = 'Training Completed'
    # messagebox.showinfo("Information", encodeListKnown)


button3=Button(window,text="Train Images",fg='#fff',bg='#5DADE2',relief=RAISED,font=("roboto",15,"bold"),command=train_image)
button3.place(x=100,y=300)

def track_user():
    train_image()
    now = datetime.now()
    date = now.strftime('%d-%m-%y')
    def markAttendance(name, id):
        with open('Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                time = now.strftime('%H:%M:%S')
                date = now.strftime('%d-%m-%y')
                f.writelines(f'\n{name},{id},{date},{time}')
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        # img = captureScreen()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = org2[matchIndex].upper()
                id = org1[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                markAttendance(name, id)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", date+" Attendance is Taken")

button4=Button(window,text="Take Attendance",fg='#fff',bg='#E67E22',relief=RAISED,font=("roboto",15,"bold"),command=track_user)
button4.place(x=20,y=400)


def del_user():

    roll_del=int(dn.get())
    src="ImagesAttendance"
    df=pd.read_csv("StudentDetails.csv")
    for roll in df['Id']:
        if(roll == roll_del):
            for image_file_name in os.listdir(src):
                roll_str = str(roll)
                if (roll_str in image_file_name):
                    messagebox.showwarning("Warning!!!", "Deleting the Given user Id info...")
                    os.remove(src + "/" + image_file_name)
                    df.drop(df.loc[df['Id'] == roll_del].index, inplace=True)
                    df.to_csv("StudentDetails.csv", index=False, encoding='utf8')
                    messagebox.showinfo("Success", roll_str + " data is successfully deleted")
                else:
                    messagebox.showerror("Error", "User with given roll number not present")

button6=Button(window,text="Delete User",fg='#fff',bg='#8E44AD',relief=RAISED,font=("roboto",15,"bold"),command=del_user)
button6.place(x=20,y=550)

window.mainloop()
