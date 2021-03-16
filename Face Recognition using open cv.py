#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import face_recognition as fr
import cv2
from twilio.rest import Client

video_capture = cv2.VideoCapture(0)

bruno_image = fr.load_image_file("bruno.jpg")
bruno_face_encoding = fr.face_encodings(bruno_image)[0]

known_face_encondings = [bruno_face_encoding]
known_face_names = ["rakshan"]

while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "unidentified"
        
        account_sid = "AC0d6fdb2a7b2d5b12276ab231dbf6d814"
        auth_token =  "82e8847ee0f5162fd8ad54990ac7a44b"

        client = Client(account_sid, auth_token)

        client.messages.create(
                  to="+918686186529",
                  from_="+12518621988",
                  body="Hello there! some unknown person entered in house")

        print("unidentified person")


        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
       

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




