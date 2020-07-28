import cv2 as cv
# haarcascade file u can download from official repo of opencv
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    print(faces)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)


    cv.imshow("IMAGE", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
