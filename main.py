import cv2
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    resized = cv2.resize(frame, (1200, 950))

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(resized)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(resized, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Frame", resized)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()