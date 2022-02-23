import cv2
from cv2 import circle
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_pose.Pose(
    static_image_mode=False) as pose:

    while True:
        ret, frame = captura.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        height, width,_ = frame.shape
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = pose.process(frameRGB)

        if resultado.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                frame, resultado.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128,9,250),thickness=2,circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,25,255),thickness=2,circle_radius=2)
            )

        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
captura.release()
cv2.destroyAllWindows()
