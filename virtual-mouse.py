import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

color_mouse_pointer = (255, 0, 255)



color_line_estetica = (0,0 , 0)
color_punto_estetico = (0, 0, 255)

X_INI = 170
Y_INI = 200
X_FIN = 300 + 760
Y_FIN = 350 + 410

aspect_ratio_screen = (X_FIN - X_INI) / (Y_FIN - Y_INI)
print("aspect_ratio_screen:", aspect_ratio_screen)

X_Y_INI = 100

#Audio 


def calculate_distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    print(np.linalg.norm(p1 - p2))
    return np.linalg.norm(p1 - p2)

def volume_controlV2(x_index,y_index,x_pulgar_p4,y_pulgar_p4
                   ,x_punto_medio,y_punto_medio):
    
    color_vol_increment = (0,255,0)
    color_vol_decrement = (0, 0, 255)
    
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

    volume = cast(interface, POINTER(IAudioEndpointVolume))
    #volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()
    #volume.SetMasterVolumeLevel(-20.0, None)
    minVol = volRange[0]
    maxVol = volRange[1]
    
    
    length = math.hypot(x_index - x_pulgar_p4, y_index - y_pulgar_p4)
    #print("longitud: ", length)
    vol = np.interp(length, [50,300], [minVol,maxVol])
    print("Length: ", length, "VOL: ", vol)
    volume.SetMasterVolumeLevel(vol, None)
    
    if length < 50:
        cv2.circle(output, (x_punto_medio, y_punto_medio), 5, color_vol_decrement, 3)
    if length > 50:
        cv2.circle(output, (x_punto_medio, y_punto_medio), 5, color_vol_increment, 3)
        
def volume_control(hand_landmarks):
    finger_oup = False
    
    color_base = (255, 0, 112)
    color_index = (255, 198, 82)
    
    x_base1 = int(hand_landmarks.landmark[0].x * width)
    y_base1 = int(hand_landmarks.landmark[0].y * height)
    
    x_index = int(hand_landmarks.landmark[8].x * width)
    y_index = int(hand_landmarks.landmark[8].y * height)
    
    x_base2 = int(hand_landmarks.landmark[9].x * width)
    y_base2 = int(hand_landmarks.landmark[9].y * height)
    
    x_medio = int(hand_landmarks.landmark[12].x * width)
    y_medio = int(hand_landmarks.landmark[12].y * height)
    
    x_pulgar_p4 = int(hand_landmarks.landmark[4].x * width)
    y_pulgar_p4 = int(hand_landmarks.landmark[4].y * height)
    
    x_pulgar_p3 = int(hand_landmarks.landmark[3].x * width)
    y_pulgar_p3 = int(hand_landmarks.landmark[3].y * height)
    
        # dedos puntos complementarios de estetica
    x_pulgar_p1 = int(hand_landmarks.landmark[1].x * width)
    y_pulgar_p1 = int(hand_landmarks.landmark[1].y * height)
    
    x_pulgar_p2 = int(hand_landmarks.landmark[2].x * width)
    y_pulgar_p2 = int(hand_landmarks.landmark[2].y * height)
    
    x_punto_medio = (x_pulgar_p4 + x_index) // 2
    y_punto_medio = (y_pulgar_p4 + y_index) // 2
    
    d_base = calculate_distance(x_base1, y_base1, x_base2, y_base2)
    d_medio = calculate_distance(x_base1,y_base1,x_medio,y_medio) 
    
    cv2.circle(output, (x_pulgar_p4, y_pulgar_p4), 5, color_index, 2)
    cv2.circle(output, (x_pulgar_p3, y_pulgar_p3), 5, color_punto_estetico, 2)
    cv2.circle(output, (x_pulgar_p1, y_pulgar_p1), 5, color_punto_estetico, 2)
    cv2.circle(output, (x_pulgar_p2, y_pulgar_p2), 5, color_punto_estetico, 2)
    
    cv2.line(output, (x_pulgar_p4, y_pulgar_p4), (x_pulgar_p3, y_pulgar_p3), color_line_estetica, 2)
    cv2.line(output, (x_pulgar_p3, y_pulgar_p3), (x_pulgar_p2, y_pulgar_p2), color_line_estetica, 2)
    cv2.line(output, (x_pulgar_p1, y_pulgar_p1), (x_base1, y_base1), color_line_estetica, 2)
    cv2.line(output, (x_pulgar_p2, y_pulgar_p2), (x_pulgar_p1, y_pulgar_p1), color_line_estetica, 2) 
    
    if d_medio > d_base:
        finger_oup = True;
        cv2.circle(output, (x_punto_medio, y_punto_medio), 5, color_index, 2)
        cv2.line(output, (x_index, y_index), (x_pulgar_p4, y_pulgar_p4), color_index, 2)
        volume_controlV2(x_index,y_index,x_pulgar_p4,y_pulgar_p4
                   ,x_punto_medio,y_punto_medio)
        
    return finger_oup

def detect_finger_down(hand_landmarks):
    finger_down = False
    color_base = (255, 0, 112)
    color_index = (255, 198, 82)
   
   
    # dedos puntos utilizados para gestion 
    x_base1 = int(hand_landmarks.landmark[0].x * width)
    y_base1 = int(hand_landmarks.landmark[0].y * height)
    
    x_base2 = int(hand_landmarks.landmark[9].x * width)
    y_base2 = int(hand_landmarks.landmark[9].y * height)

    x_index = int(hand_landmarks.landmark[8].x * width)
    y_index = int(hand_landmarks.landmark[8].y * height)
    
    
    

    d_base = calculate_distance(x_base1, y_base1, x_base2, y_base2)
    d_base_index = calculate_distance(x_base1, y_base1, x_index, y_index)

    if d_base_index < d_base:
        finger_down = True
        color_base = (255, 0, 255)
        color_index = (255, 0, 255)
        

    cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
    cv2.circle(output, (x_index, y_index), 5, color_index, 2)
        
    cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 2)
    cv2.line(output, (x_base1, y_base1), (x_index, y_index), color_index, 2)
    # calcula la distancia entre el pulgar e indice
    
        
   
    
    return finger_down



with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    
    pTime = 0
    while True:
        ret, frame = cap.read()
        
        cTime = time.time()       
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        if ret == False:
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)

       
        area_width = width - X_Y_INI * 2
        area_height = int(area_width / aspect_ratio_screen)
        aux_image = np.zeros(frame.shape, np.uint8)
        aux_image = cv2.rectangle(aux_image, (X_Y_INI, X_Y_INI), (X_Y_INI + area_width, X_Y_INI +area_height), (255, 0, 0), -1)
        output = cv2.addWeighted(frame, 1, aux_image, 0.7, 0)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[9].x * width)
                y = int(hand_landmarks.landmark[9].y * height)
                xm = np.interp(x, (X_Y_INI, X_Y_INI + area_width), (X_INI, X_FIN))
                ym = np.interp(y, (X_Y_INI, X_Y_INI + area_height), (Y_INI, Y_FIN))
                pyautogui.moveTo(int(xm), int(ym))
                if volume_control(hand_landmarks):
                     volume_control(hand_landmarks)
                elif volume_control(hand_landmarks) == False:     
                    if detect_finger_down(hand_landmarks):
                        pyautogui.click()
               
                cv2.circle(output, (x, y), 10, color_mouse_pointer, 3)
                cv2.circle(output, (x, y), 5, color_mouse_pointer, -1)
                
    
        #print("fps: ", fps)
        cv2.putText(output, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (255, 0, 0), 2)
        
        cv2.imshow('output', output)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()