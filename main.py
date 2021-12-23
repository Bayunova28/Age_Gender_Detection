#import library
import cv2

#define function to get face in frame box
def getFaceBox(face_net, frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB = False)
    face_net.setInput(blob)
    face_detection = face_net.forward()
    bboxes = []
    
    for i in range(face_detection.shape[2]):
        confidence = face_detection[0, 0, i, 2]

        if confidence > 0.7:
            x1 = int(face_detection[0, 0, i, 3] * frame_width)
            y1 = int(face_detection[0, 0, i, 4] * frame_height)
            x2 = int(face_detection[0, 0, i, 5] * frame_width)
            y2 = int(face_detection[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
    return frame, bboxes

#generate model for age, face and gender
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"

age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"

gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

#list function of gender and age
model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2 age)', '(4-6 age)', '(8-12 age)', '(15-20 age)', '(25-32 age)', '(38-43 age)', '(48-53 age)', '(60-100 age)']
gender_list = ['Female', 'Male']

#call function webcam 
webcam = cv2.VideoCapture(0)

pad = 20

#define function to custom the webcam
while True:
    retV, frame = webcam.read()
    frame, bboxes = getFaceBox(face_net, frame)

    for bbox in bboxes:
        face = frame[max(0, bbox[1] - pad) : min(bbox[3] + pad, frame.shape[0] - 1), max(0, bbox[0] - pad) : min(bbox[2] + pad, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), model_mean_values, swapRB = False)
        gender_net.setInput(blob)
        gender_pred = gender_net.forward()
        gender = gender_list[gender_pred[0].argmax()]

        age_net.setInput(blob)
        age_pred = age_net.forward()
        age = age_list[age_pred[0].argmax()]

        label = "{}, {}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA) 

    cv2.imshow("Age-Gender-Detection", frame)
    wk = cv2.waitKey(1)

    if wk == ord('q'):
        break

#define function to close webcam
webcam.release()
cv2.destroyAllWindows()
