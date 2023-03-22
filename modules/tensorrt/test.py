import cv2
import time

cap = cv2.VideoCapture('./car1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('results.avi',fourcc,fps,(width,height))
fps = 0

t1 = time.time()
while True:
    t1 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    fps = (fps + (1. / (time.time() - t1))) / 2
out.release()
cap.release()
cv2.destroyAllWindows()