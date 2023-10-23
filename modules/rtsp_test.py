import cv2
import numpy as np

def main():
    # RTSP 스트림 주소를 입력해주세요.
    rtsp_url = "rtsp://testtapo:a123456789!@172.30.1.41/stream1"

    # RTSP 스트림을 가져오기 위한 VideoCapture 객체 생성
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("RTSP 스트림에 접속할 수 없습니다.")
        return

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("스트림으로부터 프레임을 읽을 수 없습니다.")
            break

        # 프레임 디스플레이
        cv2.imshow('RTSP Stream', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 시 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
