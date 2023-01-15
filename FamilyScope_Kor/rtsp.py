import cv2
import datetime
import os
import numpy as np
# import easyocr


def writeVideo(id,pw,ip,port):
    
    #RTSP를 불러오는 곳
    video_capture = cv2.VideoCapture(f'rtsp://{id}:{pw}@{ip}:{port}/h264Preview_01_sub')
    video_capture.set(3,640)  # 영상 가로길이 설정 640 
    video_capture.set(4,360)  # 영상 세로길이 설정 360
    fps = 7
    # 가로/세로 길이
    streaming_window_width = int(video_capture.get(3))
    streaming_window_height = int(video_capture.get(4))  
    
    currentTime = datetime.datetime.now()
    fileName = str(currentTime.strftime('%Y_%m_%d_%H_%M_%S'))
    path = f'./static/{fileName}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    
    # 비디오 저장
    # cv2.VideoWriter(저장 위치, 코덱, 프레임, (가로, 세로))
    out = cv2.VideoWriter(path, fourcc, fps, (streaming_window_width, streaming_window_height))
    prev_frame = None
    prev_ret = None
    # reader = easyocr.Reader(['en'])

    while True:
        if not prev_ret:
            video_capture = cv2.VideoCapture(f'rtsp://{id}:{pw}@{ip}:{port}/h264Preview_01_sub')
        ret, frame = video_capture.read()
        prev_ret = ret
        
        if ret:
            prev_frame = frame
        cv2.imshow('streaming video', frame)
        out.write(frame)
        
        k = cv2.waitKey(1) & 0xff
        # esc 누르면 종료
        if k == 27:
            break
    video_capture.release()  # cap 객체 해제
    out.release()  # out 객체 해제
    cv2.destroyAllWindows()

if __name__ == "__main__":
    id=''
    pw=''
    ip=''
    port=''
    writeVideo(id,pw,ip,port)
