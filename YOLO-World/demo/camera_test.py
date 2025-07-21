import cv2

# 웹캠 연결
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera")
    exit()

print("Webcam is working. Press 'q' to exit the video feed.")

# 실시간 웹캠 피드
while True:
    ret, frame = camera.read()  # 프레임 읽기
    if not ret:
        print("Error: Failed to capture image")
        break

    cv2.imshow("Webcam Feed", frame)  # OpenCV 창에 출력

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()  # 웹캠 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

Traceback (most recent call last):
  File "camera_test.py", line 19, in <module>
    cv2.imshow("Webcam Feed", frame)  # OpenCV ì°½ìë ¥
cv2.error: OpenCV(4.10.0) /io/opencv/modules/highgui/src/window.cpp:1301: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'


  result = await self.call_function(
  File "/home/device03/anaconda3/envs/yw02/lib/python3.8/site-packages/gradio/blocks.py", line 1179, in call_function
    prediction = await anyio.to_thread.run_sync(
  File "/home/device03/anaconda3/envs/yw02/lib/python3.8/site-packages/anyio/to_thread.py", line 33, in run_sync
    return await get_asynclib().run_sync_in_worker_thread(
  File "/home/device03/anaconda3/envs/yw02/lib/python3.8/site-packages/anyio/_backends/_asyncio.py", line 877, in run_sync_in_worker_thread
    return await future
  File "/home/device03/anaconda3/envs/yw02/lib/python3.8/site-packages/anyio/_backends/_asyncio.py", line 807, in run
    result = context.run(func, *args)
  File "/home/device03/anaconda3/envs/yw02/lib/python3.8/site-packages/gradio/utils.py", line 695, in wrapper
    response = f(*args, **kwargs)
TypeError: detect_from_webcam() got multiple values for argument 'model_runner'




