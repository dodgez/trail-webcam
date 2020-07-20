import cv2
from datetime import datetime
import numpy as np
import time


camera_port = 2
frame_buffer = 30
sensitivity = 1.3
output_folder = "./output"  # Folder has to exist and end _without_ a trailing slash


if __name__ == '__main__':
    print("Starting up...")
    cam = cv2.VideoCapture(camera_port)
    cv2.namedWindow("Trail-WebCam")
    history = []
    diffs = []
    recording = False
    bad_frames = 0
    start_time = ""
    print("Started.")
    try:
        while True:
            ret, frame = cam.read()
            if not(ret):
                print("Failed to capture frame")
                break
            np_img = np.array(frame, dtype='uint8')

            if len(history):
                diff = np.sum((np_img - history[-1])**2)
                if len(diffs):
                    # Minimum diff to record
                    bar = sum(diffs)/len(diffs)*sensitivity
                    if not(recording) and diff > bar:  # Start recording (with buffer)
                        recording = True
                        bad_frames = 0
                        print("Recording")
                        start_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
                    elif recording and diff < bar:
                        bad_frames += 1  # Acknowledge no movement
                        if bad_frames > frame_buffer:  # Stop recording
                            print("Done recording")
                            recording = False
                            diffs = []
                            height, width, layers = frame.shape
                            video = cv2.VideoWriter(
                                f"{output_folder}/{start_time}.avi", 0, 10, (width, height))
                            for frame in history:
                                video.write(frame)
                            video.release()
                            print(f"Wrote to {output_folder}/{start_time}.avi")
                            history = []
                if not(recording):
                    diffs.append(diff)
                    if len(history) > frame_buffer:  # Trim unused footage to frame_buffer frames
                        history = history[-frame_buffer:]

            history.append(np_img)

            cv2.imshow("Trail-WebCam", np_img)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    print("Cleaning up...")
    cam.release()
    cv2.destroyAllWindows()
