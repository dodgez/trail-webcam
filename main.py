import cv2
from datetime import datetime
import numpy as np
import os
import time


camera_port = 2  # Device to capture video from
fps = 10  # FPS to record/playback at
video_buffer = 3  # Time to pad on both sides of the video
min_contour_area = 2000  # The minimum contour area to trigger recording
output_folder = "./output/"  # Folder to output the videos
show_cam = True  # Whether or not to show the realtime footage


if __name__ == '__main__':
    print("Starting up...")
    cam = cv2.VideoCapture(camera_port)
    frame_buffer = video_buffer * fps  # Buffer the start of the video
    buffer_timer = frame_buffer  # Timer to buffer the end of the video
    first_frame = None  # Comparison frame
    history = []  # Recorded frames
    occupied = False  # Is the scene occupied
    output_file = ""  # Placeholder for the output file path
    print("Started.")
    try:
        while True:
            ret, frame = cam.read()
            if not(ret):
                print("Failed to capture frame")
                break
            np_img = np.array(frame, dtype='uint8')

            gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            if first_frame is None:
                first_frame = blur
                continue

            frame_delta = cv2.absdiff(first_frame, blur)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, hierarchy = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(
                filter(lambda c: cv2.contourArea(c) >= min_contour_area, contours))

            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(np_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(contours):
                buffer_timer = frame_buffer
            elif occupied:
                buffer_timer -= 1

            if len(contours) and not(occupied):
                print("Recording...")
                occupied = True
                output_file = os.path.join(
                    output_folder, f"{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.avi")
            if buffer_timer <= 0 and occupied:
                print("Stopped recording")
                occupied = False
                height, width, layers = frame.shape
                video = cv2.VideoWriter(
                    output_file, 0, fps, (width, height))
                for frame in history:
                    video.write(frame)
                video.release()
                print(f"Wrote to {output_file}")
                history = []

            if occupied or len(history) < frame_buffer:
                history.append(frame)
            else:
                history = history[-frame_buffer:]

            if show_cam:
                cv2.imshow("Cam", np_img)
                if cv2.waitKey(1000//fps) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(1/fps)
    except KeyboardInterrupt:
        pass
    print("Cleaning up...")
    cam.release()
    cv2.destroyAllWindows()
    print("Goodbye.")
