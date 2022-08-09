import numpy as np
import cv2

from utils.realsense_utils import RealSenseInterface

def capture_video():
    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

    state = 0

    # loop runs if capturing has been initialized.
    while(True):
        # reads frames from a camera
        # ret checks return at each frame
        ret, frame = cap.read()

        # # Converts to HSV color space, OCV reads colors as BGR
        # # frame is converted to hsv
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if state == 1:
            # output the frame
            out.write(frame)

        # The original input frame is shown in the window
        cv2.imshow('Original', frame)

        # Wait for key to shift state
        k = cv2.waitKey(1)
        if k & 0xFF == ord('b'):
            break
        elif k & 0xFF == ord('r'):
            state = 1
            print("start recording!")
        elif k & 0xFF== ord('s'):
            state = 2
            print("stop recording!")
        else:
            pass


    # Close the window / Release webcam
    cap.release()

    # After we release our webcam, we also release the output
    out.release()

    # De-allocate any associated memory usage
    cv2.destroyAllWindows()

def capture_realsense():
    rsi = RealSenseInterface()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output2.mp4', fourcc, 20.0, (640, 480))

    state = 0

    while True:
        rgb, d, rgbd = rsi.get_latest_rgbd()

        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if state == 1:
            # output the frame
            out.write(frame)

        # The original input frame is shown in the window
        cv2.imshow('Original', frame)

        # Wait for key to shift state
        k = cv2.waitKey(1)
        if k & 0xFF == ord('b'):
            break
        elif k & 0xFF == ord('r'):
            state = 1
            print("start recording!")
        elif k & 0xFF== ord('s'):
            state = 2
            print("stop recording!")
        else:
            pass

    rsi.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_realsense()
