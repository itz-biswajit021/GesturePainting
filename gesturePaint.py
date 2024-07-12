import numpy as np
import cv2
import imutils
import time
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

def nothing(x):
    pass

def initialize_paint_canvas(width, height):
    return np.zeros((height, width, 3)) + 255

def initialize_trackbars(window_name):
    cv2.createTrackbar('BLUE', window_name, 0, 255, nothing)
    cv2.createTrackbar('GREEN', window_name, 165, 255, nothing)  # Default value for green in orange
    cv2.createTrackbar('RED', window_name, 255, 255, nothing)  # Default value for red in orange
    cv2.createTrackbar('THICKNESS', window_name, 2, 10, nothing)

def get_trackbar_values(window_name):
    b = cv2.getTrackbarPos('BLUE', window_name)
    g = cv2.getTrackbarPos('GREEN', window_name)
    r = cv2.getTrackbarPos('RED', window_name)
    thickness = cv2.getTrackbarPos('THICKNESS', window_name)
    return (b, g, r, thickness)

def process_frame(frame, greenLower, greenUpper):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

def find_contours(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(cnts)

def draw_on_paint_canvas(paint, drawing_history):
    for segment in drawing_history:
        pts, color, thickness = segment
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(paint, pts[i - 1], pts[i], color, thickness)

def draw_on_frame(frame, pts, color, thickness):
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(frame, pts[i - 1], pts[i], color, thickness)

def start_application():
    paint = initialize_paint_canvas(640, 480)
    greenLower = (40, 40, 40)
    greenUpper = (80, 255, 255)
    pts = []
    drawing_history = []
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    pen_down = False
    once = False

    cv2.namedWindow('Frame')
    initialize_trackbars('Frame')

    root = Tk()
    root.withdraw()  # Hide the root window

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        b, g, r, thickness = get_trackbar_values('Frame')
        color = (b, g, r)
        mask = process_frame(frame, greenLower, greenUpper)
        cnts = find_contours(mask)
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if cv2.waitKey(1) & 0xFF == ord(' '):
                pen_down = True
            if cv2.waitKey(1) & 0xFF == ord('h'):
                pen_down = False
                once = True

            if pen_down:
                pts.append(center)
            if not pen_down and once:
                drawing_history.append((pts.copy(), color, thickness))
                pts = []
                once = False

            if not pen_down:
                pts.append(None)

            paint = initialize_paint_canvas(640, 480)
            draw_on_paint_canvas(paint, drawing_history)
            if pen_down:
                draw_on_frame(frame, pts, color, thickness)

            # Draw a rectangle on the paint canvas to indicate the current color
            cv2.rectangle(paint, (0, 0), (100, 50), color, -1)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                pts.clear()
                drawing_history.clear()
                paint = initialize_paint_canvas(640, 480)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                         filetypes=[("PNG files", "*.png")],
                                                         title="Save Drawing")
                if file_path:
                    cv2.imwrite(file_path, paint)
                    print(f"Saved drawing as '{file_path}'")

            if cv2.waitKey(1) & 0xFF == ord('o'):
                file_path = filedialog.askopenfilename(defaultextension=".png",
                                                       filetypes=[("PNG files", "*.png")],
                                                       title="Open Drawing")
                if file_path:
                    loaded_paint = cv2.imread(file_path)
                    if loaded_paint is not None:
                        paint = loaded_paint
                        drawing_history.clear()
                        pts.clear()
                        print(f"Loaded drawing from '{file_path}'")

        cv2.imshow("Frame", frame)
        cv2.imshow('Paint', paint)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

def main():
    root = Tk()
    root.title("Paint Application")

    def on_start():
        root.destroy()
        start_application()

    def on_exit():
        root.destroy()

    start_button = Button(root, text="Start Drawing", command=on_start)
    exit_button = Button(root, text="Exit", command=on_exit)
    
    start_button.pack(pady=20)
    exit_button.pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
