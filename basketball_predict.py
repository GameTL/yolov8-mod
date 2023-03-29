from ultralytics import YOLO
import cv2
import time

#Drawing a crosshair
def crosshair(
                x,
                y, 
                length= 100, 
                color=(0, 255, 0), 
                thickness = 5,
                print_coords = False
                    ): 
    # Converting the coordinates to pixel values
    xp = int(x*X_RESOLUTION)
    yp = int(y*Y_RESOLUTION)

    # Drawing the horizontal line
    xpt1 = (xp, yp+length)
    xpt2 = (xp, yp-length)
    cv2.line(frame, xpt1, xpt2, color, thickness)

    # Drawing the vertical line
    ypt1 = (xp+length, yp)
    ypt2 = (xp-length, yp)
    cv2.line(frame, ypt1, ypt2, color, thickness)
    if print_coords:
        print(f"({xp}, {yp})")


start = time.time()

model = YOLO("basketball_train_4/runs/segment/train/weights/best.pt")

#* INIT INPUT SOURCE

# For webcam input:
# cap = cv2.VideoCapture(0)

# For video input:
cap = cv2.VideoCapture("basketball_train_4/4k60_1.mp4")

X_RESOLUTION = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Y_RESOLUTION = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
VIDEO_FPS = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Resolution: {X_RESOLUTION} x {Y_RESOLUTION}, FPS: {VIDEO_FPS}")

CODEC = ['mp4v','HEVC','hvc1']
fourcc = cv2.VideoWriter_fourcc(*CODEC[0])  # Define the codec to use for the video output
out = cv2.VideoWriter('output1.mp4', fourcc, VIDEO_FPS, (X_RESOLUTION, Y_RESOLUTION))  # Create the video writer object

try:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("Error")
            continue

        # Print FPS on the frame
        cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Print FPS in Terminal
        print("fps: " + str(round(1 / (time.time() - start), 2)))

        start = time.time()

        # Predict with Yolov8
        results = model.predict(source=frame, conf=0.5, show=True)[0]
        # if detected object, draw crosshair on the frame
        if results.boxes:
            for i, obj in enumerate(results.boxes):
                x, y, w, h = obj.xywhn.cpu().numpy()[0]
                crosshair(x,y)


        # Push the frame on the screen
        cv2.imshow("Image", frame)     

        # Save the frame to a video file
        out.write(frame)


        if cv2.waitKey(1) == ord("q"):
            out.release()
            cap.release()
except:
    out.release()
    cap.release()


cap.release()
out.release()

cv2.destroyAllWindows()



