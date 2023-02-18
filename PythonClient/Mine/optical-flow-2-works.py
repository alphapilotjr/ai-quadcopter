import airsim
import time
import cv2
import numpy as np


class OpticalFlowTracker:
    def __init__(self):
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=20, qualityLevel=0.3, minDistance=10, blockSize=7)
        self.trajectory_len = 40
        self.detect_interval = 5
        self.trajectories = []
        self.frame_idx = 0
        self.prev_gray = None
        self.mask = np.zeros((480, 640), dtype=np.uint8)

    def process_frame(self, frame):
        start = time.time()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame.copy()

        if len(self.trajectories) > 0:
            img0, img1 = self.prev_gray, frame_gray
            p0 = np.float32([trajectory[-1] for trajectory in self.trajectories]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_trajectories = []

            for trajectory, (x, y), good_flag in zip(self.trajectories, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                trajectory.append((x, y))
                if len(trajectory) > self.trajectory_len:
                    del trajectory[0]
                new_trajectories.append(trajectory)
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            self.trajectories = new_trajectories

            cv2.polylines(img, [np.int32(trajectory) for trajectory in self.trajectories], False, (0, 255, 0))
            cv2.putText(img, "track count: %d" % len(self.trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        if self.frame_idx % self.detect_interval == 0:
            self.mask[:] = 255

            for x, y in [np.int32(trajectory[-1]) for trajectory in self.trajectories]:
                cv2.circle(self.mask, (x, y), 5, 0, -1)

            p = cv2.goodFeaturesToTrack(frame_gray, mask=self.mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.trajectories.append([(x, y)])

        self.frame_idx += 1
        self.prev_gray = frame_gray

        end = time.time()
        if end != start:
            fps = 1 / (end - start)
        else:
            fps = 0

        # Show Results
        cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("image", frame)
        cv2.imshow("Optical Flow", img)
        cv2.imshow("Mask", self.mask)


# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Request API control
client.enableApiControl(True)

# Take off and hover for 5 seconds
client.takeoffAsync().join()
time.sleep(5)

# Set the position to move to
position = airsim.Vector3r(10, 10, -10)

tracker = OpticalFlowTracker()

# Move to the position while taking images
for i in range(100):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
    for response in responses:
        frame = cv2.imdecode(airsim.string_to_uint8_array(response.image_data_uint8), cv2.IMREAD_UNCHANGED)

        tracker.process_frame(frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

    client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, 5)

# Land the drone
client.landAsync().join()

# Release API control
client.enableApiControl(False)

# Close OpenCV window
cv2.destroyAllWindows()
