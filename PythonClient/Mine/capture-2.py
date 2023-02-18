import airsim
import time
import cv2
import numpy as np

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

# Move to the position while taking images
for i in range(100):
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene)])
    for response in responses:
        png = cv2.imdecode(airsim.string_to_uint8_array(response.image_data_uint8), cv2.IMREAD_UNCHANGED)

        cv2.imshow("image", png)
        cv2.waitKey(1)

    client.moveToPositionAsync(position.x_val, position.y_val, position.z_val, 5).join()

# Land the drone
client.landAsync().join()

# Release API control
client.enableApiControl(False)

# Close OpenCV window
cv2.destroyAllWindows()
