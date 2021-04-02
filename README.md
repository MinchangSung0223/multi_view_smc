# Stereo-calibrate

## (option) Stream 
`./stereo-calib.py -s`

## Camera calibration & Stereo calibration

### Camera calibration
`./stereo-calib.py -c`
- Parameter: 
	- Board width(default: 9)
	- Board height(default: 6)
	- Board size(default: 0.0235)
	- Frame height(default: 1080)
	- Total count(default: 13)
	- Selected devices

### Stereo calibration
`./stereo-calib.py -sc -d [SELECTED_CAMERAS]`

- Parameter: 
	- Board width(default: 9)
	- Board height(default: 6)
	- Board size(default: 0.0235)
	- Frame height(default: 1080)
	- Total count(default: 13)
	- Selected devices(NOTE: Dependent on the order)

	ex) `./stereo-calib.py -sc -d 1 0` -> Stereo-calibration from cam 1 to cam 0


# Realsense ROS launch

## How to turn realsense camera to pointcloud map

- Turn on realsense cameras using launch file

	`roslaunch realsense2_camera rs_camera.launch camera:=cam_0 serial_no:=[SERIAL_NUMBER] filters:=spatial,temporal,pointcloud`

	`roslaunch realsense2_camera rs_camera.launch camera:=cam_1 serial_no:=[SERIAL_NUMBER] filters:=spatial,temporal,pointcloud`

- Transform from ROS coordinate to camera coordinate

	`./base_zf.py [BASE_CAMERA]`

	ex) `./base_zf.py cam_1`

- Rviz

	`rviz`

- Set the fixed frame

![](./figure/rviz.png)

- Transform from camera 1 to camera 0

	`./send_transform.py [STEREO_CALIB.xml]`

	ex) ` ./send_transform.py config/trans10.xml`

### Result
- Raw pointcloud

![](./figure/raw.gif)

- Stereo

![](./figure/stereo.gif)

- Stereo + Global registration

![](./figure/ransac.gif)


## Execute all procedures at a time using tmux


- Precondition
  - Install tmux

    `sudo apt install tmux`
  - Set the base camera (recommended: If you will use more than two cameras, you can select a camera located in the middle of the configuration.)
  - Stereo-calibration
    - If the number of cameras is more than three, it should be calibrated between the base camera and another.

- Methods
  1. Open linux terminal
  2. `./multi-view`
  3. `./base_zf.py [BASE_CAMERA]`
  4. `./send_tranform.py [STEREO_CALIB.xml]`

- TODO: 
1. How many do you want to use cameras?
2. Only one command
3. More accurate
# multi_view_smc
