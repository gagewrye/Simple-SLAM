# Simple-SLAM

This is a low-cost SLAM algorithm for high-speed, reactive exploration. It is intended to take the output layer of a navigation ML model and provide directional influenced based on the occupancy map. The idea is to create a map while also encouraging the robot to explore less traveled areas without inflicting too much computational overhead.

Mobile robots often have limited computing power and must make quick decisions in dynamic environments, such as during search and rescue. Traditional SLAM techniques like Lidar require expensive equipment, high compute resources, and large storage capacities. This algorithm enables a low-power robot to build a map in real time and rapidly explore its environment while adding minimal computational complexity. It achieves this by employing a sparse matrix coordinate system to minimize storage and mapping time.
