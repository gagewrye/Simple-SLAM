# Simple-SLAM

This is a low-cost SLAM algorithm for high-speed, reactive exploration. It is intended to take the output layer of a navigation ML model and provide directional influenced based on the occupancy map. The idea is to create a map while also encouraging the robot to explore less traveled areas without inflicting too much computational overhead. It achieves this by employing a sparse matrix coordinate system to minimize storage and mapping time and directional gradient filters to encourage exploration.
