#import necessary packages
import vrep # access all the VREP elements
import numpy as np # package for linear algebra
from matplotlib import pyplot as plt #package for plot
import help_pkg #help functions that accompany the project
import amr_loc_template # Import our implemented functions
import keyboard # Need this to break the loop cleanly

#Initiliazing connection with CoppeliaSim (vrep) via RemoteAPI in Python language
clientID = help_pkg.vrep_connection()
#####################################################################################################  

#Create map of the environment
#map_world array contains the position [x_lan, y_lan] of all landmarks. 
#In case of n landmarks the dimension is n x 2 
map_world = help_pkg.map_create(clientID)
#####################################################################################################

#Get useful handles from CoppeliaSim (vrep)
#Get mobile robot handle --> required for pose queries
err_code, robot_handle = vrep.simxGetObjectHandle(clientID,"Pioneer_p3dx", vrep.simx_opmode_blocking)

#Get motors handles --> required for sending commands to the wheel motors
err_code,l_motor_handle = vrep.simxGetObjectHandle(clientID,"Pioneer_p3dx_leftMotor", vrep.simx_opmode_blocking)
err_code,r_motor_handle = vrep.simxGetObjectHandle(clientID,"Pioneer_p3dx_rightMotor", vrep.simx_opmode_blocking)
######################################################################################################

#Get robot's initial pose 
#robot_pose_prev contains the initial [x, y, theta] pose of the robot
#at the end of each iteration robot_pose_prev = robot_pose
robot_pose_prev = help_pkg.get_robot_pose(clientID, robot_handle)
#####################################################################################################

#state_mm is the state of the robot estimated ONLY from the motion (odometry model).
#It is used only for comparive reasons.
#We initialize with the pose given by the simulator:
state_mm = robot_pose_prev
######################################################################################################

#id_landmarks array contains the detected landmarks at each code iteration.
#if m landmarks are visible the dimension will be m x 3.
#Each raw: 1st element is landmark id e.g. 1.0, 2.0, 3.0,...Second and third elements are landmark's  
#range and bearing with respect to the robot
#We initialize as below, but the dimension will be formulated dynamically according to the detected landmarks
#at each cycle. This is handled by function: get_associated_landmarks
id_landmarks = np.array([[0.0, 0.0, 0.0]])
######################################################################################################

#Boolean variable to check if it is the first time we ask the sensor for measurements
#it is required for the proper communication with the sensor in coppeliasim (vrep)
initialCall=True
######################################################################################################

#Initialize the ekf state
#ekf_state contains the estimates of the robot's pose [x, y, theta]
ekf_state = np.array([robot_pose_prev[0], robot_pose_prev[1], robot_pose_prev[2]])

#Initialize the ekf covariance matrix
Sigma = 10.0*np.array([[0.01, 0.0, 0.0], 
                  [0.0, 0.01, 0.0], 
                  [0.0, 0.0, 0.01]])
######################################################################################################

#Odometry noise parameters
# a = [alpha1, alpha2, alpha3, alpha4]
a = np.array([0.05, 0.05, 0.01, 0.01]) # Adjusted a bit for better simulation results
######################################################################################################

# Lists to store trajectories for plotting
traj_real_x = []
traj_real_y = []
traj_mm_x = []
traj_mm_y = []
traj_ekf_x = []
traj_ekf_y = []

print("System Ready. Use Arrow Keys to move. Press 'q' to stop and plot.")

#This is the main loop of the code
while True:
    
    # Check for exit request
    if keyboard.is_pressed('q'):
        # Stop motors before exiting
        help_pkg.set_motor_cmds(clientID, l_motor_handle, r_motor_handle, 0, 0)
        break

    #Get robot's current pose
    #This is the actual robot pose (Ground Truth).
    robot_pose = help_pkg.get_robot_pose(clientID, robot_handle)
    
    #Get measurements from sensor
    initialCall, id_landmarks = help_pkg.read_sensor_data(initialCall, clientID, map_world, robot_pose)
    # print("ID LandMarks", id_landmarks) # Uncomment for debugging
    
    #Call keyboard teleoperation function here
    u, w = amr_loc_template.keyboard_teleop()
    
    #Call servo_controller function here
    wl, wr = amr_loc_template.servo_controller(u, w)
        
    #Set motor commands (rotational velocities)
    help_pkg.set_motor_cmds(clientID, l_motor_handle, r_motor_handle, wl, wr)
    
    #Calculate odometry inputs from current - previous poses and noise a
    # This simulates the "Encoders" giving us noisy relative motion
    odometry_meas = amr_loc_template.get_odometry_from_pose(robot_pose, robot_pose_prev, a)
    
    # --- Motion Model Estimation (Dead Reckoning) ---
    state_mm = amr_loc_template.motion_model(odometry_meas, state_mm)
    
    # --- EKF Estimation ---
    #Estimate robot's pose via EKF algorithm    
    ekf_state, Sigma = amr_loc_template.ekf_algorithm(ekf_state, Sigma, odometry_meas, a, id_landmarks, map_world)
    
    # --- Data Logging ---
    traj_real_x.append(robot_pose[0])
    traj_real_y.append(robot_pose[1])
    
    traj_mm_x.append(state_mm[0])
    traj_mm_y.append(state_mm[1])
    
    traj_ekf_x.append(ekf_state[0])
    traj_ekf_y.append(ekf_state[1])

    #At the end of each iteration we assign the current actual robot pose to the previous one
    robot_pose_prev = robot_pose;
    
print("Generating Plots...")
plt.figure(figsize=(10, 8))
plt.plot(traj_real_x, traj_real_y, label='Real Trajectory', color='blue')
plt.plot(traj_ekf_x, traj_ekf_y, label='EKF Trajectory', color='orange')
plt.plot(traj_mm_x, traj_mm_y, label='MM Trajectory (Odometry)', color='green', linestyle='dashed')

# Plot Landmarks
map_x = map_world[:, 0]
map_y = map_world[:, 1]
plt.scatter(map_x, map_y, marker='s', color='red', label='Landmarks')

plt.title('Real vs EKF vs MM Trajectory')
plt.xlabel('x - axis (m)')
plt.ylabel('y - axis (m)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

print("Done.")