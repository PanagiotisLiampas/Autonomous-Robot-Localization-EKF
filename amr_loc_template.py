import keyboard
import math
import numpy as np
from numpy.linalg import inv 
import help_pkg

# Global variables for teleop memory
current_u = 0.0
current_w = 0.0

def keyboard_teleop():
    
    global current_u, current_w
    
    step_u = 0.05
    step_w = 0.05
    max_u = 1.0
    max_w = 1.0

    # Έλεγχος πλήκτρων
    if keyboard.is_pressed('up'):
        current_u = 0.5 # Σταθερή ταχύτητα όπως ζητείται ή αυξανόμενη
    elif keyboard.is_pressed('down'):
        current_u = -0.5
    else:
        current_u = 0.0
        
    if keyboard.is_pressed('left'):
        current_w = 0.3
    elif keyboard.is_pressed('right'):
        current_w = -0.3
    else:
        current_w = 0.0
        
    # Exit condition for the main loop (optional helper)
    if keyboard.is_pressed('q'):
        print("Quitting...")

    return current_u, current_w


def servo_controller(u, w):
  
    R = 0.111  # Ακτίνα τροχών
    L = 0.381  # Απόσταση μεταξύ τροχών (άξονας)
    
    # Εξισώσεις διαφορικής οδήγησης
    wr = (2*u + w*L) / (2*R)
    wl = (2*u - w*L) / (2*R)
    
    return wl, wr

def get_odometry_from_pose(pose, pose_prev, a):
    
    x, y, theta = pose
    x_prev, y_prev, theta_prev = pose_prev
    
    # Υπολογισμός πραγματικών (χωρίς θόρυβο) μεταβολών
    delta_rot1 = math.atan2(y - y_prev, x - x_prev) - theta_prev
    delta_trans = math.sqrt((x - x_prev)**2 + (y - y_prev)**2)
    delta_rot2 = theta - theta_prev - delta_rot1
    
    # Κανονικοποίηση γωνιών στο [-pi, pi]
    delta_rot1 = math.atan2(math.sin(delta_rot1), math.cos(delta_rot1))
    delta_rot2 = math.atan2(math.sin(delta_rot2), math.cos(delta_rot2))
    
    # Παράμετροι θορύβου (alphas)
    a1, a2, a3, a4 = a
    
    # Υπολογισμός θορύβου
    # Χρησιμοποιούμε τη sample_normal_twelve από το help_pkg
    noise_rot1 = help_pkg.sample_normal_twelve(0, a1 * delta_rot1**2 + a2 * delta_trans**2)
    noise_trans = help_pkg.sample_normal_twelve(0, a3 * delta_trans**2 + a4 * delta_rot1**2 + a4 * delta_rot2**2)
    noise_rot2 = help_pkg.sample_normal_twelve(0, a1 * delta_rot2**2 + a2 * delta_trans**2)
    
    # Προσθήκη θορύβου στις μετρήσεις
    delta_hat_r1 = delta_rot1 - noise_rot1
    delta_hat_t = delta_trans - noise_trans
    delta_hat_r2 = delta_rot2 - noise_rot2
    
    return np.array([delta_hat_r1, delta_hat_t, delta_hat_r2])

def motion_model(odometry, robot_pose):
    
    d_rot1, d_trans, d_rot2 = odometry
    x, y, theta = robot_pose
    
    x_new = x + d_trans * math.cos(theta + d_rot1)
    y_new = y + d_trans * math.sin(theta + d_rot1)
    theta_new = theta + d_rot1 + d_rot2
    
    # Κανονικοποίηση γωνίας
    theta_new = math.atan2(math.sin(theta_new), math.cos(theta_new))
    
    return np.array([x_new, y_new, theta_new])

def ekf_algorithm(ekf_state, Sigma, robot_odometry, a, id_landmarks, map_world):
    
    # 1. Πρόβλεψη κατάστασης (State Prediction)
    mu_bar = motion_model(robot_odometry, ekf_state)
    
    # Ανάκτηση τιμών για ευκολία
    x = ekf_state[0]
    y = ekf_state[1]
    theta = ekf_state[2]
    
    d_rot1 = robot_odometry[0]
    d_trans = robot_odometry[1]
    d_rot2 = robot_odometry[2]
    
    # 2. Υπολογισμός Ιακωβιανού Gt (Παράγωγος ως προς το state)
    # Προσοχή: Χρησιμοποιούμε το theta της προηγούμενης κατάστασης
    Gs = np.array([
        [1, 0, -d_trans * math.sin(theta + d_rot1)],
        [0, 1,  d_trans * math.cos(theta + d_rot1)],
        [0, 0, 1]
    ])
    
    # 3. Υπολογισμός Πίνακα συνδιακύμανσης θορύβου κίνησης (Motion Noise Covariance)
    # Υπολογισμός πινάκων V (ως προς το control input) και M (θόρυβος στο control space)
    # Μια πιο άμεση προσέγγιση για τον πίνακα Rt (state space noise) βασισμένη στο βιβλίο Thrun:
    
    # Πίνακας Vt (Παράγωγος ως προς το control parameters: rot1, trans, rot2)
    # x' = x + trans * cos(theta + rot1)
    # y' = y + trans * sin(theta + rot1)
    # th' = theta + rot1 + rot2
    
    sin_term = math.sin(theta + d_rot1)
    cos_term = math.cos(theta + d_rot1)
    
    Vt = np.array([
        [-d_trans * sin_term, cos_term, 0],
        [ d_trans * cos_term, sin_term, 0],
        [1,                   0,        1]
    ])
    
    # Πίνακας Mt (Covariance of control noise)
    a1, a2, a3, a4 = a
    Mt = np.array([
        [a1*d_rot1**2 + a2*d_trans**2, 0, 0],
        [0, a3*d_trans**2 + a4*d_rot1**2 + a4*d_rot2**2, 0],
        [0, 0, a1*d_rot2**2 + a2*d_trans**2]
    ])
    
    # Rt = Vt * Mt * Vt.T
    Rt = Vt @ Mt @ Vt.T
    
    # 4. Πρόβλεψη Συνδιακύμανσης (Covariance Prediction)
    Sigma_bar = Gs @ Sigma @ Gs.T + Rt
    
    # Για κάθε ορόσημο που βλέπουμε
    N = len(id_landmarks)
    
    # Αν δεν βλέπουμε ορόσημα, κρατάμε την πρόβλεψη
    if N == 0:
        return mu_bar, Sigma_bar

    # Παράμετροι θορύβου μέτρησης (Sensor Noise) Q
    # Θεωρούμε τυπικές τιμές αν δεν δίνονται: π.χ. sigma_range = 0.1m, sigma_bearing = 0.05rad
    Q = np.array([
        [0.05, 0.0],
        [0.0, 0.05]
    ])

    for i in range(N):
        lid = int(id_landmarks[i, 0]) # ID του οροσήμου
        z_range = id_landmarks[i, 1]  # Μετρηθείσα απόσταση
        z_bearing = id_landmarks[i, 2]# Μετρηθείσα γωνία
        z = np.array([z_range, z_bearing])
        
        # Θέση του οροσήμου από τον χάρτη
        m_x = map_world[lid, 0]
        m_y = map_world[lid, 1]
        
        # Αναμενόμενη μέτρηση (Measurement Prediction) βάσει της θέσης πρόβλεψης (mu_bar)
        q = (m_x - mu_bar[0])**2 + (m_y - mu_bar[1])**2
        z_hat_range = math.sqrt(q)
        z_hat_bearing = math.atan2(m_y - mu_bar[1], m_x - mu_bar[0]) - mu_bar[2]
        
        # Κανονικοποίηση bearing
        z_hat_bearing = math.atan2(math.sin(z_hat_bearing), math.cos(z_hat_bearing))
        z_hat = np.array([z_hat_range, z_hat_bearing])
        
        # Υπολογισμός Ιακωβιανού Ht (Measurement Jacobian)
        # Ht = [[ -(mx - x)/sqrt(q), -(my - y)/sqrt(q), 0 ],
        #       [  (my - y)/q,       -(mx - x)/q,      -1 ]]
        
        dx = m_x - mu_bar[0]
        dy = m_y - mu_bar[1]
        
        Ht = np.array([
            [-dx/math.sqrt(q), -dy/math.sqrt(q), 0],
            [ dy/q,           -dx/q,            -1]
        ])
        
        # Υπολογισμός Kalman Gain (Kt)
        # S = H * Sigma * H.T + Q
        S = Ht @ Sigma_bar @ Ht.T + Q
        K = Sigma_bar @ Ht.T @ inv(S)
        
        # Ανανέωση Κατάστασης (State Update)
        # mu = mu_bar + K * (z - z_hat)
        innovation = z - z_hat
        # Κανονικοποίηση της διαφοράς γωνίας στο innovation
        innovation[1] = math.atan2(math.sin(innovation[1]), math.cos(innovation[1]))
        
        mu_bar = mu_bar + K @ innovation
        
        # Κανονικοποίηση γωνίας προσανατολισμού στο mu
        mu_bar[2] = math.atan2(math.sin(mu_bar[2]), math.cos(mu_bar[2]))
        
        # Ανανέωση Συνδιακύμανσης (Covariance Update)
        # Sigma = (I - K * H) * Sigma_bar
        I = np.eye(3)
        Sigma_bar = (I - K @ Ht) @ Sigma_bar

    return mu_bar, Sigma_bar