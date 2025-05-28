import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import cv2
import time
import csv
import sys
import random
import threading
import math
import numpy as np
import select
import argparse
qos = QoSProfile(depth=10)
qos.reliability = rclpy.qos.ReliabilityPolicy.RELIABLE
qos.durability = rclpy.qos.DurabilityPolicy.VOLATILE

# Global parameters
WHEEL_RADIUS = 0.033  # m = 33 mm
WHEEL_BASE = 0.16  # m = 160 mm
WHEEL_VEL = 3.0  # m/s
CELL_SIZE = 0.3  # m = 300 mm (the dimensions of each square tile)
GRID_SIZE = 5  # the number of cells in the grid
NUM_ORIENTATIONS = 72  # the number of angular bins (higher better with tradeoffs)
MAX_RANGE = (GRID_SIZE * CELL_SIZE) / 2  # Assume origin at center of grid

class RandomMotionLogger(Node):
    def __init__(self, start_traj_id=0):
        super().__init__('random_motion_logger')
        # Parameters
        self.N = 32  # Number of steps = number of images
        self.rate_hz = 2.5  # Frame rate
        self.wheel_vel_min = -WHEEL_VEL  # m/s - reverse
        self.wheel_vel_max = WHEEL_VEL  # m/s - forward
        self.dataset_root = '/home/earl/workspaces/rldr/trajectories'
        self.curr_traj_id = start_traj_id  # Start from user-specified trajectory ID
        self.frame_count = 0
        self.logging = False
        self.levy_beta = 1.5  # Default value for Lévy flight

        # States
        self.left_vel = 0.0
        self.right_vel = 0.0
        self.data = []
        self.initial_pose = None
        self.odom_pose = None
        self.image = None
        self.current_position = (0.0, 0.0)
        self.action_id = None
        self.x = None
        self.y = None
        self.w = None
        self.last_valid_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Example: Blank black image

        # Lock for thread safety
        self.lock = threading.Lock()

        # Subscribers
        self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            qos
        )
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            qos)
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos)
                
        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos)
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/set_pose',
            qos)
        
        TIMER_PERIOD = 1.0
        self.timer = self.create_timer(TIMER_PERIOD / self.rate_hz, self.timer_callback)

        self.wait_for_keypress_thread = threading.Thread(target=self.keypress_listener)
        self.wait_for_keypress_thread.daemon = True
        self.wait_for_keypress_thread.start()

    def keypress_listener(self):
        self.get_logger().info("Waiting for ENTER key to start a new trajectory...")
        if os.name == 'nt':  # Windows
            import msvcrt
            while rclpy.ok():
                try:
                    if msvcrt.kbhit():  # Check if a key is pressed
                        ch = msvcrt.getch().decode('utf-8')
                        if ch == '\r':  # ENTER key on Windows
                            self.get_logger().info("ENTER key detected.")
                            with self.lock:  # Ensure thread-safe access
                                self.reset_odometry()
                                time.sleep(1)  # Delay of 1 second
                                self.start_new_trajectory()
                except Exception as e:
                    self.get_logger().error(f"Exception in keypress_listener: {e}")
                    break
        else:  # Linux
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while rclpy.ok():
                    if select.select([sys.stdin], [], [], 0.1)[0]:  # Wait for input with a timeout
                        ch = sys.stdin.read(1)
                        if ch == '\n':  # ENTER key on Linux
                            self.get_logger().info("ENTER key detected.")
                            with self.lock:  # Ensure thread-safe access
                                self.reset_odometry()
                                time.sleep(1)  # Delay of 1 second
                                self.start_new_trajectory()
            except Exception as e:
                self.get_logger().error(f"Exception in keypress_listener: {e}")
                #break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def joint_states_callback(self, msg):
        try:
            # Ensure the message contains the required joint names
            if not hasattr(msg, 'name') or not hasattr(msg, 'velocity'):
                self.get_logger().warn("Received malformed JointState message: missing 'name' or 'velocity' fields.")
                return

            # Check if the required joints are present
            left_index = msg.name.index('wheel_left_joint')
            right_index = msg.name.index('wheel_right_joint')

            # Ensure the velocity array has the required indices
            if left_index >= len(msg.velocity) or right_index >= len(msg.velocity):
                self.get_logger().warn("Received JointState message with incomplete velocity data.")
                return

            # Update velocities
            self.left_vel = msg.velocity[left_index]
            self.right_vel = msg.velocity[right_index]
        except ValueError as e:
            self.get_logger().warn(f"Joint name missing in JointState message: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in joint_states_callback: {e}")

    # Define the image_callback method
    def image_callback(self, msg):
        """Callback to process incoming image messages."""
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Process the image (e.g., rotate and convert color)
            corrected_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rotated_image = cv2.rotate(corrected_image, cv2.ROTATE_90_CLOCKWISE)

            # Assign the processed image to self.image
            self.image = image #rotated_image
            self.get_logger().debug("Image received and processed.")
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")

    def odom_callback(self, msg):
        try:
            # Ensure the message contains the required pose data
            if not hasattr(msg, 'pose') or not hasattr(msg.pose, 'pose'):
                self.get_logger().warn("Received malformed Odometry message: missing 'pose' field.")
                return

            # Update odometry pose and current position
            self.odom_pose = msg.pose.pose
            self.current_position = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y
            )

            # Debugging information, can be commented out later
            self.get_logger().debug(f"Odometry updated: x={self.current_position[0]}, y={self.current_position[1]}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in odom_callback: {e}")

    def timer_callback(self):
        if not self.logging or self.frame_count >= self.N:
            return
        self.get_logger().info("Timer callback triggered.")

        # Compute velocities
        omega_L, omega_R, _ = self.generate_levy_motion()
        v = (omega_R + omega_L) * WHEEL_RADIUS / 2.0
        omega = (omega_R - omega_L) * WHEEL_RADIUS / WHEEL_BASE

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)

        # Compute relative pose
        if self.initial_pose and self.odom_pose:
            x0 = self.initial_pose.position.x
            y0 = self.initial_pose.position.y
            theta0 = self.quaternion_to_yaw(self.initial_pose.orientation)

            x1 = self.odom_pose.position.x
            y1 = self.odom_pose.position.y
            theta1 = self.quaternion_to_yaw(self.odom_pose.orientation)

            dx = x1 - x0
            dy = y1 - y0
            dtheta = theta1 - theta0

            rel_x = dx * math.cos(-theta0) - dy * math.sin(-theta0)
            rel_y = dx * math.sin(-theta0) + dy * math.cos(-theta0)
            rel_theta = (math.degrees(dtheta) + 360) % 360

            self.x = rel_x
            self.y = rel_y
            self.w = rel_theta

            self.action_id = self.discretize_pose(rel_x, rel_y, rel_theta)
            self.get_logger().info(f"x: {self.x:.2f}, y: {self.y:.2f}, w: {self.w:.2f}, action_id: {self.action_id}")

        # Capture an image
        img_path = os.path.join(self.traj_path, f"frame_{self.frame_count:04d}.jpg")
        timestamp = time.time()

        if self.image is not None:
            corrected_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            rotated_image = cv2.rotate(corrected_image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(img_path, rotated_image)
            self.last_valid_image = rotated_image
        else:
            if hasattr(self, 'last_valid_image') and self.last_valid_image is not None:
                self.get_logger().warn("No image available for this frame. Using the last valid image.")
                cv2.imwrite(img_path, self.last_valid_image)
            else:
                self.get_logger().error("No image available and no valid image to fall back on.")
                return

        # Record the data
        self.data.append({
            'frame': self.frame_count,
            'timestamp': timestamp,
            'left_vel': self.left_vel,
            'right_vel': self.right_vel,
            'img_path': os.path.basename(img_path),
            'rel_x': self.x,
            'rel_y': self.y,
            'rel_theta': self.w,
            'action_id': self.action_id 
        })

        self.frame_count += 1
        self.get_logger().info(f"Logged frame {self.frame_count}/{self.N}")

        if self.frame_count == self.N:
            self.finish_trajectory()

    def levy_flight(self):  # beta can range between 1.5 and 2.0
        """Generate a step length using an inverse power-law (Lévy-like) distribution."""
        u = random.uniform(0.0001, 1)  # avoid zero
        return u ** (-1 / self.levy_beta)

    def generate_levy_motion(self):
        # Lévy-distributed step duration
        duration = min(self.levy_flight(), 10.0)  # cap max duration to avoid excessive turns

        # Pick a base velocity, maybe biased toward moving forward
        base_velocity = random.uniform(0, self.wheel_vel_max)

        # Random offset for turning
        offset = random.uniform(-base_velocity, base_velocity)

        # Assign left and right wheel velocities
        omega_L = base_velocity + offset
        omega_R = base_velocity - offset

        return omega_L, omega_R, duration

    def reset_odometry(self):
        reset_msg = PoseWithCovarianceStamped()
        reset_msg.pose.pose.position.x = 0.0
        reset_msg.pose.pose.position.y = 0.0
        reset_msg.pose.pose.orientation.w = 1.0
        self.reset_pub.publish(reset_msg)
        # Debugging information, can be commented out later
        self.get_logger().info("Odometry reset command sent.")
        
    def start_new_trajectory(self):
        self.traj_path = os.path.join(self.dataset_root, f"traj_{self.curr_traj_id:03d}")
        os.makedirs(self.traj_path, exist_ok=True)
        self.data = []
        self.frame_count = 0
        self.logging = True
        self.initial_pose = self.odom_pose
        
        self.get_logger().info(f"Started trajectory {self.curr_traj_id}")

        # Write fixed start pose
        with open(os.path.join(self.traj_path, 'start_pose.txt'), 'w') as f:
            f.write("0.0 0.0 0.0\n")

    def finish_trajectory(self):
        self.logging = False
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        if self.initial_pose and self.odom_pose:
            x0 = self.initial_pose.position.x
            y0 = self.initial_pose.position.y
            theta0 = self.quaternion_to_yaw(self.initial_pose.orientation)
            
            x1 = self.odom_pose.position.x
            y1 = self.odom_pose.position.y
            theta1 = self.quaternion_to_yaw(self.odom_pose.orientation)
            
            dx = x1 - x0
            dy = y1 - y0
            dtheta = theta1 - theta0
            
            rel_x = dx * math.cos(-theta0) - dy * math.sin(-theta0)
            rel_y = dx * math.sin(-theta0) + dy * math.cos(-theta0)
            rel_theta = (math.degrees(dtheta) + 360) % 360

            with open(os.path.join(self.traj_path, 'final_pose.txt'), 'w') as f:
                f.write(f"{rel_x:.6f} {rel_y:.6f} {rel_theta:.2f}\n")

            self.get_logger().info(f"Final pose: x={rel_x:.2f}, y={rel_y:.2f}, theta={rel_theta:.2f}")
        else:
            self.get_logger().warn("Could not compute final pose: missing odometry.")
            rel_x, rel_y, rel_theta = 0.0, 0.0, 0.0  # fallback

        # Save log.csv
        with open(os.path.join(self.traj_path, 'log.csv'), 'w', newline='') as csvfile:
            fieldnames = ['frame', 'timestamp', 'left_vel', 'right_vel', 'img_path', 'rel_x', 'rel_y', 'rel_theta', 'action_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)


        self.curr_traj_id += 1
        self.get_logger().info(f"Saved trajectory {self.curr_traj_id} to {self.traj_path}")
        print("\n[INFO] Trajectory complete. Please reset the robot to the starting position.")
        print("Press ENTER to start the next trajectory.")

    def quaternion_to_yaw(self, q):
        """Convert a quaternion to a yaw angle (in radians), ensuring the quaternion is normalized."""
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        if norm == 0:
            self.get_logger().warn("Quaternion has zero norm. Returning yaw as 0.")
            return 0.0
        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    '''
    def discretize_pose(self, x, y, theta_deg, grid_size=GRID_SIZE, cell_size=CELL_SIZE, num_orientations=NUM_ORIENTATIONS):
        col = int(x // cell_size)
        row = int(y // cell_size)
        angle_bin = int(round(theta_deg % 360) / (360 / num_orientations)) % num_orientations
        return (row * grid_size + col) * num_orientations + angle_bin
    '''
    def discretize_pose(self, x, y, theta_deg, grid_size=GRID_SIZE, cell_size=CELL_SIZE, num_orientations=NUM_ORIENTATIONS):
        half_grid = grid_size // 2
        col = int((x / cell_size) + half_grid)
        row = int((y / cell_size) + half_grid)
        if col < 0 or col >= grid_size or row < 0 or row >= grid_size:
            return -1
        angle_bin = int(round(theta_deg % 360) / (360 / num_orientations)) % num_orientations
        return (row * grid_size + col) * num_orientations + angle_bin

def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description="Random Motion Logger")
    parser.add_argument(
        '--start_traj_id',
        type=int,
        default=0,
        help="Starting trajectory ID (default: 0)"
    )
    parsed_args = parser.parse_args()

    node = RandomMotionLogger(start_traj_id=parsed_args.start_traj_id)
    node.get_logger().info("Node initialized and spinning...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down due to KeyboardInterrupt...")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
