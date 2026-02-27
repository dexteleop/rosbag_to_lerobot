#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Convert multiple ROS2 bag files to LeRobot dataset format with video support.
Creates a single dataset with multiple episodes from a directory containing multiple ROS2 bag segments.

Usage example:
```bash
# Clear previous cache
rm -rf ~/.cache/huggingface/lerobot/username/dataset_name

# Convert dataset
python scripts/convert_sliced_rosbags.py \
  --input_directory ./data/rosbags/segments \
  --output username/dataset_name \
  --fps 30 \
  --task "task description"
```
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import logging
import subprocess
from collections import defaultdict
from time import time

# ROS2 imports
try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import Pose
    from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
except ImportError as e:
    print(f"ROS2 dependencies not found: {e}")
    sys.exit(1)

# LeRobot imports
try:
    from lerobot.record import sample_frames_from_video
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_packet_writer import VideoPacketBuffer
except ImportError as e:
    print(f"LeRobot dependencies not found: {e}")
    sys.exit(1)

# Constants
STATE_ACTION_DIM = 48

class MultiVideoRosBag2LeRobotConverter:
    """Enhanced converter for multiple ROS2 bags to single LeRobot dataset with multiple episodes."""

    def __init__(self, input_directory: str, output_repo_id: str, fps: int = 30):
        self.input_directory = Path(input_directory)
        self.output_repo_id = output_repo_id
        self.fps = fps
        self.frame_duration = 1.0 / self.fps

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Video topics mapping - all 7 cameras from your robot
        self.video_topics = {
            'left_color': '/left/color/image_raw/ffmpeg',
            'right_color': '/right/color/image_raw/ffmpeg',
            'head_camera': '/xr_video_topic/ffmpeg',
            # 'chest_camera': '/head/color/image_raw/ffmpeg',
        }

        # Initialize LeRobot dataset (will be created once)
        self.dataset = None

        # Episode tracking
        self.current_episode_index = 0

        # Index Pointer
        self.state_cur_idx = 0
        self.obser_cur_idx_di = dict()
        for camera_key in self.video_topics.keys():
            self.obser_cur_idx_di[camera_key] = 0

    def discover_bag_segments(self):
        """Discover all ROS2 bag segments in the input directory."""
        segments = []

        # Look for segment directories or direct bag files
        if self.input_directory.is_dir():
            # Check for segment_* directories
            segment_dirs = sorted(self.input_directory.glob("rosbag_*"))
            if segment_dirs:
                for segment_dir in segment_dirs:
                    # Find .db3 files in each segment
                    db3_files = list(segment_dir.glob("*.db3"))
                    if db3_files:
                        segments.append({
                            'name': segment_dir.name,
                            'path': str(segment_dir),
                            'bag_file': str(db3_files[0]),  # Take first .db3 file
                        })
            else:
                # Look for .db3 files directly in the directory
                db3_files = list(self.input_directory.glob("*.db3"))
                for i, db3_file in enumerate(sorted(db3_files)):
                    segments.append({
                        'name': f"episode_{i:03d}",
                        'path': str(db3_file.parent),
                        'bag_file': str(db3_file)
                    })

        self.logger.info(f"Discovered {len(segments)} bag segments:")
        for segment in segments:
            self.logger.info(f"  - {segment['name']}: {segment['bag_file']}")

        return segments

    def get_camera_resolution(self, camera_key):
        """Get camera resolution - different cameras may have different resolutions."""
        if camera_key == 'head_camera':
            # Head camera is higher resolution
            return (2160, 4320, 3) 
        else:
            return (480, 848, 3)  # Stereo cameras

    def setup_features(self):
        """Setup LeRobot dataset features with 48-dim data + video features."""
        features = {}

        # Complete 48-dimensional feature names
        feature_names = [
            # Joint positions (16)
            "left_joint1_position", "left_joint2_position", "left_joint3_position", "left_joint4_position",
            "left_joint5_position", "left_joint6_position", "left_joint7_position", "left_gripper_position",
            "right_joint1_position", "right_joint2_position", "right_joint3_position", "right_joint4_position",
            "right_joint5_position", "right_joint6_position", "right_joint7_position", "right_gripper_position",
            # Joint velocities (16)
            "left_joint1_velocity", "left_joint2_velocity", "left_joint3_velocity", "left_joint4_velocity",
            "left_joint5_velocity", "left_joint6_velocity", "left_joint7_velocity", "left_gripper_velocity",
            "right_joint1_velocity", "right_joint2_velocity", "right_joint3_velocity", "right_joint4_velocity",
            "right_joint5_velocity", "right_joint6_velocity", "right_joint7_velocity", "right_gripper_velocity",
            # Joint efforts (16)
            "left_joint1_effort", "left_joint2_effort", "left_joint3_effort", "left_joint4_effort",
            "left_joint5_effort", "left_joint6_effort", "left_joint7_effort", "left_gripper_effort",
            "right_joint1_effort", "right_joint2_effort", "right_joint3_effort", "right_joint4_effort",
            "right_joint5_effort", "right_joint6_effort", "right_joint7_effort", "right_gripper_effort"
        ]

        # ACTION features (48-dim)
        features["action"] = {
            "dtype": "float32",
            "shape": (48,),
            "names": {"motors": feature_names}
        }

        # OBSERVATION features (48-dim)
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (48,),
            "names": {"motors": feature_names}
        }

        # NEXT.DONE feature - episode termination marker
        features["next.done"] = {
            "dtype": "bool",
            "shape": (1,),
            "names": None
        }

        # VIDEO features - each camera as separate observation
        for camera_key in self.video_topics.keys():
            height, width, channels = self.get_camera_resolution(camera_key)

            features[f"observation.images.{camera_key}"] = {
                "dtype": "video",
                "shape": (height, width, channels),
                "names": ["height", "width", "channels"],
            }

        return features

    def create_dataset_if_needed(self):
        """Create the LeRobot dataset if not already created."""
        if self.dataset is None:
            features = self.setup_features()

            self.dataset = LeRobotDataset.create(
                repo_id=self.output_repo_id,
                fps=self.fps,
                features=features,
                robot_type="dual_arm_robot",
                use_videos=True
            )
            self.logger.info(f"Created LeRobot dataset: {self.output_repo_id}")

    def read_bag_messages(self, bag_path: str):
        """Read all relevant messages from a single bag."""
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        # Get topic types
        topic_types_dict = {}
        topic_types = reader.get_all_topics_and_types()
        for topic_metadata in topic_types:
            topic_types_dict[topic_metadata.name] = topic_metadata.type

        # Topics we care about
        joint_topics = [
            '/left_arm/joint_cmd', '/right_arm/joint_cmd',
            '/left_gripper/joint_cmd', '/right_gripper/joint_cmd',
            '/left_arm/joint_states', '/right_arm/joint_states',
            '/left_gripper/joint_states', '/right_gripper/joint_states'
        ]

        video_topics = list(self.video_topics.values())
        all_topics = joint_topics + video_topics

        all_messages = defaultdict(list)
        video_packets = defaultdict(list)

        self.logger.info(f"Reading rosbag messages from {bag_path}...")
        message_count = 0

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            message_count += 1

            if message_count % 100000 == 0:
                self.logger.info(f"  Processed {message_count} messages...")

            if topic not in all_topics:
                continue

            msg_type = topic_types_dict.get(topic)
            if not msg_type:
                continue

            try:
                msg_class = get_message(msg_type)
                msg = deserialize_message(data, msg_class)
                timestamp_sec = timestamp / 1e9

                if topic in video_topics:
                    # Handle video packets
                    camera_key = self._get_camera_key_from_topic(topic)
                    if camera_key:
                        height, width, _ = self.get_camera_resolution(camera_key)
                        video_packets[camera_key].append({
                            'data': bytes(msg.data),
                            'timestamp': timestamp_sec,
                            'pts': msg.pts,
                            'width': width,
                            'height': height,
                            'encoding': msg.encoding,
                        })
                else:
                    # Handle joint data
                    all_messages[topic].append((timestamp_sec, msg))

            except Exception as e:
                self.logger.warning(f"Failed to deserialize {topic}: {e}")
                continue

        del reader

        # Sort messages by timestamp
        for topic in all_messages:
            all_messages[topic].sort(key=lambda x: x[0])

        self.logger.info(f"  Read {message_count} total messages")
        return all_messages, video_packets

    def _get_camera_key_from_topic(self, topic):
        """Map ROS topic to camera key."""
        for camera_key, camera_topic in self.video_topics.items():
            if topic == camera_topic:
                return camera_key
        return None
    
    def find_closest_state_message(self, target_time: float, state_messages: list): # state_messages: List[Tuple]
        """Find state message closest to target time."""
        if not state_messages:
            return None, 0

        start_index = self.state_cur_idx

        best_diff = float('inf')
        best_msg = None
        best_idx = 0

        for idx,(timestamp, msg) in enumerate(
            state_messages[start_index:], start=start_index
        ):
            if target_time > timestamp:
                diff = target_time - timestamp
                if diff < best_diff:
                    best_diff = diff
                    best_msg = msg
                    best_idx = idx
                else:
                    pass
            else:
                diff = timestamp - target_time
                if diff < best_diff:
                    best_diff = diff
                    best_msg = msg
                    best_idx = idx
                else:
                    break

        self.state_cur_idx = max(best_idx, self.state_cur_idx)
        return best_msg, best_idx

    def get_closest_video_frame(self, target_time: float, camera_key: str, packets: dict): # packets: Dict[str:List[Dict]]
        """Get the closest video frame for a given timestamp."""
        if not packets or not packets[camera_key]:
            return 0

        start_index = self.obser_cur_idx_di[camera_key]

        best_idx = 0
        best_diff = float('inf')
        
        for idx, packet in enumerate(
            packets[camera_key][start_index:], start=start_index
        ):
            if target_time > packet['timestamp']:
                diff = target_time - packet['timestamp']
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
                else:
                    pass
            else:
                diff = packet['timestamp'] - target_time
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
                else:
                    break
        
        self.obser_cur_idx_di[camera_key] = best_idx

        if best_idx < len(packets[camera_key]):
            return best_idx
        else:
            self.logger.warning(f"Failed to load frame in target time: {target_time}s")
            return 0

    def find_closest_action_message(self, target_time: float, action_messages: list):
        """Find state message closest to target time."""
        if not action_messages:
            return None

        best_msg = None
        best_diff = float('inf')

        for timestamp, msg in action_messages:
            if timestamp > target_time:
                diff = timestamp - target_time
                if diff < best_diff:
                    best_diff = diff
                    best_msg = msg
                else:
                    break
            else:
                continue

        return best_msg

    def extract_joint_data(self, msg: JointState):
        """Extract joint state data."""
        positions = list(msg.position) if msg.position else []
        velocities= list(msg.velocity) if msg.velocity else []
        efforts   = list(msg.effort)   if msg.effort   else []

        # Ensure we have the right number of joints (7-DOF arms)
        max_joints = 7
        for data_list in [positions, velocities, efforts]:
            if len(data_list) > max_joints:
                data_list[:] = data_list[:max_joints]
            while len(data_list) < max_joints:
                data_list.append(0.0)

        return np.array(positions + velocities + efforts, dtype=np.float32)

    def extract_gripper_data(self, msg: JointState):
        """Extract gripper data (single joint)."""
        pos = msg.position[0] if msg.position else 0.0
        vel = msg.velocity[0] if msg.velocity else 0.0
        effort = msg.effort[0] if msg.effort else 0.0
        return np.array([pos, vel, effort], dtype=np.float32)

    def create_frame_at_time(self, target_time: float, all_messages: dict, video_packets: dict, is_last_frame: bool = False):
        """Create a frame with state, action, and video data at the specified time."""
        frame_data = {}

        state_data = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        action_data = np.zeros(STATE_ACTION_DIM, dtype=np.float32)

        state_idx = 0
        left_arm_msg, idx = self.find_closest_state_message(target_time, all_messages.get('/left_arm/joint_states', []))
        if left_arm_msg:
            joint_data = self.extract_joint_data(left_arm_msg)
            state_data[0:7] = joint_data[0:7]
            state_data[16:23] = joint_data[7:14]
            state_data[32:39] = joint_data[14:21]
        state_idx = max(state_idx, idx)

        left_gripper_state_msg, idx = self.find_closest_state_message(target_time, all_messages.get('/left_gripper/joint_states', []))
        if left_gripper_state_msg:
            gripper_data = self.extract_gripper_data(left_gripper_state_msg)
            state_data[7] = gripper_data[0]
            state_data[23] = gripper_data[1]
            state_data[39] = gripper_data[2]
        state_idx = max(state_idx, idx)

        right_arm_msg, idx = self.find_closest_state_message(target_time, all_messages.get('/right_arm/joint_states', []))
        if right_arm_msg:
            joint_data = self.extract_joint_data(right_arm_msg)
            state_data[8:15] = joint_data[0:7]
            state_data[24:31] = joint_data[7:14]
            state_data[40:47] = joint_data[14:21]
        state_idx = max(state_idx, idx)

        right_gripper_state_msg, idx = self.find_closest_state_message(target_time, all_messages.get('/right_gripper/joint_states', []))
        if right_gripper_state_msg:
            gripper_data = self.extract_gripper_data(right_gripper_state_msg)
            state_data[15] = gripper_data[0]
            state_data[31] = gripper_data[1]
            state_data[47] = gripper_data[2]
        state_idx = max(state_idx, idx)

        frame_data["observation.state"] = state_data
        frame_data["next.done"] = np.array([is_last_frame], dtype=bool)
        video_frame_idx_dict = {}

        for camera_key in self.video_topics.keys():
            video_frame_idx = self.get_closest_video_frame(target_time, camera_key, video_packets)
            video_frame_idx_dict[camera_key] = video_frame_idx

            height, width, channels = self.get_camera_resolution(camera_key)
            black_frame = np.zeros((height, width, channels), dtype=np.uint8)
            frame_data[f"observation.images.{camera_key}"] = black_frame
        observation_idx = max(video_frame_idx_dict.values())

        target_time = max(state_idx, observation_idx)
        left_arm_cmd_msg = self.find_closest_action_message(target_time, all_messages.get('/left_arm/joint_cmd', []))
        if left_arm_cmd_msg:
            joint_data = self.extract_joint_data(left_arm_cmd_msg)
            action_data[0:7] = joint_data[0:7]
            action_data[16:23] = joint_data[7:14]
            action_data[32:39] = joint_data[14:21]

        left_gripper_cmd_msg = self.find_closest_action_message(target_time, all_messages.get('/left_gripper/joint_cmd', []))
        if left_gripper_cmd_msg:
            gripper_data = self.extract_gripper_data(left_gripper_cmd_msg)
            action_data[7] = gripper_data[0]
            action_data[23] = gripper_data[1]
            action_data[39] = gripper_data[2]

        right_arm_cmd_msg = self.find_closest_action_message(target_time, all_messages.get('/right_arm/joint_cmd', []))
        if right_arm_cmd_msg:
            joint_data = self.extract_joint_data(right_arm_cmd_msg)
            action_data[8:15] = joint_data[0:7]
            action_data[24:31] = joint_data[7:14]
            action_data[40:47] = joint_data[14:21]

        right_gripper_cmd_msg = self.find_closest_action_message(target_time, all_messages.get('/right_gripper/joint_cmd', []))
        if right_gripper_cmd_msg:
            gripper_data = self.extract_gripper_data(right_gripper_cmd_msg)
            action_data[15] = gripper_data[0]
            action_data[31] = gripper_data[1]
            action_data[47] = gripper_data[2]

        frame_data["action"] = action_data

        return frame_data, video_frame_idx_dict

    def convert_single_bag(self, segment, task_description: str):
        """Convert a single bag to an episode in the dataset."""
        self.logger.info(f"\n=== Processing {segment['name']} ===")

        try:
            # Read messages from bag
            all_messages, video_packets = self.read_bag_messages(segment['path'])

            if not all_messages and not video_packets:
                self.logger.error(f"  No messages found in {segment['name']}!")
                return False

            # Get time range from joint data
            all_timestamps = []
            for topic_messages in all_messages.values():
                all_timestamps.extend([ts for ts, _ in topic_messages])

            if not all_timestamps:
                self.logger.error(f"  No joint data timestamps found in {segment['name']}!")
                return False

            start_time = min(all_timestamps)
            end_time = max(all_timestamps)
            duration = end_time - start_time

            self.logger.info(f"  The episode's time range: {start_time:.3f} to {end_time:.3f} seconds")
            self.logger.info(f"  The episode's duration: {duration:.3f} seconds")

            self.dataset.episode_buffer = self.dataset.create_episode_buffer(self.current_episode_index)

            frame_duration = 1.0 / self.fps
            current_time = start_time
            frame_count = 0
            video_frame_idx = dict()

            while current_time <= end_time:
                next_time = current_time + frame_duration
                is_last_frame = next_time > end_time

                frame_data, video_frame_idx_dict = self.create_frame_at_time(
                    current_time, all_messages, video_packets, is_last_frame
                )
                for camera_name, frame_idx in video_frame_idx_dict.items():
                    if camera_name in video_frame_idx:
                        video_frame_idx[camera_name].append(frame_idx)
                    else:
                        video_frame_idx[camera_name] = [frame_idx]

                if frame_data:
                    try:
                        self.dataset.add_frame(frame_data, task_description, current_time - start_time - self.frame_duration)
                        frame_count += 1

                        if frame_count % 1000 == 0:
                            self.logger.info(f"    Processed {frame_count} frames")

                    except Exception as e:
                        self.logger.error(f"  Failed to add frame at time {current_time}: {e}")
                        break

                current_time += frame_duration

            packet_buffer = VideoPacketBuffer(
                root_dir=self.dataset.root,
                fps=self.dataset.fps,
            )
            for camera_name, frame_idx_list in video_frame_idx.items():
                for idx in frame_idx_list:
                    packet = video_packets[camera_name][idx]
                    packet_buffer.add_packet(
                        camera_name=camera_name,
                        packet_data=packet['data'],
                        width=packet['width'],
                        height=packet['height'],
                        codec=packet['encoding'],
                    )

            for _, packet_list in packet_buffer.packets.items():
                episode_length = len(packet_list)
                break
            packet_buffer.save_episode(
                episode_index= self.dataset.num_episodes,
                dataset_meta = self.dataset.meta,
            )
            packet_buffer.clear()

            for cam_name, _ in video_packets.items():
                video_key = f"observation.images.{cam_name}"

                width, height, _ = self.get_camera_resolution(cam_name)

                video_path = self.dataset.root / self.dataset.meta.get_video_file_path(
                    self.dataset.num_episodes, video_key
                )

                if not video_path.exists():
                    logging.warning(f"Video not found: {video_path}, skipping stats for {cam_name}")
                    continue

                try:
                    sampled_frames = sample_frames_from_video(
                        video_path=video_path,
                        episode_length=episode_length,
                        fps=self.dataset.fps,
                        width=width,
                        height=height,
                    )
                    self.dataset.episode_buffer[video_key] = sampled_frames

                    logging.info(f"Replaced {cam_name} PNG paths with {len(sampled_frames)} sampled frames")
                except Exception as e:
                    logging.error(f"Failed to sample frames from {cam_name}: {e}")

            if frame_count > 0:
                self.dataset.save_episode()
                self.logger.info(f"  ✅ Episode {self.current_episode_index} complete: {frame_count} frames")
                self.current_episode_index += 1
                return True
            else:
                self.logger.error(f"  ❌ No valid frames created for {segment['name']}!")
                return False

        finally:
            subprocess.run(['rm', '-rf', str(self.dataset.root/'images')])

    def convert_all(self, task_description: str = "task description"):
        """Convert all discovered bag segments to a multi-episode dataset."""
        self.logger.info(f"Starting multi-bag conversion: {self.input_directory}")

        segments = self.discover_bag_segments()
        if not segments:
            self.logger.error("No bag segments found!")
            return False

        self.create_dataset_if_needed()

        successful_episodes = 0
        conversion_duration = time()

        for segment in segments:
            if self.convert_single_bag(segment, task_description):
                successful_episodes += 1

                self.state_cur_idx = 0
                for key in self.obser_cur_idx_di.keys():
                    self.obser_cur_idx_di[key] = 0

        conversion_duration = time() - conversion_duration

        total_episodes = len(segments)
        self.logger.info(f"\n🎉 Multi-bag conversion complete!")
        self.logger.info(f"   Dataset: {self.output_repo_id}")
        self.logger.info(f"   Episodes: {successful_episodes}/{total_episodes} successful")
        self.logger.info(f"   Total frames: {self.dataset.num_frames}")
        self.logger.info(f"   Multi-bag conversion duration: {conversion_duration} seconds")
        self.logger.info(f"   Average time to convert one rosbag: {conversion_duration/total_episodes:.4f} seconds")

        return successful_episodes > 0

def main():
    parser = argparse.ArgumentParser(description="Convert multiple ROS2 bags to LeRobot dataset with video support")
    parser.add_argument("--input_directory", default="./data/rosbags",
                       help="Directory containing ROS2 bag segments")
    parser.add_argument("--output", "-o", default="output/dataset",
                       help="Output dataset repo ID")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target FPS for dataset")
    parser.add_argument("--task", default="grab_stuffed_animals",
                       help="Task description")
    args = parser.parse_args()

    if not os.path.exists(args.input_directory):
        print(f"Error: Input directory {args.input_directory} not found!")
        sys.exit(1)

    converter = MultiVideoRosBag2LeRobotConverter(args.input_directory, args.output, args.fps)
    success = converter.convert_all(args.task)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()