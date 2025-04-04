from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        first_player_dict = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, first_player_dict)

        # Creamos un mapeo de antiguos a nuevos track_id
        stable_ids = list(chosen_players)
        stable_positions = [get_center_of_bbox(first_player_dict[pid]) for pid in stable_ids]

        last_known_positions = [first_player_dict.get(pid) for pid in stable_ids]

        disappear_counter = [0 for _ in stable_ids]
        filtered_player_detections = []

        for player_dict in player_detections:
            current_ids = list(player_dict.keys())
            current_positions = [get_center_of_bbox(player_dict[pid]) for pid in current_ids]

            new_player_dict = {}
            for i, stable_pos in enumerate(stable_positions):
                min_distance = float('inf')
                best_id = None
                for j, current_pos in enumerate(current_positions):
                    dist = measure_distance(stable_pos, current_pos)
                    if dist < min_distance:
                        min_distance = dist
                        best_id = current_ids[j]

                if best_id is not None and best_id in player_dict:
                    new_player_dict[i + 1] = player_dict[best_id]  # Usamos ID fijo: Player 1, Player 2
                    last_known_positions[i] = player_dict[best_id]
                    disappear_counter[i] = 0
                else:
                # Si no fue detectado, usamos la última posición conocida
                    disappear_counter[i] += 1   
                    if disappear_counter[i] < 30:
                        if last_known_positions[i] is not None:
                            new_player_dict[i + 1] = last_known_positions[i]

            filtered_player_detections.append(new_player_dict)

        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sorrt the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections


    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0]) if box.id is not None else -1
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            #Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
