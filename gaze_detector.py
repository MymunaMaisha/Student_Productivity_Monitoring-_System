import mediapipe as mp
import numpy as np
import cv2

mp_face_mesh = mp.solutions.face_mesh

class GazeDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        # How many seconds before flagging as "not looking"
        self.away_threshold_sec = 5.0

    def get_head_pose(self, frame):
        """Returns list of (yaw, pitch) tuples per detected face."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        poses = []
        if not results.multi_face_landmarks:
            return poses

        # 3D model points for head pose estimation
        model_pts = np.array([
            (0.0, 0.0, 0.0),        # nose tip
            (0.0, -330.0, -65.0),   # chin
            (-225.0, 170.0, -135.0),# left eye corner
            (225.0, 170.0, -135.0), # right eye corner
            (-150.0, -150.0, -125.0),# left mouth
            (150.0, -150.0, -125.0) # right mouth
        ], dtype=np.float64)

        focal = w
        cam_matrix = np.array([[focal, 0, w/2],
                                [0, focal, h/2],
                                [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4,1))

        for face_lm in results.multi_face_landmarks:
            lm = face_lm.landmark
            img_pts = np.array([
                [lm[1].x*w,   lm[1].y*h],
                [lm[152].x*w, lm[152].y*h],
                [lm[263].x*w, lm[263].y*h],
                [lm[33].x*w,  lm[33].y*h],
                [lm[287].x*w, lm[287].y*h],
                [lm[57].x*w,  lm[57].y*h],
            ], dtype=np.float64)

            _, rvec, tvec = cv2.solvePnP(model_pts, img_pts, cam_matrix, dist)
            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw, pitch = angles[1], angles[0]
            poses.append((yaw, pitch))

        return poses

    def is_looking_away(self, yaw, pitch):
        # yaw: left/right. pitch: up/down. Thresholds are degrees.
        return abs(yaw) > 25 or pitch < -15 or pitch > 20