import os
import cv2
import mediapipe as mp
import numpy as np

is_images = False
mp_face_mesh = mp.solutions.face_mesh

LOWER_FACE_MASK_POINTS = [
    # left cheekbone down the jaw and back up the right cheekbone
    50, 117, 116, 123, 147, 213, 192, 214,
    234, 93, 132, 58, 172, 136, 150, 149,
    148, 152, 377, 378, 379, 365, 397, 288,
    361, 323, 454, 434, 416, 433, 376, 352,
    346, 347, 280
]

NOSE_SIDE_POINTS = [98, 327]
NOSE_VERTICAL_POINTS = [168, 6, 197, 195, 5, 4, 1, 2]
UPPER_LIP_POINT = 13


def _landmark_to_xy(landmarks, idx, width, height):
    lm = landmarks.landmark[idx]
    return np.array([int(lm.x * width), int(lm.y * height)], dtype=np.int32)


def _build_nose_cutout(landmarks, width, height):
    nose_side_points = np.array(
        [_landmark_to_xy(landmarks, idx, width, height) for idx in NOSE_SIDE_POINTS],
        dtype=np.int32
    )
    nose_vertical_points = np.array(
        [_landmark_to_xy(landmarks, idx, width, height) for idx in NOSE_VERTICAL_POINTS],
        dtype=np.int32
    )
    upper_lip_y = _landmark_to_xy(landmarks, UPPER_LIP_POINT, width, height)[1]

    nose_center_x = int(np.mean(nose_side_points[:, 0]))
    nose_top = int(nose_vertical_points[:, 1].min())
    nose_bottom = int(nose_vertical_points[:, 1].max())
    nose_width = int(nose_side_points[:, 0].max() - nose_side_points[:, 0].min())
    nose_height = max(1, nose_bottom - nose_top)

    # Keep the cutout focused on the nose and away from the lips.
    side = max(18, int(max(nose_width * 1.35, nose_height * 0.9)))
    center_y = nose_top + int(nose_height * 0.45)

    x1 = max(0, nose_center_x - side // 2)
    x2 = min(width, nose_center_x + side // 2)
    y1 = max(0, center_y - side // 2)
    y2 = min(height, center_y + side // 2)

    lip_guard = max(4, int(nose_height * 0.08))
    max_bottom = upper_lip_y - lip_guard
    if y2 > max_bottom:
        shift = y2 - max_bottom
        y1 = max(0, y1 - shift)
        y2 = max(y1 + 1, max_bottom)

    return x1, y1, x2, y2

def generate_mouth_cheek_mask(frame, is_images):

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    with mp_face_mesh.FaceMesh(
        static_image_mode=is_images,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return mask

        landmarks = result.multi_face_landmarks[0]

        pts = np.array(
            [_landmark_to_xy(landmarks, idx, w, h) for idx in LOWER_FACE_MASK_POINTS],
            dtype=np.int32
        )

        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

        x1, y1, x2, y2 = _build_nose_cutout(landmarks, w, h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.dilate(mask, kernel)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)

    return mask


# ---------------------------
# Example
# ---------------------------


if __name__ == "__main__":
    if is_images:
        os.makedirs("outputs/images", exist_ok=True)
        for image in os.listdir("inputs/images"):
            img_path = os.path.join("inputs/images", image)
            img = cv2.imread(img_path)
            mask = generate_mouth_cheek_mask(img, is_images)
            overlay = img.copy()
            overlay[mask > 0] = (0,0,0)

            cv2.imwrite(f"outputs/images/{image}", overlay)
    else:
        os.makedirs("dataset/mask/videos", exist_ok=True)
        os.makedirs("dataset/masked_videos", exist_ok=True)

        for root, dirs, files in os.walk("dataset/videos"):
            for video in files:
                if not video.endswith(".mp4"):
                    continue

                video_path = os.path.join(root, video)

                # preserve folder structure
                rel_path = os.path.relpath(root, "dataset/videos")

                output_path = os.path.join("dataset/masked_videos", rel_path, video)
                mask_output_path = os.path.join("dataset/mask/videos", rel_path, video)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)

                cap = cv2.VideoCapture(video_path)

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 25

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'avc1')

                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # 🔥 mask video (grayscale → still needs 3 channels for codec)
                mask_out = cv2.VideoWriter(mask_output_path, fourcc, fps, (width, height))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    mask = generate_mouth_cheek_mask(frame, is_images)

                    overlay = frame.copy()
                    overlay[mask > 0] = (0,0,0)

                    out.write(overlay)

                    # 🔥 convert mask to 3-channel for video writer
                    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    mask_out.write(mask_3ch)

                cap.release()
                out.release()
                mask_out.release()
