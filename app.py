import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.face_mesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)

# Load image
image_path = "face.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load {image_path}")
    exit()

h, w, _ = image.shape
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

if not results.multi_face_landmarks:
    print("No face detected.")
    exit()

# Extract landmarks
landmarks = results.multi_face_landmarks[0].landmark


def get_point(idx):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])


# Key bone structure indices for MediaPipe Face Mesh
# Jawline corners (left/right gonion) and chin
left_jaw = get_point(172)
right_jaw = get_point(397)
chin = get_point(152)

# Cheekbones (left/right zygion)
left_cheek = get_point(234)
right_cheek = get_point(454)

# Bigonial width (jaw) vs Zygomatic width (cheekbones) ratio estimation
jaw_width = np.linalg.norm(left_jaw - right_jaw)
cheek_width = np.linalg.norm(left_cheek - right_cheek)

# Compute simple geometric proportions for a score approximation
ratio = jaw_width / cheek_width
# Idealized heuristic mapping for demonstration score (PSL style pseudo-metric)
score_base = 5.0 + (ratio * 3.0)
psl_rating = min(max(round(score_base, 1), 1.0), 10.0)

# Output analysis results
print("=== Facial Bone Analysis ===")
print(f"Cheekbone Width: {int(cheek_width)} px")
# Save analysis text to a file
with open("analysis_results.txt", "w") as f:
    f.write("=== Facial Bone Analysis ===\n")
    f.write(f"Cheekbone Width: {int(cheek_width)} px\n")
    f.write(f"Jawline Width: {int(jaw_width)} px\n")
    f.write(f"Jaw-to-Cheek Ratio: {ratio:.2f}\n")
    f.write(f"Estimated PSL Rating: {psl_rating} / 10\n")
# Draw visual feedback on the image
cv2.line(image, tuple(left_jaw.astype(int)), tuple(chin.astype(int)), (0, 255, 0), 2)
cv2.line(
    image, tuple(right_jaw.astype(int)), tuple(chin.astype(int)), (0, 255, 0), 2
)
cv2.line(
    image,
    tuple(left_cheek.astype(int)),
    tuple(right_cheek.astype(int)),
    (255, 0, 0),
    2,
)

cv2.putText(
    image,
    f"Rating: {psl_rating}/10",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2,
)
cv2.imwrite("rated_face.jpg", image)


