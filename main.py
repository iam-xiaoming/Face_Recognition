import cv2
import draw
import face_detect
import liveness_detect
import time
import sysinfo
import threading
import os
import register
import queue
import models
import shutil
from evaluate import Evaluate



# ===========================================

DEVICE_NAME = 'mac'












# =========================================================

model, device = models.load_model(device_name=DEVICE_NAME)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

embs, targets = models.loads()
eva = Evaluate(model=model, device=device, embeddings=embs, targets=targets)


if not cap.isOpened():
    print("Không mở được camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

prev_kps = None

cpu_usage = 0
ram_usage = 0

def update_sysinfo():
    global cpu_usage, ram_usage
    while True:
        cpu_usage, ram_usage = sysinfo.sys_info()
        time.sleep(2)

threading.Thread(target=update_sysinfo, daemon=True).start()

def get_options():
    while True:
        print("="*100)
        print("1. Register")
        print("2. Run")
        print("="*100)
        option = int(input())
        if option == 1 or option == 2:
            break
        print("="*60)
    return option

option = get_options()
if option == 1:
    user_name = input("Enter your name: ")
    SAVE_DIR = f"enrollment/{user_name}"
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)
    
NUM_IMAGES = 10
saved = 0
poses = []
POSE_THRESH = 3
MIN_FACE = 150
regis_img = []

save_queue = queue.Queue()

def save_worker():
    while True:
        path, img = save_queue.get()
        if path is None:
            break
        cv2.imwrite(path, img)
        save_queue.task_done()

threading.Thread(target=save_worker, daemon=True).start()

def register_process(frame, face_info):
    global saved, regis_img
    aligned_img, conf, bbox, kps = face_info
    x1, y1, x2, y2 = bbox.astype(int)
    
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    face_crop = frame[y1:y2, x1:x2]
    
    if face_crop is None or face_crop.size == 0:
        return frame

    if saved < NUM_IMAGES:
        frame = draw.draw_text(frame, f"Saved {saved}/{NUM_IMAGES}", (1080, 110), (0,255,0))
        
        if (x2 - x1) < MIN_FACE:
            frame = draw.draw_text(frame, "Face too small", (600, 50), (0, 0, 255))
            return frame

        if register.is_blurry(face_crop):
            frame = draw.draw_text(frame, "Blur!", (600, 50), (0, 0, 255))
            return frame
        yaw, pitch = register.estimate_pose_from_kps(kps)
        for py, pp in poses:
            if abs(yaw - py) < POSE_THRESH and abs(pitch - pp) < POSE_THRESH:
                frame = draw.draw_text(frame, "Pose too similar", (600, 50), (0, 0, 255))
                return frame

        img_path = f"{SAVE_DIR}/img_{saved:03d}.jpg"
        save_queue.put((img_path, aligned_img))

        poses.append((yaw, pitch))
        aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        regis_img.append(aligned_img)
        saved += 1

    else:
        frame = draw.draw_text(frame, "Enrollment Completed!", (900, 110), (0,255,0))
        if user_name:
            embs = []
            for img in regis_img:
                emb = models.get_embedding(model, img, device)
                embs.append(emb)
            emb_mean = models.average_embeddings(embs)
            models.save(user_name, emb_mean)

    return frame

count_failed = 0
prev_img = None
run_images = []
text = "UNK"

def run(frame, face_info, threshold=0.65):
    global count_failed, prev_img, run_images, text
    aligned_img, conf, bbox, kps = face_info
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    face_crop = frame[y1:y2, x1:x2]
    
    if face_crop is None or face_crop.size == 0:
        count_failed += 1
        text = "UNK"
        return (False, frame)
    
    if (x2 - x1) < MIN_FACE:
        frame = draw.draw_text(frame, "Face too small", (600, 50), (0, 0, 255))
        count_failed += 1
        text = "UNK"
        return (False, frame)

    if register.is_blurry(face_crop):
        frame = draw.draw_text(frame, "Blur!", (600, 50), (0, 0, 255))
        count_failed += 1
        text = "UNK"
        return (False, frame)
    
    if count_failed > 5:
        count_failed = 0
        prev_img = None
        run_images = []
        text = "UNK"
        return (False, frame)
            
    aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    if prev_img is not None:
        e1 = models.get_embedding(model, prev_img, device)
        e2 = models.get_embedding(model, aligned_img, device)
        score = models.cosine(e1, e2)
        frame = draw.draw_text(frame, f"Sim: {score:.2f}", (1080, 75), color=(255, 0, 0))
        if score < threshold:
            count_failed += 1
            text = "UNK"
            frame = draw.draw_text(frame, "Not the same person in frame-to-frame", (500, 50), (0, 0, 255))
            return (False, frame)
        
    prev_img = aligned_img
    run_images.append(aligned_img)
    if len(run_images) == 10:
        t0, t1 = eva.evaluate(run_images)
        if t0:
            text = t1
        else:
            text = "UNK"
            count_failed = 0
            prev_img = None
            run_images = []
            return (False, frame)
    return (True, frame)


while True:
    start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame!")
        break
    
    # flip
    frame = cv2.flip(frame, 1)
    
    # get face
    result = face_detect.get_face(frame, threshold=0.8)
    
    if result:
        aligned_img, conf, bbox, kps = result
        x1, y1, x2, y2 = bbox.astype(int)
        frame = draw.draw_bbox(frame, (x1, y1), (x2, y2))
        frame = draw.draw_text(frame, f"Conf: {conf:.2f}", (1080, 25))
        
        is_movement = liveness_detect.is_movement_kps(kps, prev_kps)
        prev_kps = kps
        if is_movement:
            frame = draw.draw_text(frame, "Liveness", (1080, 50))
        else:
            frame = draw.draw_text(frame, "Static", (1080, 50), color=(0, 0, 255))
        
        if option == 1:
            frame = register_process(frame, result)
        elif option == 2:
            is_cap_next_frame, frame = run(frame, result)
            frame = draw.draw_text(frame, f"Count failed: {count_failed}", (1080, 100), color=(0, 0, 255))
            if is_cap_next_frame:
                frame = draw.draw_text(frame, text, (50, 50), (0, 255, 0))
            else:
                frame = draw.draw_text(frame, text, (50, 50), (0, 0, 255))
    else:
        count_failed += 1
        
    end = time.time()
    fps = 1 / (end - start)
    frame = draw.draw_text(frame, f"CPU: {cpu_usage:.1f}", (920, 25))
    frame = draw.draw_text(frame, f"RAM: {ram_usage:.1f}", (920, 50))
    frame = draw.draw_text(frame, "FPS: " + str(int(fps)), (920, 75))
    
    # show
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
