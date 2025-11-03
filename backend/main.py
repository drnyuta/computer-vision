from fastapi import FastAPI, File, UploadFile, Query # Импортируем Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import math
from PIL import Image
import io
import base64 

app = FastAPI(title="Clock Face Detector API")

# Allow CORS (for working with React or any frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_image_from_upload(file: UploadFile) -> np.ndarray | None:
    # Reads the image into OpenCV format (BGR)
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def cv_to_base64(img: np.ndarray) -> str:
    # Encode the image in JPEG format
    is_success, buffer = cv2.imencode(".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if is_success:
        # Convert the buffer to a Base64 string
        return base64.b64encode(buffer).decode('utf-8')
    return ""

@app.post("/detect")
async def detect_clock(
    file: UploadFile = File(...), 
    hand_mode: str = Query(None, description="Manual mode for hand detection: 'dark' or 'light'") # Optional hand_mode parameter
):
    # Helper function to return an error with empty visual steps
    def error_response(msg: str, status_code: int = 500):
        empty_visuals = {"face_detection": "", "hands_detection": "", "final_result": ""}
        return JSONResponse({"error": msg, "visual_steps": empty_visuals}, status_code=status_code)

    # Helper function to return an error on early exit (before hand detection)
    def early_exit_response(steps_list: list[str], face_base64_str: str):
        visuals = {"face_detection": face_base64_str, "hands_detection": face_base64_str, "final_result": face_base64_str}
        return {"hour": None, "minute": None, "steps": steps_list, "visual_steps": visuals}
    
    def draw_outlined_line(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thickness: int):
        cv2.line(img, p1, p2, (0, 0, 0), thickness + 4)
        cv2.line(img, p1, p2, color, thickness)

    try:
        img = read_image_from_upload(file)
        if img is None:
            return error_response("Invalid image", 400)

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        center_img = np.array([w / 2.0, h / 2.0])

        detections = []

        # --- Contours ---
        th = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 9)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000 or len(cnt) < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (MA, ma), angle = ellipse
            ratio = MA / ma if ma > 0 else 0
            if ratio > 1.3:
                continue
            dist = np.linalg.norm(center_img - np.array([cx, cy]))
            score = (np.pi * (MA / 2.0) * (ma / 2.0)) / (1.0 + dist)
            detections.append({'type': 'ellipse', 'score': score, 'params': (cx, cy, MA, ma, angle)})

        # --- HoughCircles ---
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT,
                                   dp=1.2, minDist=100,
                                   param1=100, param2=30,
                                   minRadius=0, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles[0, :, :]))
            for (x, y, r) in circles:
                dist = np.linalg.norm(center_img - np.array([x, y]))
                score = (np.pi * (r ** 2)) / (1.0 + dist * 2.0)
                detections.append({'type': 'circle', 'score': score, 'params': (x, y, r)})

        if not detections:
            return {"hour": None, "minute": None, "steps": ["Clock face not found"]}

        best = max(detections, key=lambda d: d['score'])
        
        # Determine the parameters of the best clock face
        if best['type'] == 'circle':
            cx, cy, radius = map(int, best['params'])
            cx_f, cy_f = float(cx), float(cy)
            MA, ma, angle = float(radius * 2), float(radius * 2), 0.0
        else:
            cx_f, cy_f, MA, ma, angle = best['params']
            cx, cy = int(round(cx_f)), int(round(cy_f))
            radius = int(min(MA, ma) / 2)
        
         # --- PHASE 1: Clock Face Detection Visualization ---
        img_face_detection = img.copy()
        
        cv2.circle(img_face_detection, (cx, cy), 5, (0, 0, 255), -1) # Center
        
        if best['type'] == 'circle':
            cv2.circle(img_face_detection, (cx, cy), radius, (0, 255, 0), 3) # Circle
        else:
            cv2.ellipse(img_face_detection, (int(cx_f), int(cy_f)), (int(MA/2), int(ma/2)), angle, 0, 360, (0, 255, 0), 3) # Зеленый эллипс

        face_base64 = cv_to_base64(img_face_detection)

        # --- Hand Detection ---
        gray_masked = cv2.bitwise_and(gray, gray)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(radius * 0.95), 255, -1)
        hub_r = max(8, int(radius * 0.08))
        cv2.circle(mask, (cx, cy), hub_r, 0, -1)

        # Hand Mode Determination (dark/light hands)
        mode_source = "Auto"
        if hand_mode in ["dark", "light"]:
            mode = hand_mode 
            mode_source = "User-defined"
        else:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(face_mask, (cx, cy), int(radius * 0.9), 255, -1)
            cv2.circle(face_mask, (cx, cy), int(radius * 0.3), 0, -1)
            face_brightness = cv2.mean(gray, mask=face_mask)[0]
            mode = "dark" if face_brightness > 127 else "light"

        if mode == "dark":
            gray_processed = cv2.bitwise_not(gray_masked)
        else:
            gray_processed = gray_masked.copy()

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_processed = clahe.apply(gray_processed)
        gray_blur = cv2.GaussianBlur(gray_processed, (5, 5), 0)
        edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
        if lines is None:
            return early_exit_response(["No hands detected", f"Mode: {mode} ({mode_source})"], face_base64)

        hand_candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            d1 = np.hypot(x1 - cx, y1 - cy)
            d2 = np.hypot(x2 - cx, y2 - cy)
            min_dist = min(d1, d2)
            max_dist = max(d1, d2)
            if min_dist > hub_r * 2.0 or max_dist < radius * 0.25:
                continue
            if d1 < d2:
                tip = (x2, y2)
                length = d2
            else:
                tip = (x1, y1)
                length = d1
            dx = tip[0] - cx
            dy = -(tip[1] - cy)
            angle_rad = math.atan2(dy, dx)
            angle_deg = (90 - math.degrees(angle_rad)) % 360
            hand_candidates.append({'tip': tip, 'length': length, 'angle': angle_deg})

        if len(hand_candidates) < 2:
            return early_exit_response(["Not enough hands detected", f"Mode: {mode} ({mode_source})"], face_base64)

        # Select the two longest hands
        hand_candidates.sort(key=lambda x: x['length'], reverse=True)
        hand1, hand2 = hand_candidates[:2]

        # Logic for determining hour/minute hand
        minutes_v1 = (hand1['angle'] / 6.0) % 60
        hours_v1 = (hand2['angle'] / 30.0) % 12
        minutes_v2 = (hand2['angle'] / 6.0) % 60
        hours_v2 = (hand1['angle'] / 30.0) % 12
        
        # Calculate the expected hour hand angle for consistency check  
        expected_hour_angle_v1 = ((hours_v1 + minutes_v1 / 60.0) * 30.0) % 360
        expected_hour_angle_v2 = ((hours_v2 + minutes_v2 / 60.0) * 30.0) % 360
        
        # Error calculation
        error_v1 = min(abs(expected_hour_angle_v1 - hand2['angle']), 360 - abs(expected_hour_angle_v1 - hand2['angle']))
        error_v2 = min(abs(expected_hour_angle_v2 - hand1['angle']), 360 - abs(expected_hour_angle_v2 - hand1['angle']))
        
        length_diff_ratio = abs(hand1['length'] - hand2['length']) / max(hand1['length'], hand2['length'])

        # Apply rules: length, then angle consistency
        if abs(error_v1 - error_v2) > 10:
            if error_v1 < error_v2:
                minute_hand, hour_hand = hand1, hand2
                minutes, hours = minutes_v1, hours_v1
            else:
                minute_hand, hour_hand = hand2, hand1
                minutes, hours = minutes_v2, hours_v2
        elif length_diff_ratio > 0.05: # If the length difference is significant, the longer one is the minute hand
            if hand1['length'] > hand2['length']:
                minute_hand, hour_hand = hand1, hand2
                minutes, hours = minutes_v1, hours_v1
            else:
                minute_hand, hour_hand = hand2, hand1
                minutes, hours = minutes_v2, hours_v2
        else: # If lengths and errors are close, use the best angle consistency
            if error_v1 < error_v2:
                minute_hand, hour_hand = hand1, hand2
                minutes, hours = minutes_v1, hours_v1
            else:
                minute_hand, hour_hand = hand2, hand1
                minutes, hours = minutes_v2, hours_v2

        final_hours = int(hours) if hours != 0 else 12
        final_minutes = int(minutes)

        # --- PHASE 2: Hands Visualization ---
        img_hands_detection = img.copy()
        
        cv2.circle(img_hands_detection, (cx, cy), 5, (0, 0, 255), -1) # center

        # Draw all found line candidates (thinner and gray)
        for hand in hand_candidates:
            tip = hand['tip']
            draw_outlined_line(img_hands_detection, (cx, cy), tip, (150, 150, 150), 2)
        
        # Draw minute hand (long, yellow) with outline
        m_tip = minute_hand['tip']
        draw_outlined_line(img_hands_detection, (cx, cy), m_tip, (0, 255, 255), 3) 
        
        # Draw hour hand (short, red) with outline
        h_tip = hour_hand['tip']
        draw_outlined_line(img_hands_detection, (cx, cy), h_tip, (0, 0, 255), 5) 
        
        hands_base64 = cv_to_base64(img_hands_detection)

         # --- PHASE 3: Final Result Visualization ---
        img_final_result = img_face_detection.copy()

        draw_outlined_line(img_final_result, (cx, cy), m_tip, (0, 255, 255), 3) 
        
        draw_outlined_line(img_final_result, (cx, cy), h_tip, (0, 0, 255), 5)
        
        time_text = f"{final_hours}:{final_minutes:02d}"
        cv2.putText(img_final_result, time_text, (cx - 50, cy - radius - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img_final_result, time_text, (cx - 50, cy - radius - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA) # Тень

        final_base64 = cv_to_base64(img_final_result)
        
        return {
            "hour": final_hours,
            "minute": final_minutes,
            "mode": mode,
            "steps": [
                f"Detected clock center: ({cx},{cy}) radius={radius}",
                f"Mode: {mode} ({mode_source})",
                f"Minute angle: {minute_hand['angle']:.1f}, Hour angle: {hour_hand['angle']:.1f}",
                f"Final Time: {final_hours}:{final_minutes:02d}"
            ],
            "visual_steps": {
                "face_detection": face_base64,
                "hands_detection": hands_base64,
                "final_result": final_base64
            }
        }

    except Exception as e:
        return error_response(f"Internal Server Error: {str(e)}")
