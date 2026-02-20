"""
웹캠을 이용한 가상 마우스 제어 프로그램
MediaPipe Hands로 손 제스처를 감지해 마우스 제어.
"""

from collections import deque
from typing import Optional, Tuple
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


class VirtualMouse:
    """가상 마우스 제어 클래스"""

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        frame_margin: float = 0.1,
        smoothing_factor: float = 0.7,
        click_threshold: float = 40.0,
        right_click_threshold: Optional[float] = None,
        click_hold_frames: int = 2,
        fps_smoothing: float = 0.9,
        scroll_sensitivity: float = 3.0,
        scroll_speed: float = 2.5,
    ):
        """
        Args:
            frame_width: 웹캠 프레임 너비
            frame_height: 웹캠 프레임 높이
            screen_width: 모니터 화면 너비 (None이면 자동 감지)
            screen_height: 모니터 화면 높이 (None이면 자동 감지)
            frame_margin: 프레임 가장자리 마진 비율 (0.0 ~ 0.5)
            smoothing_factor: 스무딩 계수 (0.0 ~ 1.0, 높을수록 부드러움)
            click_threshold: 좌클릭(검지-중지) 거리 임계값 (픽셀)
            right_click_threshold: 우클릭(엄지-검지) 거리 임계값 (픽셀)
            click_hold_frames: 클릭으로 판정될 연속 프레임 수
            fps_smoothing: FPS EMA 필터링 계수 (0.0 ~ 1.0)
            scroll_sensitivity: 스크롤 시작 임계값(px)
            scroll_speed: 손 이동량 대비 스크롤 배율
        """
        if screen_width is None or screen_height is None:
            screen_width, screen_height = pyautogui.size()

        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))
        self.screen_width = max(1, int(screen_width))
        self.screen_height = max(1, int(screen_height))

        self.frame_margin = max(0.0, min(0.5, float(frame_margin)))
        margin_x = int(self.frame_width * self.frame_margin)
        margin_y = int(self.frame_height * self.frame_margin)

        self.usable_x_min = margin_x
        self.usable_x_max = self.frame_width - margin_x
        self.usable_y_min = margin_y
        self.usable_y_max = self.frame_height - margin_y

        self.smoothing_factor = max(0.0, min(1.0, float(smoothing_factor)))
        self.prev_x = self.screen_width // 2
        self.prev_y = self.screen_height // 2
        self._move_history_x = deque(maxlen=3)
        self._move_history_y = deque(maxlen=3)

        self.click_threshold = max(5.0, float(click_threshold))
        self.right_click_threshold = (
            float(right_click_threshold)
            if right_click_threshold is not None
            else max(5.0, self.click_threshold * 0.95)
        )
        self.click_hold_frames = max(1, int(click_hold_frames))
        self._left_pinch_count = 0
        self._right_pinch_count = 0
        self.click_cooldown = 0
        self.click_cooldown_time = 10

        self._fps_smoothing = max(0.0, min(1.0, float(fps_smoothing)))
        self._frame_time: Optional[float] = None
        self._fps = 0.0

        self.scroll_sensitivity = max(0.1, float(scroll_sensitivity))
        self.scroll_speed = float(scroll_speed)
        self._scroll_ref_y: Optional[float] = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _normalize_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        x = self._clamp(x, 0.0, 1.0)
        y = self._clamp(y, 0.0, 1.0)
        px = int(x * self.frame_width)
        py = int(y * self.frame_height)

        px = int(self._clamp(px, self.usable_x_min, self.usable_x_max))
        py = int(self._clamp(py, self.usable_y_min, self.usable_y_max))

        usable_width = max(1, self.usable_x_max - self.usable_x_min)
        usable_height = max(1, self.usable_y_max - self.usable_y_min)
        nx = (px - self.usable_x_min) / usable_width
        ny = (py - self.usable_y_min) / usable_height

        sx = int(self._clamp(nx * (self.screen_width - 1), 0, self.screen_width - 1))
        sy = int(self._clamp(ny * (self.screen_height - 1), 0, self.screen_height - 1))

        return sx, sy

    def _smooth_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        smoothed_x = int(self.smoothing_factor * self.prev_x + (1 - self.smoothing_factor) * x)
        smoothed_y = int(self.smoothing_factor * self.prev_y + (1 - self.smoothing_factor) * y)

        self.prev_x = smoothed_x
        self.prev_y = smoothed_y

        self._move_history_x.append(smoothed_x)
        self._move_history_y.append(smoothed_y)
        if len(self._move_history_x) == self._move_history_x.maxlen:
            smoothed_x = int(np.mean(self._move_history_x))
            smoothed_y = int(np.mean(self._move_history_y))

        return smoothed_x, smoothed_y

    def _to_pixel(self, landmark) -> Tuple[float, float]:
        return self._clamp(landmark.x, 0.0, 1.0), self._clamp(landmark.y, 0.0, 1.0)

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        x1, y1 = p1
        x2, y2 = p2
        px1 = int(self._clamp(x1, 0.0, 1.0) * self.frame_width)
        py1 = int(self._clamp(y1, 0.0, 1.0) * self.frame_height)
        px2 = int(self._clamp(x2, 0.0, 1.0) * self.frame_width)
        py2 = int(self._clamp(y2, 0.0, 1.0) * self.frame_height)
        return np.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)

    def _is_finger_up(self, tip, pip) -> bool:
        return tip.y < pip.y

    def _update_fps(self) -> str:
        now = time.perf_counter()
        if self._frame_time is None:
            self._frame_time = now
            return "FPS: --"

        elapsed = max(1e-6, now - self._frame_time)
        self._frame_time = now

        instant = 1.0 / elapsed
        if self._fps == 0.0:
            self._fps = instant
        else:
            self._fps = self._fps_smoothing * self._fps + (1.0 - self._fps_smoothing) * instant

        return f"FPS: {self._fps:.1f}"

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        fps_text = self._update_fps()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if self.click_cooldown > 0:
            self.click_cooldown -= 1

        click_state = "NONE"
        mode_text = "MOVE"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

            index_tip = hand.landmark[8]
            index_pip = hand.landmark[6]
            middle_tip = hand.landmark[12]
            middle_pip = hand.landmark[10]
            thumb_tip = hand.landmark[4]
            thumb_ip = hand.landmark[3]
            ring_tip = hand.landmark[16]
            ring_pip = hand.landmark[14]
            pinky_tip = hand.landmark[20]
            pinky_pip = hand.landmark[18]

            idx = self._to_pixel(index_tip)
            mid = self._to_pixel(middle_tip)
            thm = self._to_pixel(thumb_tip)
            idx_px = int(idx[0] * self.frame_width)
            idx_py = int(idx[1] * self.frame_height)
            mid_px = int(mid[0] * self.frame_width)
            mid_py = int(mid[1] * self.frame_height)
            thm_px = int(thm[0] * self.frame_width)
            thm_py = int(thm[1] * self.frame_height)

            index_up = self._is_finger_up(index_tip, index_pip)
            middle_up = self._is_finger_up(middle_tip, middle_pip)
            ring_up = self._is_finger_up(ring_tip, ring_pip)
            pinky_up = self._is_finger_up(pinky_tip, pinky_pip)

            screen_x, screen_y = self._normalize_coordinates(index_tip.x, index_tip.y)
            smooth_x, smooth_y = self._smooth_coordinates(screen_x, screen_y)

            # 핀치 기반 좌/우클릭
            dist_left = self._distance(idx, mid)
            dist_right = self._distance(thm, idx)

            if dist_right < self.right_click_threshold:
                self._right_pinch_count = min(self._right_pinch_count + 1, self.click_hold_frames)
            else:
                self._right_pinch_count = 0

            if dist_left < self.click_threshold:
                self._left_pinch_count = min(self._left_pinch_count + 1, self.click_hold_frames)
            else:
                self._left_pinch_count = 0

            if self._right_pinch_count >= self.click_hold_frames and self.click_cooldown == 0:
                try:
                    pyautogui.click(button="right")
                    click_state = "RIGHT CLICK"
                    self.click_cooldown = self.click_cooldown_time
                    self._right_pinch_count = 0
                    self._left_pinch_count = 0
                    self._scroll_ref_y = None
                except pyautogui.FailSafeException:
                    pass
            elif self._left_pinch_count >= self.click_hold_frames and self.click_cooldown == 0:
                try:
                    pyautogui.click(button="left")
                    click_state = "LEFT CLICK"
                    self.click_cooldown = self.click_cooldown_time
                    self._left_pinch_count = 0
                    self._right_pinch_count = 0
                    self._scroll_ref_y = None
                except pyautogui.FailSafeException:
                    pass

            # 스크롤: 검지+중지가 모두 위로 펴진 상태
            if index_up and middle_up and not ring_up and not pinky_up:
                mode_text = "SCROLL"
                if self._scroll_ref_y is None:
                    self._scroll_ref_y = (idx[1] + mid[1]) / 2.0 * self.frame_height
                else:
                    y_center = (idx[1] + mid[1]) / 2.0 * self.frame_height
                    delta_y = y_center - self._scroll_ref_y
                    if abs(delta_y) >= self.scroll_sensitivity:
                        scroll_amount = int(-delta_y * self.scroll_speed)
                        if scroll_amount != 0:
                            try:
                                pyautogui.scroll(scroll_amount)
                            except pyautogui.FailSafeException:
                                pass
                            self._scroll_ref_y = y_center
                if click_state == "NONE":
                    # 클릭이 나지 않았다면 포인트 이동 정지
                    pass
                else:
                    # 클릭이 발생해도 그 다음 프레임에서의 이동은 무시
                    pass
            else:
                self._scroll_ref_y = None
                mode_text = "MOVE"
                # 기본 이동: 검지만 사용
                if index_up:
                    try:
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0)
                    except pyautogui.FailSafeException:
                        pass

            cv2.circle(frame, (idx_px, idx_py), 10, (0, 255, 255), -1)
            cv2.circle(frame, (mid_px, mid_py), 10, (255, 0, 255), -1)
            cv2.circle(frame, (thm_px, thm_py), 8, (255, 255, 0), -1)

            cv2.line(
                frame,
                (idx_px, idx_py),
                (mid_px, mid_py),
                (255, 255, 0),
                2,
            )
            cv2.rectangle(
                frame,
                (self.usable_x_min, self.usable_y_min),
                (self.usable_x_max, self.usable_y_max),
                (255, 255, 255),
                2,
            )

            info_text = [
                f"Mode: {mode_text}",
                f"Action: {click_state}",
                f"Dist(left): {dist_left:.1f}px / dist(right): {dist_right:.1f}px",
                f"Cursor: ({smooth_x}, {smooth_y})",
                f"Hold L/R: {self._left_pinch_count}/{self.click_hold_frames} / {self._right_pinch_count}/{self.click_hold_frames}",
                fps_text,
            ]

            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        else:
            # 손 미감지
            self._left_pinch_count = 0
            self._right_pinch_count = 0
            self._scroll_ref_y = None
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                "Mode: WAIT",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                fps_text,
                (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, self.frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return frame

    def release(self):
        self.hands.close()


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    virtual_mouse = VirtualMouse(
        frame_width=actual_width or 640,
        frame_height=actual_height or 480,
        frame_margin=0.1,
        smoothing_factor=0.7,
        click_threshold=40.0,
        right_click_threshold=34.0,
        click_hold_frames=2,
    )

    print("가상 마우스 프로그램이 시작되었습니다.")
    print("한 손 동작: 검지 이동, 검지-중지 핀치=좌클릭, 엄지-검지 핀치=우클릭, 검지+중지 스크롤")
    print("'q' 키를 누르면 종료됩니다.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            processed = virtual_mouse.process_frame(frame)
            cv2.imshow("Virtual Mouse", processed)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")

    finally:
        cap.release()
        virtual_mouse.release()
        cv2.destroyAllWindows()
        print("프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()
