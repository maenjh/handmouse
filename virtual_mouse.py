"""
웹캠을 이용한 가상 마우스 제어 프로그램
MediaPipe Hands를 사용하여 손 제스처로 마우스를 제어합니다.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from typing import Tuple, Optional


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
        click_threshold: float = 40.0
    ):
        """
        Args:
            frame_width: 웹캠 프레임 너비
            frame_height: 웹캠 프레임 높이
            screen_width: 모니터 화면 너비 (None이면 자동 감지)
            screen_height: 모니터 화면 높이 (None이면 자동 감지)
            frame_margin: 프레임 가장자리 마진 비율 (0.0 ~ 0.5)
            smoothing_factor: 스무딩 계수 (0.0 ~ 1.0, 높을수록 부드러움)
            click_threshold: 클릭 감지 임계값 (픽셀 단위)
        """
        # 화면 해상도 설정
        if screen_width is None or screen_height is None:
            screen_width, screen_height = pyautogui.size()
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 프레임 마진 설정 (가장자리 인식률 저하 방지)
        self.frame_margin = frame_margin
        margin_x = int(frame_width * frame_margin)
        margin_y = int(frame_height * frame_margin)
        
        # 실제 사용 가능한 영역
        self.usable_x_min = margin_x
        self.usable_x_max = frame_width - margin_x
        self.usable_y_min = margin_y
        self.usable_y_max = frame_height - margin_y
        
        # 스무딩 설정
        self.smoothing_factor = smoothing_factor
        self.prev_x = screen_width // 2
        self.prev_y = screen_height // 2
        
        # 클릭 감지 설정
        self.click_threshold = click_threshold
        self.click_cooldown = 0
        self.click_cooldown_time = 10  # 프레임 단위
        
        # MediaPipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # PyAutoGUI 안전 설정
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01
    
    def _normalize_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        """
        웹캠 좌표를 화면 좌표로 변환
        
        Args:
            x: 웹캠 프레임 내 x 좌표 (0.0 ~ 1.0)
            y: 웹캠 프레임 내 y 좌표 (0.0 ~ 1.0)
        
        Returns:
            (screen_x, screen_y): 화면 좌표
        """
        # MediaPipe 좌표를 픽셀 좌표로 변환
        pixel_x = int(x * self.frame_width)
        pixel_y = int(y * self.frame_height)
        
        # 사용 가능한 영역으로 제한
        pixel_x = max(self.usable_x_min, min(self.usable_x_max, pixel_x))
        pixel_y = max(self.usable_y_min, min(self.usable_y_max, pixel_y))
        
        # 사용 가능한 영역을 화면 전체로 매핑
        normalized_x = (pixel_x - self.usable_x_min) / (self.usable_x_max - self.usable_x_min)
        normalized_y = (pixel_y - self.usable_y_min) / (self.usable_y_max - self.usable_y_min)
        
        # 화면 좌표로 변환
        screen_x = int(normalized_x * self.screen_width)
        screen_y = int(normalized_y * self.screen_height)
        
        return screen_x, screen_y
    
    def _smooth_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """
        좌표 스무딩 처리 (이동 평균)
        
        Args:
            x: 현재 x 좌표
            y: 현재 y 좌표
        
        Returns:
            (smoothed_x, smoothed_y): 스무딩된 좌표
        """
        smoothed_x = int(self.smoothing_factor * self.prev_x + (1 - self.smoothing_factor) * x)
        smoothed_y = int(self.smoothing_factor * self.prev_y + (1 - self.smoothing_factor) * y)
        
        self.prev_x = smoothed_x
        self.prev_y = smoothed_y
        
        return smoothed_x, smoothed_y
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        두 점 사이의 거리 계산
        
        Args:
            point1: 첫 번째 점 (x, y)
            point2: 두 번째 점 (x, y)
        
        Returns:
            거리 (픽셀 단위)
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # 픽셀 좌표로 변환
        px1 = int(x1 * self.frame_width)
        py1 = int(y1 * self.frame_height)
        px2 = int(x2 * self.frame_width)
        py2 = int(y2 * self.frame_height)
        
        return np.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임을 처리하고 마우스를 제어
        
        Args:
            frame: 입력 프레임 (BGR)
        
        Returns:
            처리된 프레임 (시각화 포함)
        """
        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        
        # RGB로 변환 (MediaPipe는 RGB를 사용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 손 감지
        results = self.hands.process(rgb_frame)
        
        # 클릭 쿨다운 감소
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        
        # 손이 감지된 경우
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 손 랜드마크 그리기
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # 검지 손가락 끝 (Landmark 8)
            index_tip = hand_landmarks.landmark[8]
            index_x = index_tip.x
            index_y = index_tip.y
            
            # 중지 손가락 끝 (Landmark 12)
            middle_tip = hand_landmarks.landmark[12]
            middle_x = middle_tip.x
            middle_y = middle_tip.y
            
            # 검지 손가락 위치를 화면 좌표로 변환
            screen_x, screen_y = self._normalize_coordinates(index_x, index_y)
            
            # 스무딩 적용
            smooth_x, smooth_y = self._smooth_coordinates(screen_x, screen_y)
            
            # 마우스 이동
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            # 검지와 중지 사이의 거리 계산
            distance = self._calculate_distance(
                (index_x, index_y),
                (middle_x, middle_y)
            )
            
            # 클릭 감지
            click_detected = False
            if distance < self.click_threshold and self.click_cooldown == 0:
                pyautogui.click()
                click_detected = True
                self.click_cooldown = self.click_cooldown_time
            
            # 시각화
            # 검지 손가락 위치 표시
            index_pixel_x = int(index_x * self.frame_width)
            index_pixel_y = int(index_y * self.frame_height)
            cv2.circle(frame, (index_pixel_x, index_pixel_y), 10, (0, 255, 255), -1)
            
            # 중지 손가락 위치 표시
            middle_pixel_x = int(middle_x * self.frame_width)
            middle_pixel_y = int(middle_y * self.frame_height)
            cv2.circle(frame, (middle_pixel_x, middle_pixel_y), 10, (255, 0, 255), -1)
            
            # 두 손가락 사이 선 그리기
            cv2.line(
                frame,
                (index_pixel_x, index_pixel_y),
                (middle_pixel_x, middle_pixel_y),
                (255, 255, 0),
                2
            )
            
            # 사용 가능한 영역 표시 (사각형)
            cv2.rectangle(
                frame,
                (self.usable_x_min, self.usable_y_min),
                (self.usable_x_max, self.usable_y_max),
                (255, 255, 255),
                2
            )
            
            # 정보 텍스트 표시
            info_text = [
                f"Distance: {distance:.1f}px",
                f"Click Threshold: {self.click_threshold}px",
                f"Screen: ({smooth_x}, {smooth_y})",
                f"Click: {'YES' if click_detected else 'NO'}"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(
                    frame, text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        else:
            # 손이 감지되지 않은 경우
            cv2.putText(
                frame, "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # 종료 안내 텍스트
        cv2.putText(
            frame, "Press 'q' to quit",
            (10, self.frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return frame
    
    def release(self):
        """리소스 해제"""
        self.hands.close()


def main():
    """메인 함수"""
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    # 웹캠 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # VirtualMouse 인스턴스 생성
    virtual_mouse = VirtualMouse(
        frame_width=640,
        frame_height=480,
        frame_margin=0.1,  # 10% 마진
        smoothing_factor=0.7,  # 스무딩 계수
        click_threshold=40.0  # 클릭 임계값
    )
    
    print("가상 마우스 프로그램이 시작되었습니다.")
    print("손을 웹캠 앞에 두고 검지 손가락으로 마우스를 제어하세요.")
    print("검지와 중지를 가까이 하면 클릭됩니다.")
    print("'q' 키를 누르면 종료됩니다.")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 프레임 처리
            processed_frame = virtual_mouse.process_frame(frame)
            
            # 결과 표시
            cv2.imshow('Virtual Mouse', processed_frame)
            
            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    
    finally:
        # 리소스 해제
        cap.release()
        virtual_mouse.release()
        cv2.destroyAllWindows()
        print("프로그램이 종료되었습니다.")


if __name__ == "__main__":
    main()

