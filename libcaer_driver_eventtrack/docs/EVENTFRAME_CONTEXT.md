# libcaer_driver_eventtrack — 개조 컨텍스트 요약

> upstream [`libcaer_driver`](https://github.com/ros-event-camera/libcaer_driver)
> (Bernd Pfrommer)를 포크하여, 기존 event **stream** 출력에 더해 dense event
> **frame**(TimeSurface)을 ROS로 publish하도록 개조한 워크스페이스의 요약 문서.

## 1. 워크스페이스 구성

| 패키지 | 역할 |
|---|---|
| `libcaer_vendor_eventframe` | libcaer C 라이브러리 벤더링 (원본과 동일, 이름만 변경) |
| `libcaer_driver_eventframe_msgs` | **새 메시지 패키지** — `msg/TimeSurface.msg` |
| `libcaer_driver_eventtrack` | 드라이버 노드 (개조 핵심) |

빌드 루트는 이 레포(`~/libcaer_driver_rep`)이며 세 패키지가 루트에 직접 존재.
`libcaer_vendor_eventframe`는 `ament_vendor`로 빌드되므로, libcaer 서브트리를
수정했다면 **빌드 전에 `libcaer/`를 커밋**해야 반영됨.

## 2. upstream 대비 개조 요지

- `~/events` (event stream, `event_camera_msgs/EventPacket`) 퍼블리셔는 **그대로 유지**.
- 그 옆에 `~/events_rep` (**TimeSurface** = event frame) 퍼블리셔를 **추가**. 완전 대체가
  아니라 병렬 출력.
- TimeSurface 생성 로직은 전부 [`src/driver.cpp`](../src/driver.cpp)에 있음
  (`src/message_converter.cpp`는 upstream 그대로).

### TimeSurface 메시지 스펙 ([msg/TimeSurface.msg](../../libcaer_driver_eventframe_msgs/msg/TimeSurface.msg))

- dense `(H, W, C)` float32 텐서. `channels = 2 * n_bins`.
- `channel = 2 * bin_idx + polarity` (bin-major interleaved, `polarity` 0=OFF/1=ON).
- 각 셀 값 = 그 시간창에서 해당 (y,x,channel)에 마지막으로 들어온 이벤트의
  정규화 시각 `t_norm ∈ [0, 1]`.
- 레이아웃(row-major): `data[y * width * channels + x * channels + c]`.
- `window_start_us` / `window_end_us`는 이 프레임을 만든 이벤트 배치의 센서 타임스탬프 경계.

### 발행 경로 (driver.cpp)

- [`polarityPacketCallback`](../src/driver.cpp): 이벤트를 SoA 버퍼(`evBufX_/Y_/P_/Tus_`)에 누적.
- **timer-driven (기본)** — [`tsTimerCallback`](../src/driver.cpp): `time_surface_window_us`
  주기의 wall-timer가 버퍼를 swap해 비우고 TimeSurface를 만들어 발행. 이벤트가 없어도
  고정 프레임레이트로(0으로 채운 프레임) 발행.
- **event-driven (legacy)** — [`computeAndPublishTimeSurface`](../src/driver.cpp): 패킷이
  window를 넘기면 콜백 안에서 쪼개 발행.
- 계산식은 BlinkTrack `util/representations.py::TimeSurface`를 미러링:
  `t_norm=(t-t0)/(tN-t0)`, `bin=clamp(floor(t_norm*n_bins),0,nb-1)`,
  `out[y,x,2*bin+p]=max(out, t_norm)`.

## 3. 파라미터

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `time_surface_enabled` | `true` | `~/events_rep` 발행 on/off |
| `time_surface_window_us` | `3000` | 시간창 = 프레임 주기 (µs) |
| `time_surface_n_bins` | `5` | 시간 bin 수 (channels = 2×) |
| `time_surface_timer_driven` | `true` | 고정 레이트(true) vs 이벤트 구동(false) |
| `time_surface_queue_size` | `10` | 발행 QoS depth |
| `packet_interval_us` | `10000` | libcaer 패킷 flush 간격 (저지연용, 예: 2500→~400fps) |
| `aps_enabled` | `False` | APS 그레이스케일 프레임(`~/image_raw`) on/off (DAVIS 전용) |

> **주의 — `packet_interval_us` ≤ `time_surface_window_us` 로 맞출 것.**
> 두 값은 독립적인 시계다: `window_us`는 프레임을 뽑는 타이머 주기(FPS = 1/window),
> `packet_interval_us`는 libcaer가 이벤트를 드라이버로 넘기는 간격(≈ latency 하한).
> `packet_interval` > `window`이면 FPS는 유지되지만 이벤트가 packet 주기로만 도착해
> **대부분의 프레임이 빈(all-zero) 프레임**이 되고, 이벤트가 담긴 프레임은 `packet_interval`
> 만큼의 덩어리를 담는다 → window를 작게 준 의미가 사라지고 대역폭만 낭비.
> 반대로 `packet_interval ≤ window`이면 매 틱마다 신선한 이벤트가 채워져 정상 동작한다.
> 예) `time_surface_window_us:=5000 packet_interval_us:=3000` → 5ms마다 발행, 매 프레임에
> 직전 5ms(≈1~2 packet)치 이벤트가 담김.

## 4. 발행 토픽 (노드명 기본 `event_camera`)

- `/event_camera/events_rep` — **TimeSurface (event frame) ← 핵심 출력**
- `/event_camera/events` — 기존 event stream
- `/event_camera/imu`, `/event_camera/image_raw`
- (본 작업 추가) `/event_camera/events_rep_image` — TimeSurface를 RGB로 변환한 시각화 이미지

## 5. 본 작업으로 추가/변경한 것

### (a) 런치 파라미터 노출
[`launch/driver_node.launch.py`](../launch/driver_node.launch.py)에 `time_surface_*` 4종과
`packet_interval_us`를 `DeclareLaunchArgument`로 노출하고 파라미터 dict에 매핑
(정수형은 `ParameterValue(..., value_type=int)`로 타입 명시).

### (c) TimeSurface → RGB 시각화 노드
- [`src/timesurface_vis_node.py`](../src/timesurface_vis_node.py) — rclpy 노드.
  - `~/events_rep` 구독(best_effort). 콜백은 **최신 메시지만 저장**, 실제 변환/발행은
    **5fps 타이머**에서만 수행 → 초당 수백 Hz 입력과 처리율 분리.
  - `BlinkTrack/util/vis.py::time_surface_to_rgb`를 그대로 포팅. msg `data`(H,W,C)를
    `(C,H,W)`로 transpose 후 함수에 투입 (채널 순서 `2*bin+p`가 BlinkTrack과 동일해
    재배열 불필요; in-place 변형 방지용 `.copy()`).
  - `sensor_msgs/Image`(rgb8)를 수동 채움으로 발행(cv_bridge 의존성 회피), 소스 header 보존.
  - 파라미터: `input_topic`, `output_topic`, `rate`(기본 5.0).
- [`launch/timesurface_vis.launch.py`](../launch/timesurface_vis.launch.py) — 전용 launch.
- [`CMakeLists.txt`](../CMakeLists.txt) `install(PROGRAMS)`에 스크립트 추가.
- [`package.xml`](../package.xml)에 `rclpy` / `python3-numpy` `exec_depend` 추가.

### 채널 순서 메모 (중요)
driver.cpp, `TimeSurface.msg`, BlinkTrack `TimeSurface.convert`가 모두
`c = 2 * bin_idx + polarity`(bin-major interleaved)로 **동일**. 따라서 시각화 노드에서
transpose만 하면 `time_surface_to_rgb`가 BlinkTrack 파이프라인과 **동일한 결과**를 냄.

## 6. 빌드 / 실행 / 검증

```bash
# 빌드
cd ~/libcaer_driver_rep
colcon build --symlink-install --packages-select \
  libcaer_driver_eventframe_msgs libcaer_driver_eventtrack
source install/setup.bash

# 드라이버 실행 (파라미터 오버라이드 예시)
ros2 launch libcaer_driver_eventtrack driver_node.launch.py \
  time_surface_n_bins:=3 time_surface_window_us:=5000
ros2 param get /event_camera time_surface_n_bins        # -> 3
ros2 topic echo /event_camera/events_rep --no-arr       # channels == 6

# 시각화 노드 (5fps)
ros2 launch libcaer_driver_eventtrack timesurface_vis.launch.py
ros2 topic hz /event_camera/events_rep_image            # ~5 Hz
ros2 run rqt_image_view rqt_image_view /event_camera/events_rep_image
```
