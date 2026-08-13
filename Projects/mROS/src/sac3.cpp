/**
 * 1kHz Hardware Timer NN Inference with Telemetry + SAC Training Support
 * 
 * Control: 1kHz NN-based using IntervalTimer
 * Telemetry: 1kHz Publishing via Ring Buffer (Decoupled)
 * Trajectory: Deterministic Chirp (+/- 2.0 rad)
 */

#include <Arduino.h>
#include <NativeEthernet.h>
#include <NativeEthernetUdp.h>
#include <micro_ros_platformio.h>
#include <uxr/client/transport.h>
#include <rmw_microros/rmw_microros.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rclc/timer.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/float32.h>

// Include Initial Weights (Namespace to avoid collision)
namespace InitialWeights {
  #include "weights_30k.h"
}

//========================
// RL Constants
//========================
#define NN_INPUT 55       // Actor NN input (no critic extras)
#define OBS_PUB_SIZE 60   // Full obs published to PC (actor + critic extras)
#define NN_HIDDEN 128
#define NN_OUTPUT 16
#define K_PREVIEW 32
#define N_ACTION_HIST 16

//========================
// Dynamic Weights (Mutable)
//========================
float fc1_weight[NN_INPUT * NN_HIDDEN];
float fc1_bias[NN_HIDDEN];
float fc2_weight[NN_HIDDEN * NN_HIDDEN];
float fc2_bias[NN_HIDDEN];
float fc3_weight[NN_HIDDEN * NN_OUTPUT];
float fc3_bias[NN_OUTPUT];

bool policy_loaded = false;
uint32_t policy_version = 0;
volatile float weight_checksum = 0.0f; // FNV-1a Hash (23-bit)

// Rule-based half-step mode (cmd=2 activates this)
volatile bool rule_based_mode = false;
static int half_step_idx = 0;
const uint8_t HALF_STEP_SEQ[8] = {1, 5, 4, 6, 2, 10, 8, 9};
const float DEAD_BAND = 0.01f;  // ~0.57 deg

// FNV-1a Hash Helper
uint32_t calc_fnv1a(const uint8_t* data, size_t len) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < len; i++) {
    hash ^= data[i];
    hash *= 16777619u;
  }
  return hash;
}

// Buffer for receiving weights
#define CHUNK_SIZE 100
#define TOTAL_WEIGHTS (NN_INPUT*NN_HIDDEN + NN_HIDDEN + NN_HIDDEN*NN_HIDDEN + NN_HIDDEN + NN_HIDDEN*NN_OUTPUT + NN_OUTPUT)

int weights_received = 0;
bool receiving_weights = false;
DMAMEM float full_weights_buffer[TOTAL_WEIGHTS + 100];
float chunk_buffer[CHUNK_SIZE + 10];

// NN Buffers (Runtime)
float hidden1[NN_HIDDEN];
float hidden2[NN_HIDDEN];
// Discrete SAC: no continuous action buffer needed

// Observation Buffer (full 60 dims: 55 actor + 5 critic extras for PC)
float obs[OBS_PUB_SIZE];

//========================
// Ring Buffer for 1kHz Publishing
//========================
struct ObservationData {
  float obs[OBS_PUB_SIZE];
  float encoder[14]; // 0-9: original, 10-13: unused (discrete SAC)
};

#define RING_BUF_SIZE 50
ObservationData ring_buffer[RING_BUF_SIZE];
volatile int rb_head = 0;
volatile int rb_tail = 0;

//========================
// Performance Measurement
//========================
volatile unsigned long last_isr_start_us = 0;
volatile float control_period_us = 1000.0f;
volatile float inference_time_us = 0.0f;
volatile float isr_execution_time_us = 0.0f;

//========================
// Error Handling
//========================
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){error_loop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

void error_loop(){
  while(1){
    digitalWrite(13, !digitalRead(13));
    delay(100);
  }
}

//========================
// 핀 설정
//========================
const int PIN_HW  = 5;
const int PIN_RCL = 6;
const int PIN_LED = 13;

//========================
// Encoder
//========================
#define ENCODER_A 20
#define ENCODER_B 21

const int CPR = 1024;
const int QUAD_FACTOR = 4;
const float RAD_PER_COUNT = (2.0f * PI) / (CPR * QUAD_FACTOR);
const float ALPHA = 0.05f;

volatile long encoderCount = 0;
volatile float currentRad = 0.0f;
volatile float filtSpeedRad = 0.0f;
static unsigned long prevTime_enc = 0;
static long prevCount = 0;

void ISR_A() {
  bool A = digitalRead(ENCODER_A);
  bool B = digitalRead(ENCODER_B);
  if (A == B) encoderCount++; else encoderCount--;
}

void ISR_B() {
  bool A = digitalRead(ENCODER_A);
  bool B = digitalRead(ENCODER_B);
  if (A != B) encoderCount++; else encoderCount--;
}

static inline void updateEncoder() {
  unsigned long now = micros();
  unsigned long dt_us = now - prevTime_enc;
  if (dt_us < 100) return; 
  
  float dt = dt_us / 1000000.0f;
  
  long count;
  noInterrupts();
  count = encoderCount;
  interrupts();
  
  long deltaCount = count - prevCount;
  prevCount = count;
  prevTime_enc = now;
  
  float rawSpeedRad = (deltaCount * RAD_PER_COUNT) / dt;
  filtSpeedRad = ALPHA * rawSpeedRad + (1.0f - ALPHA) * filtSpeedRad;
  currentRad = count * RAD_PER_COUNT;
}

//========================
// Stepper Motor (L298N)
//========================
const int ENA = 9;
const int ENB = 10;
const int IN1 = 27;
const int IN2 = 28;
const int IN3 = 29;
const int IN4 = 30;

const uint8_t L298N_PATTERN[16] = {
  0b0000, 0b0001, 0b0010, 0b0011,
  0b0100, 0b0101, 0b0110, 0b0111,
  0b1000, 0b1001, 0b1010, 0b1011,
  0b1100, 0b1101, 0b1110, 0b1111
};

static uint8_t last_action = 0;
static uint8_t action_history[N_ACTION_HIST] = {0};
static uint8_t action_hist_idx = 0;

void motor_stop() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  digitalWrite(ENA, LOW); digitalWrite(ENB, LOW);
}

void motor_enable() {
  digitalWrite(ENA, HIGH);
  digitalWrite(ENB, HIGH);
}

void apply_action(uint8_t action) {
  uint8_t pattern = L298N_PATTERN[action & 0x0F];
  digitalWrite(IN1, (pattern & 0x01) ? HIGH : LOW);
  digitalWrite(IN2, (pattern & 0x02) ? HIGH : LOW);
  digitalWrite(IN3, (pattern & 0x04) ? HIGH : LOW);
  digitalWrite(IN4, (pattern & 0x08) ? HIGH : LOW);
  
  last_action = action;
  action_history[action_hist_idx] = action;
  action_hist_idx = (action_hist_idx + 1) % N_ACTION_HIST;
}

//========================
// Neural Network
//========================
inline float relu(float x) { return (x > 0.0f) ? x : 0.0f; }

uint8_t policy_forward(float* obs) {
  // Layer 1
  for (int i = 0; i < NN_HIDDEN; i++) {
    float sum = fc1_bias[i];
    for (int j = 0; j < NN_INPUT; j++) {
      sum += obs[j] * fc1_weight[j * NN_HIDDEN + i];
    }
    hidden1[i] = relu(sum);
  }

  // Layer 2
  for (int i = 0; i < NN_HIDDEN; i++) {
    float sum = fc2_bias[i];
    for (int j = 0; j < NN_HIDDEN; j++) {
      sum += hidden1[j] * fc2_weight[j * NN_HIDDEN + i];
    }
    hidden2[i] = relu(sum);
  }

  // Output Layer: argmax over 16 logits (Discrete SAC)
  // No softmax needed — argmax is monotonic
  float max_logit = -1e30f;
  uint8_t best_action = 0;
  for (int i = 0; i < NN_OUTPUT; i++) {
    float logit = fc3_bias[i];
    for (int j = 0; j < NN_HIDDEN; j++) {
      logit += hidden2[j] * fc3_weight[j * NN_OUTPUT + i];
    }
    if (logit > max_logit) {
      max_logit = logit;
      best_action = i;
    }
  }
  return best_action;
}

// Load Weights
void load_weights_from_buffer(float* buffer) {
  int idx = 0;
  for(int i=0; i<NN_INPUT*NN_HIDDEN; i++)  fc1_weight[i] = buffer[idx++];
  for(int i=0; i<NN_HIDDEN; i++)           fc1_bias[i]   = buffer[idx++];
  for(int i=0; i<NN_HIDDEN*NN_HIDDEN; i++) fc2_weight[i] = buffer[idx++];
  for(int i=0; i<NN_HIDDEN; i++)           fc2_bias[i]   = buffer[idx++];
  for(int i=0; i<NN_HIDDEN*NN_OUTPUT; i++) fc3_weight[i] = buffer[idx++];
  for(int i=0; i<NN_OUTPUT; i++)           fc3_bias[i]   = buffer[idx++];
  
  policy_loaded = true;
  policy_version++;
  
  // Calculate Checksum (FNV-1a)
  // Cast float buffer to uint8_t for byte-wise hashing
  uint32_t hash = calc_fnv1a((const uint8_t*)fc1_weight, sizeof(fc1_weight));
  hash ^= calc_fnv1a((const uint8_t*)fc1_bias, sizeof(fc1_bias)); // XOR combine
  hash ^= calc_fnv1a((const uint8_t*)fc2_weight, sizeof(fc2_weight));
  hash ^= calc_fnv1a((const uint8_t*)fc2_bias, sizeof(fc2_bias));
  hash ^= calc_fnv1a((const uint8_t*)fc3_weight, sizeof(fc3_weight));
  hash ^= calc_fnv1a((const uint8_t*)fc3_bias, sizeof(fc3_bias));
  
  // Mask to 23 bits for exact float representation
  weight_checksum = (float)(hash & 0x7FFFFF);
}

//========================
// Deterministic Chirp Trajectory
//========================
volatile bool episode_running = false;
volatile bool trigger_end_signal = false;
volatile uint32_t trajectory_tick = 0;     // Integer tick counter (1kHz)
volatile float trajectory_time_s = 0.0f;   // Computed from tick (no accumulation error)

const float CHIRP_AMP = 1.5708f; // π/2 rad (Python sim: chirp_amp_rad=π/2)
const float CHIRP_F0 = 0.1f;
const float CHIRP_F1 = 1.0f;
const float CHIRP_DURATION = 30.0f;

volatile float target_angle_rad = 0.0f;
volatile float target_velocity_rad = 0.0f;

// Precompute trajectory point (Deterministic)
void compute_trajectory_2rad(float t, volatile float* angle, volatile float* velocity) {
  if (t >= CHIRP_DURATION) {
    *angle = 0.0f;
    *velocity = 0.0f;
    return;
  }
  
  // Linear Chirp: f(t) = f0 + k*t
  float k = (CHIRP_F1 - CHIRP_F0) / CHIRP_DURATION;
  float phase = 2.0f * PI * (CHIRP_F0 * t + 0.5f * k * t * t);
  
  // Angle
  float raw_angle = CHIRP_AMP * sinf(phase);
  
  // Clamp to +/- chirp_amp
  if (raw_angle > CHIRP_AMP) raw_angle = CHIRP_AMP;
  if (raw_angle < -CHIRP_AMP) raw_angle = -CHIRP_AMP;
  *angle = raw_angle;
  
  // Velocity (Derivative of chirp)
  // d/dt [ A * sin(2*pi * (f0*t + 0.5*k*t^2)) ]
  // = A * cos(...) * 2*pi * (f0 + k*t)
  *velocity = CHIRP_AMP * cosf(phase) * 2.0f * PI * (CHIRP_F0 + k * t);
}

void start_episode_deterministic() {
  motor_stop();
  noInterrupts();
  encoderCount = 0;
  interrupts();
  currentRad = 0.0f;
  filtSpeedRad = 0.0f;
  prevTime_enc = micros();
  prevCount = 0;
  
  last_action = 0;
  memset(action_history, 0, sizeof(action_history));
  action_hist_idx = 0;
  
  // Deterministic Reset
  trajectory_tick = 0;
  trajectory_time_s = 0.0f;
  last_isr_start_us = 0; 
  trigger_end_signal = false;
  episode_running = true;
  
  // Reset Ring Buffer
  rb_head = 0;
  rb_tail = 0;
  
  motor_enable();
}

//========================
// Network & ROS Params
//========================
IPAddress agent_ip(192, 168, 1, 12);  // PC Ethernet IP
uint16_t agent_port = 8888;
IPAddress local_ip(192, 168, 1, 10);
IPAddress gateway(192, 168, 1, 1);
IPAddress subnet(255, 255, 255, 0);
byte mac[6] = {0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED};
EthernetUDP udp;

rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timerT;

rcl_publisher_t obs_publisher;
rcl_publisher_t encoder_publisher;
rcl_publisher_t episode_cmd_teensy_pub;
rcl_publisher_t policy_ack_publisher;

rcl_subscription_t episode_cmd_pc_sub;
rcl_subscription_t policy_chunk_sub;

std_msgs__msg__Float32MultiArray obs_msg;
std_msgs__msg__Float32MultiArray encoder_msg;
std_msgs__msg__Int32 episode_cmd_sub_msg;
std_msgs__msg__Float32MultiArray policy_chunk_msg;
std_msgs__msg__Int32 episode_cmd_pub_msg;
std_msgs__msg__Float32 policy_ack_msg;

float obs_pub_buf[OBS_PUB_SIZE];
float encoder_pub_buf[14];

//========================
// Transport
//========================
bool transport_open(struct uxrCustomTransport * transport) {
  (void) transport;
  Ethernet.begin(mac, local_ip, gateway, subnet);
  delay(300);
  return udp.begin(agent_port);
}
bool transport_close(struct uxrCustomTransport * transport) {
  (void) transport; udp.stop(); return true;
}
size_t transport_write(struct uxrCustomTransport* transport, const uint8_t* buf, size_t len, uint8_t* err) {
  (void) transport; (void) err;
  udp.beginPacket(agent_ip, agent_port);
  udp.write(buf, len);
  udp.endPacket();
  return len;
}
size_t transport_read(struct uxrCustomTransport* transport, uint8_t* buf, size_t len, int timeout, uint8_t* err) {
  (void) transport; (void) err;
  unsigned long start = millis();
  while ((millis() - start) < (unsigned long)timeout) {
    int packetSize = udp.parsePacket();
    if (packetSize > 0) {
      int read_len = udp.read(buf, len);
      return (read_len > 0) ? read_len : 0;
    }
  }
  return 0;
}
void set_microros_teensy_ethernet_transports() {
  rmw_uros_set_custom_transport(false, NULL, transport_open, transport_close, transport_write, transport_read);
}

//========================
// Callbacks
//========================
void episode_cmd_callback(const void * msgin) {
  const std_msgs__msg__Int32 * msg = (const std_msgs__msg__Int32 *)msgin;
  int cmd = msg->data;
  digitalWrite(PIN_LED, !digitalRead(PIN_LED)); // Visual Debug: Toggle LED on command
  if (cmd == 0 && !episode_running) { rule_based_mode = false; start_episode_deterministic(); }
  else if (cmd == 2 && !episode_running) { rule_based_mode = true; half_step_idx = 0; start_episode_deterministic(); }
  else if (cmd == 1) { episode_running = false; motor_stop(); }
}

void policy_chunk_callback(const void * msgin) {
  const std_msgs__msg__Float32MultiArray * msg = (const std_msgs__msg__Float32MultiArray *)msgin;
  
  // Protocol: [chunk_idx, total_chunks, data...]
  if (msg->data.size < 2) return; 

  int chunk_idx = (int)msg->data.data[0];
  int total_chunks = (int)msg->data.data[1];
  int num_weights = msg->data.size - 2;

  // Start condition: Reset on first chunk
  if (chunk_idx == 0) {
    weights_received = 0;
    receiving_weights = true;
  }
  
  if (!receiving_weights) return;

  // Direct Offset Write (Robust to out-of-order within reason, mainly removes -1.0 dependency)
  int offset = chunk_idx * CHUNK_SIZE; // CHUNK_SIZE is 100
  
  for (int i = 0; i < num_weights; i++) {
    if (offset + i < TOTAL_WEIGHTS) {
      full_weights_buffer[offset + i] = msg->data.data[2 + i];
    }
  }
  weights_received += num_weights;

  // Check completion by index (since Best Effort might drop packets, this is a heuristic)
  // Ideally we check if weights_received approx equals TOTAL_WEIGHTS, but let's allow
  // the update to happen on the last packet.
  if (chunk_idx == total_chunks - 1) {
    load_weights_from_buffer(full_weights_buffer);
    receiving_weights = false;
    
    // Ack with version (Positive)
    policy_ack_msg.data = (float)policy_version;
    rcl_publish(&policy_ack_publisher, &policy_ack_msg, NULL);
  } else {
    // Ack with progress (Negative index)
    // Helps debugging if we see -1, -2, -3 ...
    policy_ack_msg.data = -(float)chunk_idx; 
    rcl_publish(&policy_ack_publisher, &policy_ack_msg, NULL);
  }
}

// Heartbeat Timer Callback
void telemetry_callback(rcl_timer_t *timerT, int64_t last_call_timeT) {
  RCLC_UNUSED(last_call_timeT);
  // Optional: Toggle LED or just keep connection alive
}

//========================
// 1kHz Control Interrupt
//========================
IntervalTimer controlTimer;

void controlISR() {
  unsigned long isr_start = micros();
  digitalToggleFast(PIN_HW);
  
  if (last_isr_start_us > 0) {
    control_period_us = (float)(isr_start - last_isr_start_us);
  }
  last_isr_start_us = isr_start;
  
  updateEncoder();
  
  if (episode_running) {
    // 1. Update Time (integer tick → no float accumulation error)
    trajectory_tick++;
    trajectory_time_s = trajectory_tick * 0.001f;
    
    // 2. Check Expiry: 30000 ticks (30s chirp) + 1000 ticks (1s settle) = 31000
    if (trajectory_tick >= 31000) {
      trigger_end_signal = true;
      episode_running = false;
      motor_stop();
      return;
    }
    
    // 3. Compute Target (Deterministic)
    compute_trajectory_2rad(trajectory_time_s, &target_angle_rad, &target_velocity_rad);
    
    // 4. Construct Observation
    // Normalization Constants (Difficulty 2)
    // theta_scale = 1.57 * 1.3 = 2.041
    // theta_dot_scale = 2.041 * 2 * PI * 1.2 = 15.38
    const float THETA_SCALE = 1.5708f;    // chirp_amp = π/2 (Python sim)
    const float THETA_DOT_SCALE = 9.8696f; // chirp_amp * 2π * max(f0,f1) = 1.5708 * 6.2832 * 1.0
    const float ACTION_SCALE = 15.0f;
    const float EPS = 1e-9f;

    // 4. Construct Observation (NORMALIZED)
    obs[0] = target_angle_rad / (THETA_SCALE + EPS);
    obs[1] = target_velocity_rad / (THETA_DOT_SCALE + EPS);
    obs[2] = currentRad / (THETA_SCALE + EPS);
    obs[3] = filtSpeedRad / (THETA_DOT_SCALE + EPS);
    obs[4] = (float)last_action / ACTION_SCALE;
    obs[5] = sin(currentRad); // These might be raw? Checking Python...
    obs[6] = cos(currentRad); 
  
    // I MUST UPDATE THE LAYOUT TO MATCH PYTHON EXACTLY!
    
    // Constructing Python-matching vector:
    obs[0] = target_angle_rad / (THETA_SCALE + EPS);
    obs[1] = target_velocity_rad / (THETA_DOT_SCALE + EPS);
    
    // Preview (32) — clamp to chirp boundary to prevent premature braking
    for(int i=0; i<K_PREVIEW; i++) {
       float t_pred = trajectory_time_s + (i+1)*0.001f;
       // Clamp: never show settling period (target=0) in preview
       if (t_pred > CHIRP_DURATION - 0.001f) t_pred = CHIRP_DURATION - 0.001f;
       float pred_angle, pred_vel;
       compute_trajectory_2rad(t_pred, &pred_angle, &pred_vel);
       obs[2 + i] = pred_angle / (THETA_SCALE + EPS);
    }
    
    // Last Action (1)
    obs[34] = (float)last_action / ACTION_SCALE;
    
    // History (16) - FIFO oldest first (Python: deque.append → oldest first)
    for(int i=0; i<N_ACTION_HIST; i++) {
      int idx = (action_hist_idx + i) % N_ACTION_HIST;
      obs[35 + i] = (float)action_history[idx] / ACTION_SCALE;
    }
    
    // Encoder Feats (4) - [51..54]
    obs[51] = currentRad / (THETA_SCALE + EPS);
    obs[52] = filtSpeedRad / (THETA_DOT_SCALE + EPS);
    
    float error = currentRad - target_angle_rad;
    float error_dot = filtSpeedRad - target_velocity_rad;
    // error_scale = 0.175 (~10 deg)
    const float ERROR_SCALE = 0.175f;
    
    // Clip errors to -2.0 to 2.0 (after norm) as per Python
    obs[53] = constrain(error / (ERROR_SCALE + EPS), -2.0f, 2.0f);
    obs[54] = constrain(error_dot / (ERROR_SCALE + EPS), -2.0f, 2.0f);
    
    // obs[0..54] = Actor obs (55 dims) — used by NN (policy_forward reads obs[0..NN_INPUT-1])
    // obs[55..59] = Critic extras — not used by NN, but published to PC for critic training
    obs[55] = trajectory_time_s / CHIRP_DURATION;
    obs[56] = (float)((last_action >> 0) & 1);
    obs[57] = (float)((last_action >> 1) & 1);
    obs[58] = (float)((last_action >> 2) & 1);
    obs[59] = (float)((last_action >> 3) & 1);

    // 5. Action Selection (NN or Rule-based)
    uint8_t action = 0;
    if (rule_based_mode) {
        // Half-step control: error = target - current
        float ctrl_error = target_angle_rad - currentRad;
        if (ctrl_error > DEAD_BAND) {
            half_step_idx = (half_step_idx + 1) % 8;
        } else if (ctrl_error < -DEAD_BAND) {
            half_step_idx = (half_step_idx - 1 + 8) % 8;
        }
        action = HALF_STEP_SEQ[half_step_idx];
    } else {
        if (policy_loaded) {
            unsigned long inf_start = micros();
            action = policy_forward(obs);
            inference_time_us = (float)(micros() - inf_start);
        }
    }
    
    // 6. Apply
    apply_action(action);
    
    // 7. Ring Buffer Push
    int next_head = (rb_head + 1) % RING_BUF_SIZE;
    if (next_head != rb_tail) {
      for(int i=0; i<OBS_PUB_SIZE; i++) ring_buffer[rb_head].obs[i] = obs[i];
      ring_buffer[rb_head].encoder[0] = currentRad;
      ring_buffer[rb_head].encoder[1] = filtSpeedRad;
      ring_buffer[rb_head].encoder[2] = target_angle_rad;
      ring_buffer[rb_head].encoder[3] = target_velocity_rad;
      ring_buffer[rb_head].encoder[4] = (float)action;
      ring_buffer[rb_head].encoder[5] = control_period_us;
      ring_buffer[rb_head].encoder[6] = trajectory_time_s; // Send Hardware Time
      ring_buffer[rb_head].encoder[7] = inference_time_us;
      ring_buffer[rb_head].encoder[8] = isr_execution_time_us;
      ring_buffer[rb_head].encoder[9] = weight_checksum;
      // Discrete SAC: no continuous action needed, zero-fill
      ring_buffer[rb_head].encoder[10] = 0.0f;
      ring_buffer[rb_head].encoder[11] = 0.0f;
      ring_buffer[rb_head].encoder[12] = 0.0f;
      ring_buffer[rb_head].encoder[13] = 0.0f;

      rb_head = next_head;
    }
  }
  
  isr_execution_time_us = (float)(micros() - isr_start);
}

//========================
// Setup & Loop
//========================
void setup() {
  // (No RNG needed - deterministic argmax over logits)

  pinMode(PIN_HW, OUTPUT);
  pinMode(PIN_RCL, OUTPUT);
  pinMode(PIN_LED, OUTPUT);
  
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  motor_stop();
  
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), ISR_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), ISR_B, CHANGE);
  
  set_microros_teensy_ethernet_transports();
  
  allocator = rcl_get_default_allocator();
  rcl_init_options_t init_options = rcl_get_zero_initialized_init_options();
  RCCHECK(rcl_init_options_init(&init_options, allocator));
  RCCHECK(rcl_init_options_set_domain_id(&init_options, 121));
  RCCHECK(rclc_support_init_with_options(&support, 0, NULL, &init_options, &allocator));
  RCCHECK(rclc_node_init_default(&node, "teensy_sac_node", "", &support));
  
  // Publishers
  RCCHECK(rclc_publisher_init_best_effort(
    &obs_publisher, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "rl_observation"));
    
  rmw_qos_profile_t qos = rmw_qos_profile_sensor_data;
  qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
  qos.depth = 1;
  RCCHECK(rclc_publisher_init(&encoder_publisher, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "encoder_feedback", &qos));
  encoder_msg.data.data = encoder_pub_buf;
  encoder_msg.data.size = 14;
  encoder_msg.data.capacity = 14;
    
  RCCHECK(rclc_publisher_init_default(
    &episode_cmd_teensy_pub, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
    "episode_cmd_teensy"));
    
  RCCHECK(rclc_publisher_init_default(
    &policy_ack_publisher, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
    "policy_ack"));
    
  // Subscribers
  RCCHECK(rclc_subscription_init(
    &episode_cmd_pc_sub, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
    "episode_cmd_pc", &qos));
    
  RCCHECK(rclc_subscription_init(
    &policy_chunk_sub, &node,
    ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
    "policy_chunk", &qos));
    
    
  RCCHECK(rclc_timer_init_default(&timerT, &support, 
    RCL_MS_TO_NS(100), telemetry_callback)); // Heartbeat

  RCCHECK(rclc_executor_init(&executor, &support.context, 4, &allocator));
  RCCHECK(rclc_executor_add_timer(&executor, &timerT));
  RCCHECK(rclc_executor_add_subscription(&executor, &episode_cmd_pc_sub, &episode_cmd_sub_msg, &episode_cmd_callback, ON_NEW_DATA));
  RCCHECK(rclc_executor_add_subscription(&executor, &policy_chunk_sub, &policy_chunk_msg, &policy_chunk_callback, ON_NEW_DATA));
  
  // Buffers
  obs_msg.data.capacity = OBS_PUB_SIZE;
  obs_msg.data.data = obs_pub_buf;
  obs_msg.data.size = OBS_PUB_SIZE;
  
  // encoder_msg.data.capacity = 9; // This line is removed as per the instruction
  // encoder_msg.data.data = encoder_pub_buf; // This line is removed as per the instruction
  // encoder_msg.data.size = 9; // This line is removed as per the instruction
  
  policy_chunk_msg.data.capacity = CHUNK_SIZE + 10;
  policy_chunk_msg.data.data = chunk_buffer;
  policy_chunk_msg.data.size = 0;

  // Start 1kHz Timer
  controlTimer.begin(controlISR, 1000); // 1000us = 1kHz
  
  // Load Initial Weights from Header
  // fc1, fc2 are same size (NN_INPUT×128, 128×128) — copy directly
  memcpy(fc1_weight, InitialWeights::fc1_weight, sizeof(fc1_weight));
  memcpy(fc1_bias,   InitialWeights::fc1_bias,   sizeof(fc1_bias));
  memcpy(fc2_weight, InitialWeights::fc2_weight, sizeof(fc2_weight));
  memcpy(fc2_bias,   InitialWeights::fc2_bias,   sizeof(fc2_bias));
  // fc3 changed from 128×4 to 128×16 — zero-init (will be overwritten by PC policy)
  memset(fc3_weight, 0, sizeof(fc3_weight));
  memset(fc3_bias, 0, sizeof(fc3_bias));

  policy_loaded = false;  // No valid discrete policy until PC sends one
  policy_version = 0;
  
  // No valid initial checksum (fc3 is zeroed, policy_loaded=false)
  weight_checksum = 0.0f;
  
  trajectory_tick = 0;
  trajectory_time_s = 0.0f; // Explicit Reset at boot
}

void loop() {
  // Handle state signal
  if (trigger_end_signal) {
    episode_cmd_pub_msg.data = 1; // STOP
    rcl_publish(&episode_cmd_teensy_pub, &episode_cmd_pub_msg, NULL);
    trigger_end_signal = false;
  }

  // Poll ROS
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));
  
  // Publish from Ring Buffer (Flush whatever is in it, regardless of episode state)
  while (rb_head != rb_tail) {
     digitalToggleFast(PIN_RCL);
     
     ObservationData& data = ring_buffer[rb_tail];
     
     // 1. Publish Encoder Feedback
     for(int i=0; i<14; i++) encoder_pub_buf[i] = data.encoder[i];
     RCSOFTCHECK(rcl_publish(&encoder_publisher, &encoder_msg, NULL));
     
     // 2. Obs Pub
     for(int i=0; i<OBS_PUB_SIZE; i++) obs_msg.data.data[i] = data.obs[i];
     rcl_publish(&obs_publisher, &obs_msg, NULL);
     
     rb_tail = (rb_tail + 1) % RING_BUF_SIZE;
  }
  
  if (!episode_running) {
    digitalWrite(PIN_LED, (millis()/500)%2);
  }
}