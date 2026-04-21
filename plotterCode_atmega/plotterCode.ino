// ==============================
// Full software closed-loop XY control
// For DC motors + L298N/L293D + 2 linear pots
// ==============================

// -------- Position feedback pins --------
const int X_SENSE_PIN = A0;
const int Y_SENSE_PIN = A1;

// -------- Motor pins --------
// X axis motor
// Y axis motor
const int EN_Y  = 9;
const int Y_IN1 = 7;
const int Y_IN2 = 8;

// X axis motor
const int EN_X  = 3;
const int X_IN1 = 5;
const int X_IN2 = 4;
// -------- Mechanical calibration --------
// Replace these with your measured endpoint ADC values
int X_ADC_MIN = 17;   // X at 0 mm
int X_ADC_MAX = 800;   // X at X_TRAVEL_MM

int Y_ADC_MIN = 17;   // Y at 0 mm
int Y_ADC_MAX = 1000;   // Y at Y_TRAVEL_MM

const float X_TRAVEL_MM = 80.0;
const float Y_TRAVEL_MM =80.0;

// -------- Controller tuning --------
float KP_X = 6.0;
float KP_Y = 6.0;

const float DEADBAND_MM = 1.0;   // stop within this error
const int MIN_PWM = 90;          // enough to overcome friction
const int MAX_PWM = 180;         // safety cap

// -------- Targets --------
float targetX_mm = 40.0;
float targetY_mm = 50.0;

// -------- Timing --------
unsigned long lastControlMs = 0;
unsigned long lastPrintMs = 0;
const unsigned long CONTROL_PERIOD_MS = 10;
const unsigned long PRINT_PERIOD_MS = 50;

// ==============================
// Helpers
// ==============================
float clampFloat(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

float adcToMm(int adc, int adcMin, int adcMax, float travelMm) {
  if (adcMax == adcMin) return 0.0;
  float frac = (float)(adc - adcMin) / (float)(adcMax - adcMin);
  frac = clampFloat(frac, 0.0, 1.0);
  return frac * travelMm;
}

int readXadc() {
  return analogRead(X_SENSE_PIN);
}

int readYadc() {
  return analogRead(Y_SENSE_PIN);
}

float readXmm() {
  return adcToMm(readXadc(), X_ADC_MIN, X_ADC_MAX, X_TRAVEL_MM);
}

float readYmm() {
  return adcToMm(readYadc(), Y_ADC_MIN, Y_ADC_MAX, Y_TRAVEL_MM);
}

// ==============================
// Motor drive
// ==============================
void driveX(int pwmCmd) {
  int pwm = abs(pwmCmd);
  pwm = constrain(pwm, 0, MAX_PWM);

  if (pwmCmd > 0) {
    digitalWrite(X_IN1, HIGH);
    digitalWrite(X_IN2, LOW);
    analogWrite(EN_X, pwm);
  } else if (pwmCmd < 0) {
    digitalWrite(X_IN1, LOW);
    digitalWrite(X_IN2, HIGH);
    analogWrite(EN_X, pwm);
  } else {
    analogWrite(EN_X, 0);
    digitalWrite(X_IN1, LOW);
    digitalWrite(X_IN2, LOW);
  }
}

void driveY(int pwmCmd) {
  int pwm = abs(pwmCmd);
  pwm = constrain(pwm, 0, MAX_PWM);

  if (pwmCmd > 0) {
    digitalWrite(Y_IN1, HIGH);
    digitalWrite(Y_IN2, LOW);
    analogWrite(EN_Y, pwm);
  } else if (pwmCmd < 0) {
    digitalWrite(Y_IN1, LOW);
    digitalWrite(Y_IN2, HIGH);
    analogWrite(EN_Y, pwm);
  } else {
    analogWrite(EN_Y, 0);
    digitalWrite(Y_IN1, LOW);
    digitalWrite(Y_IN2, LOW);
  }
}

void stopAllMotors() {
  driveX(0);
  driveY(0);
}

// ==============================
// Controller
// ==============================
int positionController(float errorMm, float kp) {
  if (abs(errorMm) <= DEADBAND_MM) {
    return 0;
  }

  int pwm = (int)(kp * abs(errorMm));

  if (pwm < MIN_PWM) pwm = MIN_PWM;
  if (pwm > MAX_PWM) pwm = MAX_PWM;

  if (errorMm > 0) return pwm;
  return -pwm;
}

void updateClosedLoop() {
  float xNow = readXmm();
  float yNow = readYmm();

  float errX = targetX_mm - xNow;
  float errY = targetY_mm - yNow;

  int cmdX = positionController(errX, KP_X);
  int cmdY = positionController(errY, KP_Y);

  driveX(cmdX);
  driveY(cmdY);
}

// ==============================
// Motion interface
// ==============================
void setTarget(float xMm, float yMm) {
  targetX_mm = clampFloat(xMm, 0.0, X_TRAVEL_MM);
  targetY_mm = clampFloat(yMm, 0.0, Y_TRAVEL_MM);
}

bool atTarget() {
  float xNow = readXmm();
  float yNow = readYmm();
  return (abs(targetX_mm - xNow) <= DEADBAND_MM &&
          abs(targetY_mm - yNow) <= DEADBAND_MM);
}

// Blocking move, good for first testing
void moveTo(float xMm, float yMm) {
  setTarget(xMm, yMm);

  unsigned long start = millis();
  const unsigned long timeoutMs = 8000;

  while (!atTarget()) {
    updateClosedLoop();

    if (millis() - start > timeoutMs) {
      break;
    }

    delay(CONTROL_PERIOD_MS);
  }

  stopAllMotors();
}

// ==============================
// Serial commands
// G x y     -> move to x,y in mm
// P         -> print current position
// C         -> print raw ADC values
// ==============================
void handleSerial() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0) return;

  if (line == "P") {
    Serial.print("Xmm: ");
    Serial.print(readXmm(), 2);
    Serial.print(", Ymm: ");
    Serial.println(readYmm(), 2);
    return;
  }

  if (line == "C") {
    Serial.print("Xadc: ");
    Serial.print(readXadc());
    Serial.print(", Yadc: ");
    Serial.println(readYadc());
    return;
  }

  if (line.startsWith("G")) {
  int firstSpace = line.indexOf(' ');
  int secondSpace = line.indexOf(' ', firstSpace + 1);

  if (firstSpace > 0 && secondSpace > firstSpace) {
    float xCmd = line.substring(firstSpace + 1, secondSpace).toFloat();
    float yCmd = line.substring(secondSpace + 1).toFloat();

    setTarget(xCmd, yCmd);

    Serial.print("Target set to ");
    Serial.print(targetX_mm, 2);
    Serial.print(", ");
    Serial.println(targetY_mm, 2);
  } else {
    Serial.println("Bad G command");
  }
}
}

// ==============================
// Setup / loop
// ==============================
void setup() {
  pinMode(EN_X, OUTPUT);
  pinMode(X_IN1, OUTPUT);
  pinMode(X_IN2, OUTPUT);

  pinMode(EN_Y, OUTPUT);
  pinMode(Y_IN1, OUTPUT);
  pinMode(Y_IN2, OUTPUT);

  stopAllMotors();

  Serial.begin(115200);
  delay(500);

  Serial.println("Closed-loop XY controller ready");
  Serial.println("Use commands:");
  Serial.println("  G x y");
  Serial.println("  P");
  Serial.println("  C");

  // Start at center
  setTarget(X_TRAVEL_MM / 2.0, Y_TRAVEL_MM / 2.0);
}

void loop() {
  handleSerial();

  unsigned long now = millis();

  if (now - lastControlMs >= CONTROL_PERIOD_MS) {
    lastControlMs = now;
    updateClosedLoop();
  }

  if (now - lastPrintMs >= PRINT_PERIOD_MS) {
    lastPrintMs = now;

    // Serial.print("Xadc: ");
    // Serial.print(readXadc());
    // Serial.print(", Yadc: ");
    // Serial.print(readYadc());
    // Serial.print(", Xmm: ");
    // Serial.print(readXmm(), 2);
    // Serial.print(", Ymm: ");
    // Serial.print(readYmm(), 2);
    // Serial.print(", TX: ");
    // Serial.print(targetX_mm, 2);
    // Serial.print(", TY: ");
    // Serial.println(targetY_mm, 2);
  }
}