#include <Wire.h>
#include <U8g2lib.h>
#include <Servo.h>

#define BUTTON_PIN      7
#define SERVO_HEAD_PIN  9
#define SERVO_ARM1_PIN  10
#define SERVO_ARM2_PIN  11
#define HEAD_CENTER     40
#define HEAD_NOD_DOWN   37
#define ARM1_REST       90
#define ARM1_UP         60
#define ARM1_WAVE_A     100
#define ARM1_WAVE_B     160
#define ARM2_REST       90
#define ARM2_UP         60
#define ARM2_WAVE_A     100
#define ARM2_WAVE_B     160

// Standard pulse-width range used by the Servo library (µs).
// Most hobby servos (SG90 etc.) are happy with 544-2400.
// If your servo doesn't reach full physical range at 0/180,
// narrow this (e.g. 600-2400) rather than changing the angle
// constants above, so "40 degrees" always means the same thing.
#define SERVO_MIN_US 544
#define SERVO_MAX_US 2400

U8G2_SH1106_128X64_NONAME_F_HW_I2C oled(U8G2_R0, U8X8_PIN_NONE);

Servo headServo;
Servo arm1Servo;
Servo arm2Servo;

char buf[32];
char currentEmotion[12] = "neutral";

// ── ROBOT STATE ────────────────────────────────────────
enum RobotState { STATE_LOADING, STATE_ACTIVE };
RobotState robotState = STATE_LOADING;

bool isListening = false;
bool isSpeaking  = false;

// Mouth animation timing
unsigned long lastMouthToggle = 0;
bool mouthOpen = false;
const unsigned long MOUTH_TOGGLE_MS = 140;

// Safety auto-stop in case SPEAK_END is never received
unsigned long speakStartTime = 0;
const unsigned long MAX_SPEAK_DURATION_MS = 8000;

// Loading screen animation timing
unsigned long lastLoadingFrame = 0;
int loadingDots = 0;
const unsigned long LOADING_FRAME_MS = 400;

// ── SERVO ─────────────────────────────────────────────
// Thin wrapper so the rest of the code (doGesture, setup, etc.)
// doesn't need to change — it still just calls servoWrite(pin, angle).
void servoWrite(int pin, int degrees) {
  degrees = constrain(degrees, 0, 180);

  if (pin == SERVO_HEAD_PIN) {
    headServo.write(degrees);
  } else if (pin == SERVO_ARM1_PIN) {
    arm1Servo.write(degrees);
  } else if (pin == SERVO_ARM2_PIN) {
    arm2Servo.write(degrees);
  }
}

// ── THICK LINE HELPER ─────────────────────────────────
// U8g2 has no lineWidth — draw N parallel lines for thickness
void thickLine(int x0, int y0, int x1, int y1, int t) {
  for (int i = -(t/2); i <= t/2; i++) {
    // Offset perpendicular to line direction
    bool steep = abs(y1-y0) > abs(x1-x0);
    if (steep) {
      oled.drawLine(x0+i, y0, x1+i, y1);
    } else {
      oled.drawLine(x0, y0+i, x1, y1+i);
    }
  }
}

// Thick filled ellipse outline — draw concentric ellipses
void thickEllipse(int cx, int cy, int rx, int ry, int t) {
  for (int i = 0; i < t; i++) {
    oled.drawEllipse(cx, cy, rx+i, ry+i);
  }
}

// ── FACE DRAWING ──────────────────────────────────────
void drawFace(const char* emotion, bool blink) {
  oled.clearBuffer();

  if (strcmp(emotion, "happy") == 0) {

    // Eyes — big filled discs with shine
    if (!blink) {
      oled.drawDisc(38, 24, 9);
      oled.drawDisc(90, 24, 9);
      // Shine dot (black cutout)
      oled.setDrawColor(0);
      oled.drawDisc(41, 21, 2);
      oled.drawDisc(93, 21, 2);
      oled.setDrawColor(1);
    } else {
      thickLine(28, 24, 48, 24, 4);
      thickLine(80, 24, 100, 24, 4);
    }

    // Big bold smile
    thickLine(28, 42, 34, 50, 4);
    thickLine(34, 50, 42, 55, 4);
    thickLine(42, 55, 52, 58, 4);
    thickLine(52, 58, 64, 59, 4);
    thickLine(64, 59, 76, 58, 4);
    thickLine(76, 58, 86, 55, 4);
    thickLine(86, 55, 94, 50, 4);
    thickLine(94, 50, 100, 42, 4);

  } else if (strcmp(emotion, "excited") == 0) {

    // Bold raised eyebrows
    thickLine(26, 10, 48, 5, 4);
    thickLine(80, 5, 102, 10, 4);

    // Big eyes
    if (!blink) {
      oled.drawDisc(38, 27, 10);
      oled.drawDisc(90, 27, 10);
      oled.setDrawColor(0);
      oled.drawDisc(42, 23, 3);
      oled.drawDisc(94, 23, 3);
      oled.setDrawColor(1);
    } else {
      thickLine(27, 27, 49, 27, 4);
      thickLine(79, 27, 101, 27, 4);
    }

    // Open oval mouth — thick outline
    thickEllipse(64, 50, 15, 9, 3);

  } else if (strcmp(emotion, "calm") == 0) {

    // Half-closed eyes
    if (!blink) {
      oled.drawDisc(38, 28, 8);
      oled.drawDisc(90, 28, 8);
      // Black eyelid over top half
      oled.setDrawColor(0);
      oled.drawBox(28, 18, 22, 11);
      oled.drawBox(80, 18, 22, 11);
      oled.setDrawColor(1);
      // Redraw bottom eyelid line thick
      thickLine(28, 29, 50, 29, 3);
      thickLine(80, 29, 102, 29, 3);
    } else {
      thickLine(28, 28, 50, 28, 4);
      thickLine(80, 28, 102, 28, 4);
    }

    // Gentle slight smile
    thickLine(40, 46, 52, 51, 3);
    thickLine(52, 51, 64, 52, 3);
    thickLine(64, 52, 76, 51, 3);
    thickLine(76, 51, 88, 46, 3);

  } else if (strcmp(emotion, "concerned") == 0) {

    // Worried inward brows — thick and angled
    thickLine(26, 12, 50, 20, 4);
    thickLine(78, 20, 102, 12, 4);

    // Eyes
    if (!blink) {
      oled.drawDisc(38, 31, 8);
      oled.drawDisc(90, 31, 8);
      oled.setDrawColor(0);
      oled.drawDisc(41, 28, 2);
      oled.drawDisc(93, 28, 2);
      oled.setDrawColor(1);
    } else {
      thickLine(28, 31, 48, 31, 4);
      thickLine(80, 31, 100, 31, 4);
    }

    // Bold frown
    thickLine(28, 56, 34, 47, 4);
    thickLine(34, 47, 42, 41, 4);
    thickLine(42, 41, 52, 38, 4);
    thickLine(52, 38, 64, 37, 4);
    thickLine(64, 37, 76, 38, 4);
    thickLine(76, 38, 86, 41, 4);
    thickLine(86, 41, 94, 47, 4);
    thickLine(94, 47, 100, 56, 4);

  } else {
    // neutral
    if (!blink) {
      oled.drawDisc(38, 26, 8);
      oled.drawDisc(90, 26, 8);
      oled.setDrawColor(0);
      oled.drawDisc(41, 23, 2);
      oled.drawDisc(93, 23, 2);
      oled.setDrawColor(1);
    } else {
      thickLine(28, 26, 48, 26, 4);
      thickLine(80, 26, 100, 26, 4);
    }
    // Double thick flat mouth
    thickLine(40, 46, 88, 46, 4);
  }

  // ── OVERLAYS: listening ears / speaking mouth ────────
  // Drawn on top of whichever emotion face was just rendered.
  if (isListening) {
    drawEars();
  }

  if (isSpeaking) {
    drawTalkingMouth();
  }

  oled.sendBuffer();
}

// ── LOADING SCREEN ─────────────────────────────────────
void drawLoadingScreen() {
  oled.clearBuffer();
  oled.setFont(u8g2_font_7x13_tf);
  oled.drawStr(26, 34, "Loading");

  char dots[4] = "   ";
  for (int i = 0; i < loadingDots; i++) dots[i] = '.';
  dots[3] = '\0';
  oled.drawStr(94, 34, dots);

  oled.sendBuffer();
}

void updateLoadingScreen() {
  unsigned long now = millis();
  if (now - lastLoadingFrame > LOADING_FRAME_MS) {
    lastLoadingFrame = now;
    loadingDots = (loadingDots + 1) % 4;
    drawLoadingScreen();
  }
}

// ── LISTENING EARS ─────────────────────────────────────
// Simple pointed "ear" shapes above each side of the face.
void drawEars() {
  // Left ear
  oled.drawTriangle(4, 22, 16, 2, 26, 20);
  // Right ear
  oled.drawTriangle(124, 22, 112, 2, 102, 20);
}

// ── TALKING MOUTH ──────────────────────────────────────
// Overrides whatever static mouth the current emotion drew,
// alternating open/closed to look like speech.
void drawTalkingMouth() {
  // Clear the mouth region first (sits below eye-level for all emotions)
  oled.setDrawColor(0);
  oled.drawBox(25, 39, 80, 23);
  oled.setDrawColor(1);

  if (mouthOpen) {
    oled.drawFilledEllipse(64, 50, 14, 9);
  } else {
    thickLine(40, 48, 88, 48, 4);
  }
}

void updateMouthAnimation() {
  if (!isSpeaking) return;

  unsigned long now = millis();

  // Safety auto-stop if SPEAK_END never arrives
  if (now - speakStartTime > MAX_SPEAK_DURATION_MS) {
    isSpeaking = false;
    drawFace(currentEmotion, false);
    return;
  }

  if (now - lastMouthToggle > MOUTH_TOGGLE_MS) {
    lastMouthToggle = now;
    mouthOpen = !mouthOpen;
    drawFace(currentEmotion, false);
  }
}

// ── BLINK ─────────────────────────────────────────────
unsigned long nextBlink = 0;
unsigned long blinkAt   = 0;
bool eyesOpen = true;

void updateBlink() {
  unsigned long now = millis();
  if (eyesOpen && now >= nextBlink) {
    eyesOpen = false;
    blinkAt  = now;
    drawFace(currentEmotion, true);
  } else if (!eyesOpen && now - blinkAt > 130) {
    eyesOpen = true;
    nextBlink = now + 2000 + random(2500);
    drawFace(currentEmotion, false);
  }
}

// ── SETUP ─────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  delay(500);

  // Attach servos with an explicit pulse-width range so the
  // angle-to-pulse mapping is fixed and known (instead of
  // depending on whatever the manual bit-bang loop happened to do).
  headServo.attach(SERVO_HEAD_PIN, SERVO_MIN_US, SERVO_MAX_US);
  arm1Servo.attach(SERVO_ARM1_PIN, SERVO_MIN_US, SERVO_MAX_US);
  arm2Servo.attach(SERVO_ARM2_PIN, SERVO_MIN_US, SERVO_MAX_US);

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  servoWrite(SERVO_HEAD_PIN, HEAD_CENTER);
  servoWrite(SERVO_ARM1_PIN, ARM1_REST);
  servoWrite(SERVO_ARM2_PIN, ARM2_REST);

  Wire.begin();
  oled.begin();
  randomSeed(analogRead(0));
  nextBlink = 2500 + random(2000);

  strcpy(currentEmotion, "neutral");

  // Boot straight into the loading screen — Python sends
  // "READY" once the model has finished loading.
  robotState = STATE_LOADING;
  drawLoadingScreen();

  Serial.println("ARDUINO_READY");
}

// ── LOOP ──────────────────────────────────────────────
void loop() {

  if (robotState == STATE_LOADING) {
    updateLoadingScreen();
  } else {
    if (isSpeaking) {
      updateMouthAnimation();
    } else {
      updateBlink();
    }

    if (digitalRead(BUTTON_PIN) == LOW) {
      Serial.println("BUTTON PRESSED");
      delay(300);
    }
  }

  if (Serial.available()) {
    memset(buf, 0, sizeof(buf));
    int len = Serial.readBytesUntil('\n', buf, sizeof(buf) - 1);
    if (len > 0 && buf[len-1] == '\r') { buf[len-1] = '\0'; len--; }
    if (len == 0) return;

    Serial.print("GOT: "); Serial.println(buf);

    if (strcmp(buf, "LOADING") == 0) {
      robotState = STATE_LOADING;
      loadingDots = 0;
      drawLoadingScreen();
    }
    else if (strcmp(buf, "READY") == 0) {
      robotState = STATE_ACTIVE;
      isListening = false;
      isSpeaking  = false;
      setEmotion("neutral");
    }
    else if (strcmp(buf, "LISTEN_ON") == 0) {
      isListening = true;
      drawFace(currentEmotion, false);
    }
    else if (strcmp(buf, "LISTEN_OFF") == 0) {
      isListening = false;
      drawFace(currentEmotion, false);
    }
    else if (strcmp(buf, "SPEAK_START") == 0) {
      isSpeaking = true;
      mouthOpen = false;
      lastMouthToggle = millis();
      speakStartTime  = millis();
      drawFace(currentEmotion, false);
    }
    else if (strcmp(buf, "SPEAK_END") == 0) {
      isSpeaking = false;
      drawFace(currentEmotion, false);
    }
    else if (strcmp(buf, "happy") == 0)     { setEmotion("happy");     doGesture("nod");   }
    else if (strcmp(buf, "excited") == 0)   { setEmotion("excited");   doGesture("cheer"); }
    else if (strcmp(buf, "calm") == 0)      { setEmotion("calm");      doGesture("idle");  }
    else if (strcmp(buf, "neutral") == 0)   { setEmotion("neutral");   doGesture("idle");  }
    else if (strcmp(buf, "concerned") == 0) { setEmotion("concerned"); doGesture("nod");   }
    else if (strncmp(buf, "GESTURE:", 8) == 0) { doGesture(buf + 8); }
    else if (strncmp(buf, "SPEAK:", 6) == 0)   {
      // Backward-compatible: still works even without SPEAK_START/SPEAK_END,
      // auto-stops via MAX_SPEAK_DURATION_MS if no explicit end arrives.
      setEmotion("happy");
      isSpeaking = true;
      mouthOpen = false;
      lastMouthToggle = millis();
      speakStartTime  = millis();
    }
    else if (strcmp(buf, "test") == 0)      { runHardwareTest(); }
    else if (strcmp(buf, "END") == 0)       {
      returnToRest();
      isListening = false;
      isSpeaking  = false;
      setEmotion("neutral");
    }
    else { Serial.print("UNKNOWN: "); Serial.println(buf); }
  }
}

void setEmotion(const char* emotion) {
  strncpy(currentEmotion, emotion, 11);
  currentEmotion[11] = '\0';
  eyesOpen = true;
  nextBlink = millis() + 2000 + random(2000);
  drawFace(currentEmotion, false);
  Serial.print("FACE: "); Serial.println(currentEmotion);
}

void doGesture(const char* gesture) {
  Serial.print("GESTURE: "); Serial.println(gesture);

  if (strcmp(gesture, "nod") == 0) {
    servoWrite(SERVO_HEAD_PIN, HEAD_NOD_DOWN); delay(300);
    servoWrite(SERVO_HEAD_PIN, HEAD_CENTER);   delay(300);
    servoWrite(SERVO_HEAD_PIN, HEAD_NOD_DOWN); delay(300);
    servoWrite(SERVO_HEAD_PIN, HEAD_CENTER);

  } else if (strcmp(gesture, "wave") == 0) {
    servoWrite(SERVO_ARM2_PIN, ARM2_WAVE_A); delay(300);
    servoWrite(SERVO_ARM2_PIN, ARM2_WAVE_B); delay(300);
    servoWrite(SERVO_ARM2_PIN, ARM2_WAVE_A); delay(300);
    servoWrite(SERVO_ARM2_PIN, ARM2_WAVE_B); delay(300);
    servoWrite(SERVO_ARM2_PIN, ARM2_REST);

  } else if (strcmp(gesture, "shake") == 0) {
    servoWrite(SERVO_ARM1_PIN, ARM1_WAVE_A);
    servoWrite(SERVO_ARM2_PIN, ARM2_WAVE_B); delay(300);
    servoWrite(SERVO_ARM1_PIN, ARM1_WAVE_B);
    servoWrite(SERVO_ARM2_PIN, ARM2_WAVE_A); delay(300);
    servoWrite(SERVO_ARM1_PIN, ARM1_WAVE_A);
    servoWrite(SERVO_ARM2_PIN, ARM2_WAVE_B); delay(300);
    servoWrite(SERVO_ARM1_PIN, ARM1_REST);
    servoWrite(SERVO_ARM2_PIN, ARM2_REST);

  } else if (strcmp(gesture, "cheer") == 0) {
    servoWrite(SERVO_ARM1_PIN, ARM1_UP);
    servoWrite(SERVO_ARM2_PIN, ARM2_UP);   delay(400);
    servoWrite(SERVO_ARM1_PIN, ARM1_REST);
    servoWrite(SERVO_ARM2_PIN, ARM2_REST); delay(300);
    servoWrite(SERVO_ARM1_PIN, ARM1_UP);
    servoWrite(SERVO_ARM2_PIN, ARM2_UP);   delay(400);
    servoWrite(SERVO_ARM1_PIN, ARM1_REST);
    servoWrite(SERVO_ARM2_PIN, ARM2_REST);

  } else if (strcmp(gesture, "point") == 0) {
    servoWrite(SERVO_ARM1_PIN, ARM1_UP);   delay(300);
    servoWrite(SERVO_ARM1_PIN, ARM1_REST);

  } else {
    returnToRest();
  }
}

void returnToRest() {
  servoWrite(SERVO_HEAD_PIN, HEAD_CENTER);
  servoWrite(SERVO_ARM1_PIN, ARM1_REST);
  servoWrite(SERVO_ARM2_PIN, ARM2_REST);
  Serial.println("REST");
}

void runHardwareTest() {
  Serial.println("TEST: starting...");
  const char* emos[] = {"happy","excited","calm","concerned","neutral"};
  for (int i = 0; i < 5; i++) {
    setEmotion(emos[i]);
    for (int b = 0; b < 2; b++) {
      delay(700);
      drawFace(currentEmotion, true);  delay(130);
      drawFace(currentEmotion, false);
    }
  }
  const char* gestures[] = {"nod","wave","shake","cheer","point"};
  for (int i = 0; i < 5; i++) {
    Serial.print("Testing: "); Serial.println(gestures[i]);
    doGesture(gestures[i]); delay(300);
  }
  returnToRest();
  setEmotion("neutral");
  Serial.println("TEST: complete");
}
