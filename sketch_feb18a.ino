/*
  Serial Plotter:
  -> Voltage plotted
  -> Time shown as number only (not plotted)
*/

const int analogPin = A0;
const float referenceVoltage = 5.0;
const unsigned long samplingInterval = 10; // 100 Hz

unsigned long startTime;
unsigned long lastSampleTime = 0;

void setup() {
  Serial.begin(9600);
  startTime = millis();
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= samplingInterval) {
    lastSampleTime = currentTime;

    int rawADC = analogRead(analogPin);
    float voltage = (rawADC * referenceVoltage) / 1023.0;
    float timeSeconds = (currentTime - startTime) / 1000.0;

    // Convert time to STRING (so Plotter ignores it)
    String timeString = String(timeSeconds, 3);

    Serial.print(timeString);
    Serial.print(" ");      // space separator
    Serial.println(voltage, 4);
  }
}