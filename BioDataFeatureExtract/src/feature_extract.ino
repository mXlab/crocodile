
#include "CsvReader.h"
#include "CsvWriter.h"
#include "BioData.h" // inclue BioData Respiration module

using namespace pq; // use namespace pq to access Plaquette objects in Arduino IDE

Metro printerMetro (0.1); // print every 0.1 seconds

std::vector<int> values = { 0, 0, 0 };

unsigned long simulatedMicros = 0;
unsigned long microsCallback() {
  return simulatedMicros;
}

// Create instance for sensor 
//  Argument 1 : function to get external ADC value 
//  Argument 2 (optional): sampling rate (default = 50 Hz)
Heart heart([]() -> int { return values[0]; });
SkinConductance eda([]() -> int { return values[1]; });
Respiration resp([]() -> int { return values[2]; });

CsvReader reader("./data/crocodile_session_recording_04.csv");
CsvWriter writer("./data/crocodile_session_features_04.csv");

constexpr unsigned long FILE_SAMPLE_RATE = 100; // sample rate in Hz
constexpr unsigned long MICROS_PER_STEP  = 1000000UL / FILE_SAMPLE_RATE;

void setup() {
 // Setup Plaquette
  Plaquette.referenceClock(microsCallback);
  Plaquette.begin();

 // Initialize serial port
  Serial.begin(9600); 
  
 // Initialize sensor.
  resp.reset();

    std::vector<std::string> header;

    header.push_back("heart_normalized");
    header.push_back("heart_bmp");
    header.push_back("heart_amplitude_change");
    header.push_back("heart_bpm_change");

    header.push_back("eda_scr");
    header.push_back("eda_scl");

    header.push_back("resp_normalized");
    header.push_back("resp_scaled");
    header.push_back("resp_exhaling");

    header.push_back("resp_normalized_amplitude");
    header.push_back("resp_scaled_amplitude");
    header.push_back("resp_amplitude_level");
    header.push_back("resp_amplitude_change");
    header.push_back("resp_amplitude_variability");

    header.push_back("resp_rpm");
    header.push_back("resp_normalized_rpm");
    header.push_back("resp_scaled_rpm");
    header.push_back("resp_rpm_level");
    header.push_back("resp_rpm_change");
    header.push_back("resp_rpm_variability");

    // Assuming writer is an instance of CsvWriter
    writer.writeHeader(header);
}


void loop() {
  // Call a Plaquette step at every loop
  Plaquette.step();

  if (reader.nextRow(values))  {
    // Update sensor. 
    heart.update();
    eda.update();
    resp.update();

    // Output values.
    std::vector<float> row;

    row.push_back(heart.getNormalized());
    row.push_back(heart.getBPM());
    row.push_back(heart.amplitudeChange());
    row.push_back(heart.bpmChange());

    row.push_back(eda.getSCR());
    row.push_back(eda.getSCL());

    row.push_back(resp.getNormalized());
    row.push_back(resp.getScaled());
    row.push_back(resp.isExhaling() ? 1.0f : 0.0f);  // convert bool to float

    row.push_back(resp.getNormalizedAmplitude());
    row.push_back(resp.getScaledAmplitude());
    row.push_back(resp.getAmplitudeLevel());
    row.push_back(resp.getAmplitudeChange());
    row.push_back(resp.getAmplitudeVariability());

    row.push_back(resp.getRpm());
    row.push_back(resp.getNormalizedRpm());
    row.push_back(resp.getScaledRpm());
    row.push_back(resp.getRpmLevel());
    row.push_back(resp.getRpmChange());
    row.push_back(resp.getRpmVariability());

    // Assuming writer is an instance of CsvWriter
    writer.writeRow(row);

    simulatedMicros += MICROS_PER_STEP;
 }
 else
  exit(0);


}   