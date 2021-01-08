/////////////////////////////////Recording helpers//////////////////////////////////
/*
    This file contains all helpers function for the recording class. These could potentially be added to the class

    -setupRecording()
    -endRecordingSession()
    -recordingDisplay()

*/
//------------------------------------------------------------------------------------------------

void lcdSetup() {
  lcd.init();                      // initialize the lcd
  lcd.backlight();
}

//------------------------------------------------------------------------------------------------

void setupButtons(int intervalms) {
  //initialize all the buttons
  startButton.attach(START_BUTTON_PIN , INPUT_PULLUP);
  startButton.interval(intervalms);
  
  markerButton.attach(MARKER_BUTTON_PIN , INPUT_PULLUP);
  markerButton.interval(intervalms);

}

//------------------------------------------------------------------------------------------------

void setupAllSensors() {
  // this function runs the necessary code to setup the sensors

  heart.reset();
  sc1.reset();
  resp.reset();

}

//------------------------------------------------------------------------------------------------

void udpSetup() {
  //This function runs the necessary code to setup the UDP connection

  Ethernet.init(CS_PIN);    // You can use Ethernet.init(pin) to configure the CS pin
  Ethernet.begin(mac, ip);  // start the Ethernet
  detectHardware();
  detectCable();
  Udp.begin(localPort); //start UDP
}
