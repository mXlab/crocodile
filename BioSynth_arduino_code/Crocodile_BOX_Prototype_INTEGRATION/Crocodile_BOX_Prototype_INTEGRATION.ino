
//Include all subfiles of the project
#include "Global.h" //include file containing global variables


//Encoder dependencies
//#define ENCODER_DO_NOT_USE_INTERRUPTS
#define ENCODER_OPTIMIZE_INTERRUPTS
#include <Encoder.h>
#define ENCODER_PHASE_A 5
#define ENCODER_PHASE_B 6
#define ENCODER_SWITCH 2
Encoder myEnc(ENCODER_PHASE_A, ENCODER_PHASE_B);
long oldPosition = -999;

#include "Helpers.h" // include file containing helpers function
#include "hardware_helpers.h"
#include "setup_Helpers.h"
//#include "OSC_Helpers.h" //include file containing helpers function regarding OSC -- DONT NEED FOR BOX
#include "update_helpers.h"
#include "recording_helpers.h"

// 0 - DISP RESET
// 1 - SERIAL SEND
// 2 - Encoder Button / StartStop recording 
// 3 - FootPedal Button / Place marker 
// 4 - SD_ETHERNET ADAPTER
// 5 - ENCODER A 
// 6 - ENCODER B 
// 7 - MOSI 
// 8 - SD_ETHERNET ADAPTER / DISP DO 
// 9 - SD_ETHERNET ADAPTER
// 10 - SD ETHERNET - CS PIN -  
// 11 - SD_ETHERNET ADAPTER
// 12 - SD_ETHERNET ADAPTER / DISP MISO 
// 13 - SD_ETHERNET ADAPTER
// 14 - (A0) DISP_SCK 
// 15 - (A1) DISP_CS (map to pot in audio designs)
// 16 - (A2) GSR2
// 17 - (A3) Temp
// 18 - (A4) DISP SCL
// 19 - (A5) DISP SDA
// 20 - (A6) GSR1 (PWM)
// 21 - (A7) PULSE (PWM)
// 22 - Audio
// 23 - Audio
// 25 - Resp LED
// 32 - GSR LED 

// VIN - SD_ETHERNET ADAPTER
// GND - SD_ETHERNET ADAPTER



// TO - DO :::
// SEND DATA VIA UDP @ 1000HZ IF POSSIBLE
// BEING ABLE TO RECORD MULTIPLE SESSIONS WITHOUT DISCONNECTING THE ARDUINO
//FOOT PEDAL gets stuck

void setup() {

  Serial.begin(9600);

  ///udpSetup(); //set up the UDP connection NO NEED OF UDP FOR THE BOX
  setupAllSensors(); // restart all the sensors to initial state

  checkForCard(); //check if a SD card is inserted NO NEED TO RUN FOR PROTOTYPE
  cardInfo(); //display the informations of the inserted sd card NO NEED TO RUN FOR PROTOTYPE
 
  setupButtons(1); //setup all the buttons and set the refresh rate at 1ms
  lcdSetup(); //setup the lcd screen
  pinMode(LED_HEART, OUTPUT);
  pinMode(LED_GSR1, OUTPUT);
  pinMode(LED_GSR2, OUTPUT);
  pinMode(LED_TEMP, OUTPUT);
  //digitalWrite(7,HIGH);

  lcdUpdate.restart();
}

void loop() {

  updateButtons(); //update all the buttons state
   
  if( r.isRecording() == false){ //update encoder only when not recording
    updatePotentiometer(); //update the potentiometer value
  };

  if (r.isIdle() ) //verify if its time to stop the recording
  {
    //Serial.println("idle state"); //debug
    idleDisplay();

    if ( startButton.fell())
    {
      //Serial.println("start button pressed"); //debug
      r.startCountdown(); //start the 10 seconds countdown before the recording starts
      displayIndex = 1; //switch the idle display mode
    }

    if (r.updateCountdown() == true)
    {
      //this happens when the countdown is over
      //Serial.println("countdown over"); //debug
      r.stopCountdown();
      r.resetCountdown();
      //Serial.println("Setting up recording"); //debug
      r.setupRecording(); //Give permission to setup recording in the next loop
      recordingLCDIndex = 0; //make sure the recording animation start at 0
    }

  }
  else if (r.isReadyToStop() ) //verify if its time to stop the recording
  {
    //Serial.println("stopping recording"); //debug
    if (r.endDelayNotStarted)
    {
      r.startEndDelay(); //start delay to display info on lcd
      //Serial.println("end delay"); //debug
      sprintf(lcdLine1, "%s", lcdRecordingOver);

    }

    if (r.updateEndDelay() == true)
    {
      //Serial.println("update end delay"); //debug
      r.stopEndDelay();
      //Serial.println("end recording session"); //debug
      endRecordingSession(); //goes trought all the steps to end the recording
    }
  }


  else  if (r.isReadyToStart() && fileOpen == false) //verify if it can start the recording
  {
    String nameInfo = infoEmotion + " " + filename[3] + filename[4] + filename[5] + " Laurence"; //compose the subjectName header line
    r.setSubjectName(nameInfo);
    r.setSignals(signalTypes);
    setupRecording(); //goes through all the steps to setup and start the recording
  }


  else  if ( r.isRecording() ) //verify if it's in recording state
  {
   // Serial.println("recording state"); //debug
    if ( startButton.fell()) //verify if the stop button was pressed
    {
    //  Serial.println("stopped button pressed"); //debug
      r.stopProcess = true;
    }
    //Serial.println("recording display"); //debug
    recordingDisplay();

    //Serial.println("before interupts"); //debug
    noInterrupts(); //prevents from interrupting until interrupts() is called to transfer to the buffer
  //  Serial.println("inside interrupt"); //debug
    updateAllSensors(); //update the sensors every loop
    if (bufferA.isFull() && readyToWrite == false) //verify if it's ready to transfer the buffer
    {
    //  Serial.println("transfer buffer"); //debug
      transferBuffer(); //transfer the buffer to the temporary write buffer
    }
   // Serial.println("Enterupt end"); //debug
    interrupts(); //permits interrupts again
   // Serial.println("outside interrupts"); //debug

    if (readyToWrite == true ) //verify if write buffer is ready to be written to SD card
    {
    //  Serial.println("writting to card"); //debug
      writeToCard(); //write the temps buffer to the SD card
    }
  }
 // Serial.println("before update lcd"); //debug
if(lcdUpdate.hasPassed(46))
 {  lcdUpdate.restart();
 updateLCD(); //update lcd display buffers

  
}
  //Serial.println("loop end"); //debug
}
