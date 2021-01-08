
//Include all subfiles of the project
#include "Global.h" //include file containing global variables
#include "Helpers.h" // include file containing helpers function
#include "hardware_helpers.h" 
#include "setup_Helpers.h"
#include "OSC_Helpers.h" //include file containing helpers function regardind OSC
#include "update_helpers.h"
#include "recording_helpers.h"



// 0 - 
// 1 - 
// 2 - Button 0 Start recording
// 3 - Button 1 Place marker
// 4 - SD_ETHERNET ADAPTER
// 5 - EMPTY
// 6 - EMPTY
// 7 - EMPTY
// 8 - SD_ETHERNET ADAPTER
// 9 - SD_ETHERNET ADAPTER
// 10 - SD_ETHERNET ADAPTER
// 11 - SD_ETHERNET ADAPTER
// 12 - SD_ETHERNET ADAPTER
// 13 - SD_ETHERNET ADAPTER
// 14 - (A0) Selection potentiometer
// 15 - (A1) Heart
// 16 - (A2) GSR
// 17 - (A3) Temp
// 18 - (A4) LCD SCREEN
// 19 - (A5) LCD SCREEN
// 20 - (A6) GSR2
// 21 - EMPTY
// 22 - EMPTY
// 23 - EMPTY

// VIN - SD_ETHERNET ADAPTER
// GND - SD_ETHERNET ADAPTER



// TO - DO :::
// SEND DATA VIA UDP @ 1000HZ IF POSSIBLE
// BEING ABLE TO RECORD MULTIPLE SESSIONS WITHOUT DISCONNECTING THE ARDUINO





void setup() {

  Serial.begin(9600);

  udpSetup(); //set up the UDP connection
  setupAllSensors(); // restart all the sensors to initial state
  checkForCard(); //check if a SD card is inserted
  cardInfo(); //display the informations of the inserted sd card
  setupButtons(1); //setup all the buttons and set the refresh rate at 1ms
  lcdSetup(); //setup the lcd screen

}

void loop() {

  updateAllSensors(); //update the sensors every loop
  updatePotentiometer(); //update the potentiometer value
  updateButtons(); //update all the buttons state
  oscUpdate(); //look if an osc message arrived and parse it
  updateLCD(); //update lcd display buffers


  if (r.isIdle() ) //verify if its time to stop the recording
    {
      idleDisplay();
    
      if ( startButton.fell()) 
        {
          r.startCountdown(); //start the 10 seconds countdown before the recording starts
          displayIndex = 1; //switch the idle display mode
        }

      if (r.updateCountdown() == true) 
        {
          //this happens when the countdown is over
          r.stopCountdown(); 
          r.resetCountdown();
          r.setupRecording(); //Give permission to setup recording in the next loop
          recordingLCDIndex = 0; //make sure the recording animation start at 0
        }
    }



  if (r.isReadyToStop() ) //verify if its time to stop the recording
    {
      if (r.endDelayNotStarted) 
        {
          r.startEndDelay(); //start delay to display info on lcd
          lcdRecordingOver.toCharArray(lcdLine1, 17);
        }

      if (r.updateEndDelay() == true) 
        {
          r.stopEndDelay(); 
          endRecordingSession(); //goes trought all the steps to end the recording
        }
    }



  if (r.isReadyToStart() && fileOpen == false) //verify if it can start the recording
  {
    String nameInfo = infoEmotion + " " + filename[3] +filename[4] +filename[5] + " Laurence"; //compose the subjectName header line
    r.setSubjectName(nameInfo);
    r.setSignals(signalTypes);
    setupRecording(); //goes trought all the steps to setup and start the recording



  }


  if ( r.isRecording() ) //verify if it's in recording states
  {
    
    if ( startButton.fell()) //verify if the stop button was pressed
      {
        r.stopProcess = true;
      }
      
    recordingDisplay(); 


    noInterrupts(); //prevents from interrupting until interrupts() is called to transfert the buffer

    if (bufferA.isFull() && readyToWrite == false) //verify if it's ready to transfer the buffer
    {
      transferBuffer(); //transfer the buffer to the temporary write buffer
    }
    interrupts(); //permits interrupts again

    if (readyToWrite == true ) //verify if write buffer is ready to be written to SD card
    {
      writeToCard(); //write the temps buffer to the SD card
    }
  }

}
