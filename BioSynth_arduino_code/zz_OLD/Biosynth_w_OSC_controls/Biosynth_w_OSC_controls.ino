
#include "Global.h" //include file containing global variables
#include "OSC_Helpers.h" //include file containing helpers function regardind OSC
#include "Helpers.h" // include file containing helpers function

//USED PINS 

// 0 - Button 0
// 1 - Button 1
// 2 - Button 2

// 4 - SD_ETHERNET ADAPTER
// 8 - SD_ETHERNET ADAPTER
// 9 - SD_ETHERNET ADAPTER
// 10 - SD_ETHERNET ADAPTER
// 11 - SD_ETHERNET ADAPTER
// 12 - SD_ETHERNET ADAPTER
// 13 - SD_ETHERNET ADAPTER
// 14 - (A0) Heart

// 16 - (A2) GSR
// 17 - (A3) Temp
// 18 - Button 3
// 19 - Button 4
// 20 - Button 5
// 21 - Start button
// 22 - Stop button


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
  setupButtons(10);


}

void loop() {

  
  updateAllSensors(); //update the sensors every loop
  updateButtons();
  oscUpdate();
  
 if (r.isIdle() ) //verify if its time to stop the recording
  {
    if( startButton.fell()){
      r.setupRecording();
      }
    
  }

  if (r.isReadyToStop() ) //verify if its time to stop the recording
  {
    
    endRecordingSession(); //goes trought all the steps to end the recording
  }


  if (r.isReadyToStart() && fileOpen == false) //verify if it can start the recording
  {
    checkFileName();
    setupRecording(); //goes trought all the steps to setup and start the recording
  }


  if ( r.isRecording() ) //verify if it's in recording states
  {
    //CHECK IF CAN PUT IF STOP BUTTON PUSHED HERE.
    if( stopButton.fell()){
      r.stopProcess = true;
      }

    
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
