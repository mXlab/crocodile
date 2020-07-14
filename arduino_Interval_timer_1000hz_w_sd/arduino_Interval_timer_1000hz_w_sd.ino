
#include "Global.h" //include file containing global variables
#include "OSC_Helpers.h" //include file containing helpers function regardind OSC
#include "Helpers.h" // include file containing helpers function



// TO - DO :::
// INTEGRATE MARKERS
// SEND DATA VIA UDP @ 1000HZ IF POSSIBLE
// BEING ABLE TO RECORD MULTIPLE SESSIONS WITHOUT DISCONNECTING THE ARDUINO


void setup() {
  //recommendedSetup(); //recommended steps to setup PJRC SD  and WIZ820io adapter
  Serial.begin(9600);

  udpSetup(); //set up the UDP connection
  setupAllSensors(); // restart all the sensors to initial state
  checkForCard(); //check if a SD card is inserted
  cardInfo(); //display the informations of the inserted sd card



}

void loop() {
  //noInterrupts();
  
  
  //interrupts();
  updateAllSensors(); //update the sensors every loop
  oscUpdate();


  if (r.isReadyToStop() ) //verify if its time to stop the recording
  {
    endRecordingSession(); //goes trought all the steps to end the recording
  }


  if (r.isReadyToStart() && fileOpen == false) //verify if it can start the recording
  {
    setupRecording(); //goes trought all the steps to setup and start the recording
  }


  if ( r.isRecording() ) //verify if it's in recording states
  {

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
