
#include "Global.h"; //include file containing global variables
#include "Helpers.h"; // include file containing helpers function
#include "Udp_helpers.h";


void setup() {

  delay(2000); //delay for debbuging and manipulation, remove when R&D over
  recommendedSetup(); //recommended steps to setup PJRC SD  and WIZ820io adapter
  Serial.begin(9600);
  udpSetup();
  setupAllSensors(); // restart all the sensors to initial state
  testClass(); //test methods of recording classes, remove when UDP communication implemented
  checkForCard(); //check if a SD card is inserted
  cardInfo(); //display the informations of the inserted sd card



}

void loop() {
  updateAllSensors(); //update the sensors every loop
  udpUpdate();
  parseMessage();
  clearBuffer();
  

  if (looped == maxLoop && r.isReadyToStop() ) //verify if its time to stop the recording 
  { 
     endRecordingSession(); //goes trought all the steps to end the recording 
  }

  
  
  if (r.isRecording() == false && fileOpen == false) //verify if it can start the recording
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
