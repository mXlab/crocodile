

void updateData() {
  /*
   * This is the function that is run by the interval timer
   * It samples the data from the sensor and push it to the buffer
   * Sending the pushed data via OSC to the computer should be done here
   */
  unsigned long temp = millis()- stamp;
  int marker;

  if( r.marker == true){
    marker = 1;
    r.resetMarkerBool();
    }else{
      marker = 0 ;
    }
 
  int temp1 = 6;
  int temp2 = 5;
  int temp3 = 4;
  //pushing temporary data for development
  bufferA.push(temp);
  bufferA.push(pressedButton);
  bufferA.push(temp1);
  bufferA.push(temp2);
  bufferA.push(temp3);
 
// pushing sampled data from the sensor
//  bufferA.push(heart.getRaw());
//  bufferA.push(sc1.getRaw());
//  bufferA.push(resp.getRaw());

}
//------------------------------------------------------------------------------------------------

void checkFileName(){  

   
    while( filenameAvailable == false ){
         //check if file already exist
        if(SD.exists(filename)){
        // if yes, make a different name
          char endDigit = filename[5];
          char middleDigit = filename[4];
          char startDigit = filename[3];

     
          if( endDigit >= 57){
            
            endDigit = 48; 
            middleDigit = middleDigit + 1;
            
            if(middleDigit >= 57){
              
              middleDigit = 48;
              startDigit = startDigit + 1;
              
              }
          }else{

          endDigit = endDigit +1;
          
          }

          filename[5] = endDigit;
          filename[4] = middleDigit;
          filename[3] = startDigit;
     
        }else {
          Serial.print("New filename: ");
          Serial.print(filename);
          Serial.println();
          filenameAvailable = true;
          }
      
      }
  
  }
//------------------------------------------------------------------------------------------------


void setupRecording(){
// this function runs the necessary code to setup the recording before it starts
  
    Serial.println("Open File");
    recFile = SD.open( filename , FILE_WRITE);
    fileOpen = true;
    
    r.startRecording();
    r.headerPrinted = false;
    Serial.println("Start recording");
    captureData.begin( updateData, 1000);
    stamp = millis();
}

//------------------------------------------------------------------------------------------------

void datalog(int bufferArg[BUFFER_SIZE]) {
  // This function formats the data and log it to the file on the sd card
  
  int numChan = NUM_SIGNALS + 2; // +2 because of the timestamp and marker columns

  int formatBuffer[NUM_SIGNALS] = {0};
  unsigned long timestamp = 0 ;
  int marker = 0 ; 
 
  
  if (recFile) {

    for ( int i = 0 ; i < BUFFER_SIZE ; i++) {

      int formatIndex = i % numChan;
      
      if (formatIndex == 0 ) {
        timestamp = bufferArg[i];
      } else if(formatIndex == 1) {
        marker =  bufferArg[i];
        }
      else {
        
        formatBuffer[formatIndex - 2] = bufferArg[i];
      }
      if ( formatIndex == numChan - 1) {

        recFile.println(r.formatData(timestamp,marker, formatBuffer));

      }
    }

    recFile.flush();
  } else {
    Serial.println("cant write to file");
  }
}


//------------------------------------------------------------------------------------------------
 
 void writeToCard(){
      // write the data to the sd card
      
      datalog(writeBuffer);
  
      if (r.stopProcess) {

        Serial.println("Stop sensors");
        captureData.end();
        bufferA.clear();
        r.stopRecording();
      }


      readyToWrite = false;
      } 

//------------------------------------------------------------------------------------------------

void transferBuffer(){
  //transfer bufferA to write buffer
  
      for ( int i = 0 ; i < BUFFER_SIZE ; i++ ) {

        writeBuffer[i] = bufferA[i];
      }
      bufferA.clear(); //clear the buffer so it is not full
      readyToWrite = true; //tells the board we are ready to write
  
  }



//------------------------------------------------------------------------------------------------

void endRecordingSession(){
  // this function runs the necessary code to setup the recording before it starts

   Serial.println("Writing last data to card");
    recFile.flush();
    
    if ( r.headerPrinted == false) {

      recFile.seek(0);
      r.clearHeaderBuffer();
      recFile.println(r.formatHeader1());
      r.clearHeaderBuffer();
      recFile.println(r.formatHeader2());
      r.clearHeaderBuffer();
      recFile.println(r.formatHeader3());
      r.clearHeaderBuffer();
      recFile.println(r.formatHeader4());
      r.clearHeaderBuffer();
      recFile.println();
      recFile.println(r.channelNames());

      r.headerPrinted = true;
    }
    bufferA.clear(); //potentially unecessary
    recFile.close();
    Serial.println("File Closed");
    filenameAvailable = false;
    fileOpen = false; //uncomment when sure not to overite
    r.readyToStartAgain();
  }

//------------------------------------------------------------------------------------------------
void detectHardware(){
    
// This function verify if hardware to communicate with the computer is connected
  
    if (Ethernet.hardwareStatus() == EthernetNoHardware) {
    Serial.println("Ethernet shield was not found.");
    while (true) {
      delay(1); // do nothing, no point running without Ethernet hardware
    }
  }
  else if (Ethernet.hardwareStatus() == EthernetW5100) {
    Serial.println("W5100 Ethernet controller detected.");
  }
  else if (Ethernet.hardwareStatus() == EthernetW5200) {
    Serial.println("W5200 Ethernet controller detected.");
  }
  else if (Ethernet.hardwareStatus() == EthernetW5500) {
    Serial.println("W5500 Ethernet controller detected.");
  }
  }
  
//------------------------------------------------------------------------------------------------

void detectCable(){
// This function verify is a cable is connected to the hardware
 
  if (Ethernet.linkStatus() == LinkOFF) {
    Serial.println("Ethernet cable is not connected.");
  }else{
        Serial.println("Ethernet cable is connected!");
    }
}

//------------------------------------------------------------------------------------------------

void showIP(){
  //This function prints out the microcontroller ip to the serial port. Used for dev, could be remove
  
  Serial.print("Device IP adress: ");
  Serial.print(Ethernet.localIP());
  Serial.println();
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

//------------------------------------------------------------------------------------------------

void cardInfo() {
//This function print out the card info to the serial port
  
  // print the type of card
  Serial.print("\nCard type: ");
  switch (card.type()) {
    case SD_CARD_TYPE_SD1:
      Serial.println("SD1");
      break;
    case SD_CARD_TYPE_SD2:
      Serial.println("SD2");
      break;
    case SD_CARD_TYPE_SDHC:
      Serial.println("SDHC");
      break;
    default:
      Serial.println("Unknown");
  }

  // Now we will try to open the 'volume'/'partition' - it should be FAT16 or FAT32
  if (!volume.init(card)) {
    Serial.println("Could not find FAT16/FAT32 partition.\nMake sure you've formatted the card");
    return;
  }


  // print the type and size of the first FAT-type volume
  uint32_t volumesize;
  Serial.print("\nVolume type is FAT");
  Serial.println(volume.fatType(), DEC);
  Serial.println();

  volumesize = volume.blocksPerCluster();    // clusters are collections of blocks
  volumesize *= volume.clusterCount();       // we'll have a lot of clusters
  if (volumesize < 8388608ul) {
    Serial.print("Volume size (bytes): ");
    Serial.println(volumesize * 512);        // SD card blocks are always 512 bytes
  }
  Serial.print("Volume size (Kbytes): ");
  volumesize /= 2;
  Serial.println(volumesize);
  Serial.print("Volume size (Mbytes): ");
  volumesize /= 1024;
  Serial.println(volumesize);


  Serial.println("\nFiles found on the card (name, date and size in bytes): ");
  root.openRoot(volume);

  // list all files in the card with date and size
  root.ls(LS_R | LS_DATE | LS_SIZE);

}

//------------------------------------------------------------------------------------------------

void recommendedSetup() {
  //PJRC recommended setup code for ethernet sd card module
  pinMode(9, OUTPUT);
  digitalWrite(9, LOW);    // begin reset the WIZ820io
  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);  // de-select WIZ820io
  pinMode(4, OUTPUT);
  digitalWrite(4, HIGH);   // de-select the SD Card
  digitalWrite(9, HIGH);   // end reset pulse
}

//------------------------------------------------------------------------------------------------

void checkForCard() {
  //This function check if a SD card is inserted
  
  Serial.print("\nInitializing SD card...");
  // we'll use the initialization code from the utility libraries
  // since we're just testing if the card is working!
  if (!card.init(SPI_FULL_SPEED, chipSelect)) {
    Serial.println("initialization failed. Things to check:");
    Serial.println("* is a card inserted?");
    Serial.println("* is your wiring correct?");
    Serial.println("* did you change the chipSelect pin to match your shield or module?");
    return;
  } else {
    Serial.println("Wiring is correct and a card is present.");
    SD.begin(chipSelect);
  }


}

//------------------------------------------------------------------------------------------------

void setupAllSensors() {
// this function runs the necessary code to setup the sensors

  heart.reset();
  sc1.reset();
  resp.reset();

}

//------------------------------------------------------------------------------------------------

void updateAllSensors() {
// this function updates the sensor every loop

  heart.update();
  sc1.update();
  resp.update();

}

//------------------------------------------------------------------------------------------------

void setupButtons(int intervalms){
  for (int i = 0; i < NUM_BUTTONS; i++) {
    buttons[i].attach( BUTTON_PINS[i] , INPUT_PULLUP  );       //setup the bounce instance for the current button
    buttons[i].interval(intervalms);              // interval in ms
  }

  startButton.attach(START_BUTTON_PIN , INPUT_PULLUP);
  startButton.interval(intervalms);   
  stopButton.attach(STOP_BUTTON_PIN , INPUT_PULLUP);
  stopButton.interval(intervalms);   

  
}


void updateButtons(){

  
  startButton.update();
  stopButton.update();
  
    for (int i = 0; i < NUM_BUTTONS; i++)  {
    // Update the Bounce instance :
    buttons[i].update();
    //
    buttonStatus[i] = buttons[i].read();
    if( buttonStatus[i] == 0 ){
    pressedButton = i;
    break;
   }

   if( i == NUM_BUTTONS -1){
    //no button were pressed
    pressedButton = -1;
    }
  }
  }
