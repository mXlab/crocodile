void updateData() {
  //push sensor data to arrays
  unsigned long temp = millis();
  bufferA.push(temp);
  bufferA.push(1);
  bufferA.push(2);
  bufferA.push(3);

  //   bufferA.push(heart.getRaw());
  //  / bufferA.push(sc1.getRaw());
  //  bufferA.push(resp.getRaw());
}

//------------------------------------------------------------------------------------------------
void setupRecording(){
  //open file
    Serial.println("Open File");
    recFile = SD.open( filename , FILE_WRITE);

    //recFile.println();
    fileOpen = true;

    captureData.begin( updateData, 1000);
    r.startRecording();
    r.headerPrinted = false;
    Serial.println("START RECORDING");
}

//------------------------------------------------------------------------------------------------

void datalog(int bufferArg[BUFFER_SIZE]) {
  int numChan = NUM_SIGNALS + 1;

  int formatBuffer[NUM_SIGNALS] = {0};
  unsigned long timestamp = 0 ;
  if (recFile) {
    Serial.println("Writting to card");

    for ( int i = 0 ; i < BUFFER_SIZE ; i++) {

      int formatIndex = i % numChan;

      if (formatIndex == 0 ) {
        timestamp = bufferArg[i];
      } else {
        formatBuffer[formatIndex] = bufferArg[i];
      }
      if ( formatIndex == numChan - 1) {

        recFile.println(r.formatData(timestamp, formatBuffer));

      }
    }


    recFile.print("LOOPED");
    recFile.print(looped);
    recFile.println();
    recFile.flush();
    //dataWrote = true;
  } else {
    Serial.println("cant write to file");
  }
}


//------------------------------------------------------------------------------------------------
 void writeToCard(){
      
      datalog(writeBuffer);
      looped++;
      if (looped == maxLoop) {

        Serial.println("Stop sensors");
        captureData.end();
        bufferA.clear();
        r.stopRecording();
      }

      readyToWrite = false;
      } 

//------------------------------------------------------------------------------------------------

void transferBuffer(){
  //transfer buffer to write buffer
      for ( int i = 0 ; i < BUFFER_SIZE ; i++ ) {
        //Serial.println(i);
        writeBuffer[i] = bufferA[i];
      }
      bufferA.clear(); //clear the buffer so it is not full
      readyToWrite = true; //tells the board we are ready to write
  
  }

  //------------------------------------------------------------------------------------------------

void endRecordingSession(){
   Serial.println("End Recording");
    recFile.flush();
    if ( r.headerPrinted == false) {
      recFile.seek(0);

      recFile.println(r.formatHeader1());
      recFile.println(r.formatHeader2());
      recFile.println(r.formatHeader3());
      recFile.println(r.formatHeader4());
      recFile.println();
      recFile.println(r.channelNames());

      r.headerPrinted = true;
    }
    bufferA.clear(); //potentially unecessary
    recFile.close();
    Serial.println("File Closed");
    //fileOpen = false; //uncomment when sure not to overite
    r.readyToStartAgain();
  }

//------------------------------------------------------------------------------------------------




void cardInfo() {
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

void setupAllSensors() {

  heart.reset();
  sc1.reset();
  resp.reset();

}

void updateAllSensors() {

  heart.update();
  sc1.update();
  resp.update();

}

void testClass() {
  String test = "Etienne Montenegro";
  String loc = "Montreal";
  String signals[4] = {"heart", "gsr1", "gsr2", "resp"};
  int rate = 1000;
  int testData[5] = {340985654, 1024, 1024, 1024, 1024};

  r.setSubjectName ( test);
  r.setLocation (loc);
  r.setSignals(signals);
  r.setRecRate(rate);
  r.setDate("25", "06", "20");

  Serial.println(r.getSubjectName());
  Serial.println(r.getLocation());
  Serial.println(r.getSignal(0));
  Serial.println(r.getDate());
  Serial.print(r.getRecRate());
  Serial.print(" Hz");
  Serial.println();
  Serial.println(r.channelNames());
}
