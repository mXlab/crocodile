////////////////////////////////Helpers funtions//////////////////////////////////
/*
    This file contains all the misc helpers functions necessary:
    -createFilename()
    -checkFileName()
    -datalog()
    -writeToCard()
    -transferBuffer()
    -emotionSelection()
    -idleDisplay()
    
*/

//------------------------------------------------------------------------------------------------

void createFilename() {

  String _name = emotionFilename + fileDigits + fileExtension;
  Serial.println(_name);
  _name.toCharArray(filename, sizeof(filename));

}
//------------------------------------------------------------------------------------------------

void checkFileName() {


  while ( filenameAvailable == false ) {
    //check if file already exist
    if (SD.exists(filename)) {
      // if yes, make a different name
      char endDigit = filename[5];
      char middleDigit = filename[4];
      char startDigit = filename[3];


      if ( endDigit >= 57) {

        endDigit = 48;
        middleDigit = middleDigit + 1;

        if (middleDigit >= 57) {

          middleDigit = 48;
          startDigit = startDigit + 1;

        }
      } else {

        endDigit = endDigit + 1;

      }

      filename[5] = endDigit;
      filename[4] = middleDigit;
      filename[3] = startDigit;


    } else {
      Serial.print("New filename: ");
      Serial.print(filename);
      Serial.println();
      filenameAvailable = true;
    }

  }

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
      } else if (formatIndex == 1) {
        marker =  bufferArg[i];
      }
      else {

        formatBuffer[formatIndex - 2] = bufferArg[i];
      }
      if ( formatIndex == numChan - 1) {
        
        recFile.println(r.formatData(timestamp, marker, formatBuffer));

      }
    }

    recFile.flush();
  } else {
    Serial.println("cant write to file");
  }
}

//------------------------------------------------------------------------------------------------

void writeToCard() {
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

void transferBuffer() {
  //transfer bufferA to write buffer

  for ( int i = 0 ; i < BUFFER_SIZE ; i++ ) {

    writeBuffer[i] = bufferA[i];
  }
  bufferA.clear(); //clear the buffer so it is not full
  readyToWrite = true; //tells the board we are ready to write

}

//------------------------------------------------------------------------------------------------

int emotionSelection( int potValue) {

  int emotionIndex = constrain(floor(potValue / 146), 0 , NUM_EMOTIONS - 1) ;

  return emotionIndex;
}

//------------------------------------------------------------------------------------------------

void idleDisplay(){
      if (displayIndex == 0) {
      
      int emotionIndex = emotionSelection(potVal); //set the good emotion index based on the potentiometer current value
      emotionFilename = emotionsFile[emotionIndex];  //select the filename prefix based on the emotionIndex
      selectedEmotion = emotions[emotionIndex]; //select the string to display on the lcd based on the emotionIndex
      infoEmotion = emotionsName[emotionIndex]; //select the string to put in the header based on the emotionIndex

      //update lcd buffers
      idleLine1.toCharArray(lcdLine1, 17);
      selectedEmotion.toCharArray(lcdLine2, 17);
      
    } else if ( displayIndex == 1 ) {
      
      sprintf(lcdLine1, "Start in %-6d", r.ctdwnIndex); //compose message and update lcd buffer
      String startLine2 = "File: ";

      if (filenameNotChecked == true) {
      
        //create and check filename based on selected emotion
        filenameNotChecked = false;
        createFilename();
        checkFileName();
      }

      startLine2.concat(filename); //Compose the string that's gonna show on the second line of the lcd
      startLine2.toCharArray(lcdLine2, 17); //update lcd buffer

    }
}
