/////////////////////////////////Setup Helpers//////////////////////////////////
/*
    This file contains all the functions updating every loop
    
    -updateData() <- IntervalTimer
    -updatepAllSensors()
    -updateButtons()
    -updatePotentiometer
    -updateLCD()

*/
//------------------------------------------------------------------------------------------------

void updateData() {
  /*
     This is the function that is run by the interval timer
     It samples the data from the sensor and push it to the buffer
     Sending the pushed data via OSC to the computer should be done here
  */
  timestamp = millis() - stamp;
  int marker;

  if (markerButton.read() == 0) { //when the button is pressed
    r.placeMarker();
    feelingIt.toCharArray(lcdLine2,17);
  }

  if ( r.marker == true) {
    marker = 1;
    r.resetMarkerBool();
  } else {
    marker = 0 ;
  }

//  int temp1 = 6;
//  int temp2 = 5;
//  int temp3 = 4;
//  int temp4 = 3;
  //pushing temporary data for development
  bufferA.push(timestamp);
  bufferA.push(marker);
//  bufferA.push(temp1);
//  bufferA.push(temp2);
//  bufferA.push(temp3);
//  bufferA.push(temp4);



  // pushing sampled data from the sensor
    bufferA.push(heart.getRaw());
    bufferA.push(sc1.getRaw());
    bufferA.push(sc2.getRaw());
    bufferA.push(resp.getRaw());

}

//------------------------------------------------------------------------------------------------

void updateAllSensors() {
  // this function updates the sensor every loop

  heart.update();
  //Serial.println(heart.getRaw());
  sc1.update();
   //Serial.println(sc1.getRaw());
  sc2.update();
  //Serial.println(sc2.getRaw()); 
  resp.update();
 //Serial.println(resp.getRaw());
}

//------------------------------------------------------------------------------------------------

void updateButtons() {
  //update the state of the buttons every loop
  startButton.update();
  markerButton.update();
  //Serial.println(markerButton.read());

}

//------------------------------------------------------------------------------------------------

void updatePotentiometer() {
  //potVal = analogRead(POT_PIN);
  //encoder code. doesnt seem to work
  long newPosition = myEnc.read()/4;
    

  if (newPosition != oldPosition) { //if position changed
    
    oldPosition = newPosition;
    //Serial.println(oldPosition); 
    }
}

//------------------------------------------------------------------------------------------------

void updateLCD() {
  lcd.setCursor(0, 0);
  lcd.print(lcdLine1);
  lcd.setCursor(0, 1);
  lcd.print(lcdLine2);

}
