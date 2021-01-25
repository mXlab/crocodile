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
    sprintf(lcdLine2, "%s" , feelingIt);
;
  }

  if ( r.marker == true) {
    marker = 1;
    r.resetMarkerBool();
  } else {
    marker = 0 ;
  }

  bufferA.push(timestamp);
  bufferA.push(marker);


  // pushing sampled data from the sensor
    bufferA.push(heart.getRaw());
    //Serial.println(heart.getRaw());
    bufferA.push(sc1.getRaw());
    //Serial.println(sc1.getRaw());

    bufferA.push(sc2.getRaw());
    //Serial.println(sc2.getRaw()); 

    bufferA.push(resp.getRaw());
    //Serial.println(resp.getRaw());


}

//------------------------------------------------------------------------------------------------

void updateAllSensors() {
  // this function updates the sensor every loop
  
  //CHANGE THE LED SIGNAL SCALING HERE
  heart.update();
  analogWrite(LED_HEART,map(heart.getRaw(),500,600 , 0 , 255));
  //Serial.println(heart.getRaw());  //uncomment to print heart signal in the serial monitor
  sc1.update();
  analogWrite(LED_GSR1,map(sc1.getRaw(),500 ,900 , 0 , 255));
 //Serial.println(sc1.getRaw());  //uncomment to print GSR1  signal in the serial monitor
  sc2.update();
  analogWrite(LED_GSR2,map(sc2.getRaw(),500 ,900 , 0 , 255));
  //Serial.println(sc2.getRaw());   //uncomment to print GSR2  signal in the serial monitor
  resp.update();
  analogWrite(LED_TEMP,map(resp.getRaw(),0 ,1023 , 0 , 255));
  //Serial.println(resp.getRaw());  //uncomment to print temp  signal in the serial monitor
}

//------------------------------------------------------------------------------------------------

void updateButtons() {
  //update the state of the buttons every loop
  startButton.update();
  //Serial.println(startButton.read());
  markerButton.update();
  //Serial.println(markerButton.read());

}

//------------------------------------------------------------------------------------------------

void updatePotentiometer() {

  long newPosition = myEnc.read()/4;


  if (newPosition != oldPosition) { //if position changed
//    Serial.print("ENC RAW : " );
//    Serial.println(myEnc.read());
    
    oldPosition = constrain(newPosition %7, 0 , NUM_EMOTIONS - 1);
    potVal = oldPosition;
//    Serial.print("ENC VALUE : " );
//    Serial.println(oldPosition); 
    }
}

//------------------------------------------------------------------------------------------------

void updateLCD() {

  lcd.setCursor(0, 0);
  lcd.print(lcdLine1);
  lcd.setCursor(0, 1);
  lcd.print(lcdLine2);
 
}
