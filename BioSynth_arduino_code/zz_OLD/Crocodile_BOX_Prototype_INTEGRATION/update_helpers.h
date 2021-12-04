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

    heart.update();
  sc1.update();
  sc2.update();
  resp.update();
  // pushing sampled data from the sensor
    bufferA.push(heart.getNormalized()*100);
    //Serial.println(heart.getRaw());
    bufferA.push(sc1.getRaw());
    //Serial.println(sc1.getRaw());

    bufferA.push(sc2.getRaw());
    //Serial.println(sc2.getRaw()); 

    bufferA.push(resp.getRaw());
    //Serial.println(resp.getRaw());
    Serial.printf("%.2f,%d,%d,%d",heart.getNormalized()*100,sc1.getRaw(),sc2.getRaw(),resp.getRaw());
  Serial.println();
}

//------------------------------------------------------------------------------------------------

void updateAllSensors() {
  // this function updates the sensor every loop
  
  //CHANGE THE LED SIGNAL SCALING HERE
  heart.update();
  sc1.update();
  sc2.update();
  resp.update();

  int intHEART = (heart.getNormalized()*100);
  int intGSR1 = (sc1.getSCR()*100);
  int intGSR2 = (sc2.getSCR()*100);
  int intRESP = (resp.getNormalized()*100);

  Serial.printf("%d,%d,%d,%d",intHEART,intGSR1,intGSR2,intRESP);
  +
  
  
  
  Serial.println();
 // analogWrite(LED_HEART,map(intHEART,1 , 100, 0 , 255));
  //Serial.println(heart.getNormalized());  //uncomment to print heart signal in the serial monitor
  
 // analogWrite(LED_GSR1,map(intGSR1,1 , 100 , 0 , 255));
 //Serial.println(sc1.getSCR());  //uncomment to print GSR1  signal in the serial monitor
 
 
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