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
  int sensorData[4] = {0,0,0,0};

  bufferA.push(timestamp);
  bufferA.push(marker);

  for( int i = 0 ; i < 4 ; i++ )
  {    
    switch(i)
    {
      case 0:
        if(connectedSensors[i])
        {   
            heart.update();
            sensorData[i] = heart.getRaw();
            bufferA.push(sensorData[i]);
            setLedBrightness(i , (float)sensorData[i]/1024);
        }
        break;
      case 1:
        if(connectedSensors[i])
        {
            sc1.update();
            sensorData[i] = sc1.getRaw();
            bufferA.push(sensorData[i]);
            setLedBrightness(i , (float)sensorData[i]/1024);
        }
        else
        {
            setLedBrightness(i , 0);
        }
        break;
      case 2:
        if(connectedSensors[i])
        {
         resp.update();
         sensorData[i] = resp.getRaw();
         bufferA.push(sensorData[i]);
         setLedBrightness(i , (float)sensorData[i]/1024); 
        }
        else
        {
            setLedBrightness(i , 0);
        }
        break;
      case 3:
        if(connectedSensors[i])
        {  
          sc2.update();
          sensorData[i] = sc2.getRaw();
          bufferA.push(sensorData[i]);
          setLedBrightness(i , (float)sensorData[i]/1024);
            
        }
        else
        {
            setLedBrightness(i , 0);
        }
        break;    
    }
  }

  if(PLOT_SENSOR) //used for debugging purpose
  {
    Serial.printf("%d,%d,%d,%d",sensorData[0],sensorData[1],sensorData[2],sensorData[3] );
    Serial.println();
  }    
}

//------------------------------------------------------------------------------------------------

void updateButtons() 
{
  //update the state of the buttons every loop
  startButton.update();
  markerButton.update();
}

//------------------------------------------------------------------------------------------------

void updatePotentiometer() 
{
 
  long newPosition = myEnc.read()/4;


  //prevents the encoder from going out of bounds
  if(newPosition < 0)
  {   
    setEncoder(0); 
    newPosition = myEnc.read()/4;
  }
  else if( newPosition > NUM_EMOTIONS-1)
  {   
    setEncoder(NUM_EMOTIONS -1);
    newPosition = myEnc.read()/4;
  }

  if (newPosition != oldPosition) 
  { //if position changed
    oldPosition = constrain(newPosition %7, 0 , NUM_EMOTIONS - 1);
    potVal = oldPosition;
  }
}

//------------------------------------------------------------------------------------------------

void updateLCD() 
{
  lcd.setCursor(0, 0);
  lcd.print(lcdLine1);
  lcd.setCursor(0, 1);
  lcd.print(lcdLine2);
}
