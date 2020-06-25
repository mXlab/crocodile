//include Custom recording class
#include "Recording.h"

//include sd library
#include <SD.h>
#include <SD_t3.h>
#include <SPI.h>

//include biodata library
#include <Respiration.h>
#include <MinMax.h>
#include <Lop.h>
#include <Threshold.h>
//#include <Hip.h>
#include <SkinConductance.h>
#include <Average.h>
#include <Heart.h>

//include circular buffer library
#include <CircularBuffer.h>



#define BUFFER_SIZE 512
#define NUM_SIGNALS 3  //Set number of signal recorded here 


IntervalTimer captureData;
Recording r(NUM_SIGNALS);

//Create instance for each sensor
Heart heart(A0);
SkinConductance sc1(A3);
Respiration resp(A2);



// set up variables using the SD utility library functions:
Sd2Card card;
SdVolume volume;
SdFile root;
File recFile;
bool fileOpen = false;

const int chipSelect = 4; //cs for sd card


char filename[11] = "rec030.txt";  //initial fileanme here

//Create instances for circular buffers
CircularBuffer<unsigned long , BUFFER_SIZE >bufferA;
unsigned long writeBuffer[BUFFER_SIZE] = {0};

bool readyToWrite = false;
int looped = 0;
int maxLoop = 200;

char debugHeader[65] = "#################################################################";



void setup() {
  
  delay(2000);
 
  recommendedSetup();
  Serial.begin(9600);
  setupAllSensors();

  testClass();

  
  Serial.println(r.formatHeader1());
  Serial.println(r.formatHeader2());
  Serial.println(r.formatHeader3());
  Serial.println(r.formatHeader4());


  checkForCard();
  cardInfo();

  

}

void loop() {
updateAllSensors();

//verify if recording time ws reached and is the file is not stopped already
//could add another check to see if its in the recording state


if(looped == maxLoop && r.isReadyToStop() ){
  Serial.println("End Recording");
  recFile.seek(0);
  recFile.println(r.formatHeader1());
  recFile.println(r.formatHeader2());
  recFile.println(r.formatHeader3());
  recFile.println(r.formatHeader4());
  recFile.println();

  recFile.close();
  Serial.println("File Closed");
  captureData.end();
  //fileOpen = false; //uncomment when sure not to overite
  r.readyToStartAgain();
  }


//verify if we just started a recording and the  file is not open
if(r.isRecording() == false && fileOpen == false){
  //open file
  Serial.println("Open File");
  recFile = SD.open( filename , FILE_WRITE);
  recFile.println(r.channelNames());
  fileOpen = true;
  
  captureData.begin( updateData, 1000);
  r.startRecording();
  Serial.println("START RECORDING");
}


if( r.isRecording() )
{
//only do this while recording
  noInterrupts()
  if(bufferA.isFull() && readyToWrite == false){


    //transfer buffer to write buffer
    for( int i = 0 ; i < BUFFER_SIZE ; i++ ) {
      //Serial.println(i);
       writeBuffer[i] = bufferA[i];
      }
    readyToWrite = true; //tells the board we are ready to write
    bufferA.clear(); //clear the buffer so it is not full
  }
  interrupts();

  if(readyToWrite == true ){
   
   datalog(writeBuffer);
   readyToWrite = false;
   looped++;
   if(looped == maxLoop){
    r.stopRecording();
    }
  }

}

}

void updateData(){
  //push sensor data to arrays
  unsigned long temp = micros();
    bufferA.push(temp);
    bufferA.push(1);
    bufferA.push(2);
    bufferA.push(3);
    
//   bufferA.push(heart.getRaw());
//  / bufferA.push(sc1.getRaw());
//  bufferA.push(resp.getRaw());
  }


void datalog(int bufferArg[BUFFER_SIZE]){
  int numChan = NUM_SIGNALS + 1;
 
  int formatBuffer[numChan] = {0};
  if(recFile){
  Serial.println("Writting to card");
  
  for( int i = 0 ; i < BUFFER_SIZE ; i++){
    
    int formatIndex = i%numChan;
     formatBuffer[formatIndex] = bufferArg[i];

    if ( formatIndex == numChan -1){
    
      recFile.println(r.formatData(formatBuffer));
    
    }
  }

  
  recFile.println("LOOPED");
  recFile.flush();
  //dataWrote = true;
  }else{
    Serial.println("cant write to file");
    }
  }




void cardInfo(){
    // print the type of card
  Serial.print("\nCard type: ");
  switch(card.type()) {
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
void checkForCard(){
  
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

 void setupAllSensors(){
  
  heart.reset();
  sc1.reset();
  resp.reset();
 
  }

void updateAllSensors(){

   heart.update();
   sc1.update();
   resp.update();
  
  }

void testClass(){
  String test = "Etienne Montenegro";
 String loc = "Montreal";
 String signals[4] = {"heart","gsr1","gsr2","resp"};
 int rate = 1000;
 int testData[5] = {340985654,1024,1024,1024,1024};
 
 r.setSubjectName ( test);
 r.setLocation (loc);
 r.setSignals(signals);
 r.setRecRate(rate);
 r.setDate("25","06","20");
 
 Serial.println(r.getSubjectName());
 Serial.println(r.getLocation());
 Serial.println(r.getSignal(0));
 Serial.println(r.getDate());
 Serial.print(r.getRecRate());
 Serial.print(" Hz");
 Serial.println();
 Serial.println(r.formatData(testData));

 Serial.println(r.channelNames());
  }
