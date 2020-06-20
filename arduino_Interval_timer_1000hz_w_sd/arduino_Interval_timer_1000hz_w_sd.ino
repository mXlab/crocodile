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

#include <CircularBuffer.h>
#define BUFFER_SIZE 512

IntervalTimer captureData;

//Create instance for each sensor
Heart heart(A0);
SkinConductance sc1(A3);
Respiration resp(A2);

File recordFile;

// set up variables using the SD utility library functions:
Sd2Card card;
SdVolume volume;
SdFile root;
const int chipSelect = 4; //cs for sd card
char filename[11] = "rec000.txt";
//Create instances for circular buffers
CircularBuffer<int , BUFFER_SIZE >bufferA;
char writeBuffer[BUFFER_SIZE] = {0};
bool readyToWrite = false;
int looped = 0;

bool recordingStart = false;
bool recording = false;
bool recordingStop = false;
bool fileOpen = false;
File recFile;

void setup() {
  // put your setup code here, to run once:

delay(2000);
 
  recommendedSetup();
  Serial.begin(9600);
  setupAllSensors();
  checkForCard();
  cardInfo();

captureData.begin( updateData, 1000);
recordingStart = true;
}

void loop() {
updateAllSensors();

//verify if recording time ws reached and is the file is not stopped already
//could add another check to see if its in the recording state
if(looped == 10 && recordingStop == false ){

  recFile.close();
  Serial.println("File Closed");
 
  recordingStop = true;
  recording = false;
  
  }

//verify if we just started a recording and the  file is not open
if(recordingStart == true && fileOpen == false){
  //open file
  Serial.println("Open File");
  recFile = SD.open( filename , FILE_WRITE);
  fileOpen = true;
  recording = true;
  Serial.println("START RECORDING");
}


if( recording == true)
{
//only do this while recording
  noInterrupts()
  if(bufferA.isFull() && readyToWrite == false){
  //Serial.println("Capture");
  Serial.println(bufferA[128]);
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
  }

}

}

void updateData(){
  //push sensor data to arrays

  bufferA.push(heart.getRaw());
  bufferA.push(sc1.getRaw());
  bufferA.push(resp.getRaw());
  }


void datalog(char bufferArg[BUFFER_SIZE]){

  if(recFile){
  Serial.println("Writting to card");
  recFile.write(bufferArg , BUFFER_SIZE);
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
