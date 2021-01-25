/////////////////////////////////GLOBAL VARIABLES//////////////////////////////////
/*
    This file contains all the #include statements, all the define statements
    and all the global variable definition

*/

//----------------------------------DEFINE STATEMENTS-------------------------------//

#define CS_PIN 10
#define BUFFER_SIZE 768 //set the buffer size here. it needs the be a multiple of the number of columns ex: timestamps, marker, heart, gsr, resp -- 5 columns so the buffer is 640
#define NUM_SIGNALS 4  //Set number of signal recorded here 

#define START_BUTTON_PIN 2
#define MARKER_BUTTON_PIN 3

#define POT_PIN A0
#define NUM_EMOTIONS  7
#define REFRESH_RATE 5000 //IN MICROSECOND DIVIDE 1000000 BY THE REFRESH RATE IN HZ ( EXEMPLE FOR 200HZ 1000000/200 = 5000)
#define LED_HEART 7
#define LED_GSR1 14
#define LED_GSR2 0
#define LED_TEMP 8
//----------------------------------ADDING LIBRARIES-------------------------------//

#include <LiquidCrystalFast.h>
#include <Chrono.h>
#include <LightChrono.h>
#include "Recording.h" //include Custom recording class


//include sd library
#include <SD.h>
#include <SD_t3.h>
#include <SPI.h>

//include Ethernet library
#include <Ethernet.h>
#include <EthernetUdp.h>


//include biodata library
//some of these file could be removed
#include <Respiration.h>
#include <MinMax.h> //this one
#include <Lop.h> //this one
#include <Threshold.h> //this one
//#include <Hip.h> //this file is causing a bug
#include <SkinConductance.h>
#include <Average.h> // this one 
#include <Heart.h>


//include circular buffer library
#include <CircularBuffer.h>

//include OSC library
//could probaly just keep a couple of these
#include <OSCBoards.h>
#include <OSCBundle.h>
#include <OSCData.h>
#include <OSCMatch.h>
#include <OSCMessage.h>
#include <OSCTiming.h>
#include <SLIPEncodedSerial.h>
#include <SLIPEncodedUSBSerial.h>

#include <Bounce2.h>

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
//-----------------------------GLOBAL VARIABLES DEFINITION--------------------------//

LiquidCrystal_I2C lcd(0x27, 20, 4); // set the LCD address to 0x27 for a 16 chars and 2 line display

IntervalTimer captureData; //create instance of Interval timer for sensors
Recording r(NUM_SIGNALS); //create instance of Recording object


Heart heart(A7); //Create instance for heart sensor
SkinConductance sc1(A6); //Create instance for gsr sensor
SkinConductance sc2(A2);
Respiration resp(A3); //create instance for respiration sensor



// set up variables using the SD utility library functions:
Sd2Card card;
SdVolume volume;
SdFile root;

File recFile; //create a instance of File object for the recording file
bool fileOpen = false; //boolean to keep track of if a file is open to write
const int chipSelect = 4; //cs pin for sd card

String fileDigits = "000";
String fileExtension = ".txt";
char filename[11] = "rec199.txt";  //initial filename here

String signalTypes[NUM_SIGNALS] = {"Heart" , "GSR1","GSR2", "TEMP"};


// Enter da MAC address and IP address for your controller below.
// The IP adress will be dependent on your local network:
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED }; //default Arduino mac adress
IPAddress ip(169, 254, 2, 78); //Microcontroller ip adress

unsigned int localPort = 8888;      // local port to listen on
int computerPort = 8888; //check if can replace this for localPort in the code ( most probably yes ) because they hold the same value
IPAddress computerIP ; //ip adress of the computer we're talking to.

//OSCBundle sample; //create an osc bundle object to send the sample @ 1000hz
EthernetUDP Udp; // An EthernetUDP instance to let us send and receive packets over UDP

CircularBuffer<unsigned long , BUFFER_SIZE >bufferA; //Create an instance of a circular buffer to store data before sending
unsigned long writeBuffer[BUFFER_SIZE] = {0}; // Temporary buffer that holds the data while writting to card


//These bools can be used as a protection to prevent starting a recording without all th infos for the header
bool Connected = false;
bool nameReceived = false;
bool locReceived = false;
bool dateReceived = false;
bool rateReceived = false;


bool readyToWrite = false; //boolean to prevent writing to the card when its not time

unsigned long stamp = 0; //holds the millis timestamp when the recording starts

bool filenameAvailable = false;



Bounce startButton =  Bounce();
Bounce markerButton = Bounce();

String emotionFilename;
String selectedEmotion ;
String emotions[NUM_EMOTIONS] = {" Happiness      " , " Sadness        " , " Fear           " , " Anger          " , " Arousal        " , " Surprise       " , " Neutral        "} ; //spaces in the strings are use to format text to display
String emotionsName[NUM_EMOTIONS] = {"Happiness" , "Sadness" , "Fear" , "Anger" , "Arousal" , "Surprise" , "Neutral"} ; //spaces in the strings are use to format text to display
String emotionsFile[NUM_EMOTIONS] = {"HAP" , "SAD" , "FEA" , "ANG" , "ARO" , "SUR" , "NEU"} ;
String recordingLCD[4] = { "  Recording     ", "  Recording.  ", "  Recording..  ", "  Recording...  "};
char feelingIt[16] = "   Feeling it   ";
int recordingLCDIndex = 0;
char emptyLine[16] = "                 ";
String infoEmotion;

int displayIndex = 0;
int potVal = 0;
unsigned long timestamp;
Chrono recordingUpdate;
Chrono lcdUpdate;
//buffers use one more char than the screen can display for a null terminator
char lcdLine1[17];
char lcdLine2[17];

bool filenameNotChecked = true;
char idleLine1[16] = "Select emotion: ";
char lcdRecordingOver[16] = "End of recording";
