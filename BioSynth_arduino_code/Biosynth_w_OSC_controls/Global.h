
/////////////////////////////////GLOBAL VARIABLES//////////////////////////////////
/*
    This file contains all the #include statements, all the define statements
    and all the global variable definition

*/

//----------------------------------ADDING LIBRARIES-------------------------------//

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

//----------------------------------DEFINE STATEMENTS-------------------------------//

#define CS_PIN 10
#define BUFFER_SIZE 640 //set the buffer size here. it needs the be a multiple of the number of columns ex: timestamps, marker, heart, gsr, resp -- 5 columns so the buffer is 640
#define NUM_SIGNALS 3  //Set number of signal recorded here 
#define NUM_BUTTONS 6
#define START_BUTTON_PIN 21
#define STOP_BUTTON_PIN 22

//-----------------------------GLOBAL VARIABLES DEFINITION--------------------------//


IntervalTimer captureData; //create instance of Interval timer for sensors
Recording r(NUM_SIGNALS); //create instance of Recording object


Heart heart(A0); //Create instance for heart sensor
SkinConductance sc1(A3); //Create instance for gsr sensor
Respiration resp(A2); //create instance for respiration sensor



// set up variables using the SD utility library functions:
Sd2Card card;
SdVolume volume;
SdFile root;

File recFile; //create a instance of File object for the recording file
bool fileOpen = false; //boolean to keep track of if a file is open to write
const int chipSelect = 4; //cs pin for sd card
char filename[11] = "rec199.txt";  //initial filename here



// Enter da MAC address and IP address for your controller below.
// The IP adress will be dependent on your local network:
byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED }; //default Arduino mac adress
IPAddress ip(169,254,2,78); //Microcontroller ip adress

unsigned int localPort = 8888;      // local port to listen on
int computerPort = 8888; //check if can replace this for localPort in the code ( most probably yes ) because they hold the same value
IPAddress computerIP ; //ip adress of the computer we're talking to. 

// buffers for receiving and sending data
// can probably remove these two buffer because using osc now
//char packetBuffer[UDP_TX_PACKET_MAX_SIZE];  // buffer to hold incoming packet,
//char ReplyBuffer[] = "aCknowledged";        // a string to send back

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



const uint8_t BUTTON_PINS[NUM_BUTTONS] = {0, 1, 2, 18, 19, 20};
int pressedButton;
int buttonStatus[NUM_BUTTONS] = {0};
Bounce startButton =  Bounce();
Bounce stopButton =  Bounce();
Bounce * buttons = new Bounce[NUM_BUTTONS];
