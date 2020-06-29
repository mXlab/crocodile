/////////////////////////////////GLOBAL VARIABLES//////////////////////////////////


//include Custom recording class
#include "Recording.h"


//include sd library
#include <SD.h>
#include <SD_t3.h>
#include <SPI.h>

#include <Ethernet.h>
#include <EthernetUdp.h>


//include biodata library
//some of these file could be removed
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

#define CS_PIN 10
#define BUFFER_SIZE 512 //set the buffer size here 
#define NUM_SIGNALS 3  //Set number of signal recorded here 


IntervalTimer captureData; //create instance of Interval timer for sensors
Recording r(NUM_SIGNALS); //create instance of Recording object

Heart heart(A0); //Create instance for heart sensor
SkinConductance sc1(A3); //Create instance for gsr sensor
Respiration resp(A2); //create instance for respiration sensor



// set up variables using the SD utility library functions:
Sd2Card card;
SdVolume volume;
SdFile root;
File recFile;
bool fileOpen = false;
const int chipSelect = 4; //cs for sd card



char filename[11] = "rec040.txt";  //initial fileanme here


//Setup UDP



// Enter a MAC address and IP address for your controller below.
// The IP address will be dependent on your local network:
byte mac[] = {
  0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED
};
IPAddress ip(169,254,2,78);

unsigned int localPort = 8888;      // local port to listen on

IPAddress ComputerIP ;
int ComputerPort ;
// buffers for receiving and sending data
char packetBuffer[UDP_TX_PACKET_MAX_SIZE];  // buffer to hold incoming packet,
char ReplyBuffer[] = "aCknowledged";        // a string to send back

// An EthernetUDP instance to let us send and receive packets over UDP
EthernetUDP Udp;

bool Connected = false; 
bool nameReceived = false; 
bool locReceived = false; 
bool dateReceived = false; 
bool rateReceived = false;






//Create instances for circular buffers
CircularBuffer<unsigned long , BUFFER_SIZE >bufferA;
unsigned long writeBuffer[BUFFER_SIZE] = {0};

bool readyToWrite = false;
int looped = 0;
int maxLoop = 200;
