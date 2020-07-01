
void sendMessage( char reply[] ){
//    send a reply to the IP address and port that sent us the packet we received
    Udp.beginPacket(computerIP, computerPort);
    Udp.write(reply);
    Udp.endPacket();
  }


void clearBuffer(){
  //clear the array, add after each loops
   memset(packetBuffer, 0, sizeof packetBuffer);
  }
  

String extractMessage(){ 
  String temp = packetBuffer;
  temp.remove(0,3);
  return temp;
  }

void udpUpdate(){
   // if there's data available, read a packet
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    //Serial.print("Received packet of size ");
    //Serial.println(packetSize);
    //Serial.print("From ");
    IPAddress remote = Udp.remoteIP();
    for (int i=0; i < 4; i++) {
      //Serial.print(remote[i], DEC);
      if (i < 3) {
        //Serial.print(".");
      }
    }
    //Serial.print(", port ");
    //Serial.println(Udp.remotePort());

    // read the packet into packetBufffer
    Udp.read(packetBuffer, UDP_TX_PACKET_MAX_SIZE);
    //Serial.println("Contents:");
    //Serial.println(packetBuffer);
  }
}

void parseMessage(){
     //uses a combination of special characters to parse the received messages
     //
     //Con - sends a message to confirm connection to device
     //IFn - sends a message containing subject name
     //IFl - sends a message containing location
     //IFd - sends a message containing date
     //IFr - sends a message containing rate
     //IFs - sends a message containing signals
     //Sta - Start the recording
     //Sto - Stop the recording
     //Mkr - Place a marker there
     
     
 
   
    if(packetBuffer[0] == 67 && packetBuffer[1] == 111 && packetBuffer[2] == 110){ //parse for "Con" message
    Serial.println("Connection to computer");
    computerIP = Udp.remoteIP();
    computerPort = Udp.remotePort();
    Connected = true;
    sendMessage( "Connected");  //tells the computer it connected to the device
   
    }else if(packetBuffer[0] == 73 && packetBuffer[1] == 70 && packetBuffer[2] == 110){
     
      Serial.println("Received Name");
      r.setSubjectName( extractMessage() );
      sendMessage( "rcvdN");
     
      }else if(packetBuffer[0] == 73 && packetBuffer[1] == 70 && packetBuffer[2] == 108){
     
      Serial.println("Received Location");
      r.setLocation( extractMessage() );
      locReceived = true;
      sendMessage( "rcvdL");
     
      }else if(packetBuffer[0] == 73 && packetBuffer[1] == 70 && packetBuffer[2] == 100){
       
      Serial.println("Received Date");
      String d = extractMessage(); //transform this to setDate format
      //Serial.println(d);
      String dd = d.substring(0,2);
      //Serial.println(dd);
      String mm = d.substring(2,4);
      //Serial.println(mm);
      String yy = d.substring(4,6);
      //Serial.println(yy);
      r.setDate(dd,mm,yy);
      sendMessage( "rcvdD");
     
      }else if(packetBuffer[0] == 73 && packetBuffer[1] == 70 && packetBuffer[2] == 114){
       
      Serial.println("Received Sample Rate");
      String rate =  extractMessage();
      r.setRecRate(rate.toInt() );
      sendMessage( "rcvdR");
       
      }else if(packetBuffer[0] == 73 && packetBuffer[1] == 70 && packetBuffer[2] == 115){
       
      Serial.println("Received Signals types");
      String signalsTemp =  extractMessage();
      //Serial.println(signalsTemp);
      String signalsArray[NUM_SIGNALS] ;
      int idx = 0;
      int substringLastIdx = 0;


      
      for( int i = 0 ; i < signalsTemp.length() ; i++ ){
        char current = signalsTemp[i];
        //Serial.println(current);
        if( i == signalsTemp.length()-1 ) {
          
          signalsArray[idx] = signalsTemp.substring(substringLastIdx, i+1);
          //Serial.println(signalsArray[idx]);
          }  
        if( current == 44){
          signalsArray[idx] = signalsTemp.substring(substringLastIdx, i);
          //Serial.println(signalsArray[idx]);
          substringLastIdx = i+1 ;
          idx++;
          }
        }
        
 
      r.setSignals(signalsArray);
      sendMessage( "rcvdS");
       
      }else if(packetBuffer[0] == 83 && packetBuffer[1] == 116 && packetBuffer[2] == 97){
       
      Serial.println("Start Recording");
      sendMessage( "rcSta");
      r.setupRecording();
       
      }else if(packetBuffer[0] == 83 && packetBuffer[1] == 116 && packetBuffer[2] == 111){
       
      Serial.println("Stop Recording");
      sendMessage( "rcSto");
      r.stopProcess = true;
       
      }else if(packetBuffer[0] == 77 && packetBuffer[1] == 107 && packetBuffer[2] == 114){
       
      Serial.println("Place Marker");
     
      sendMessage( "rcMkr");
       
      }
 
  }


void detectHardware(){
    if (Ethernet.hardwareStatus() == EthernetNoHardware) {
    Serial.println("Ethernet shield was not found.");
    while (true) {
      delay(1); // do nothing, no point running without Ethernet hardware
    }
  }
  else if (Ethernet.hardwareStatus() == EthernetW5100) {
    Serial.println("W5100 Ethernet controller detected.");
  }
  else if (Ethernet.hardwareStatus() == EthernetW5200) {
    Serial.println("W5200 Ethernet controller detected.");
  }
  else if (Ethernet.hardwareStatus() == EthernetW5500) {
    Serial.println("W5500 Ethernet controller detected.");
  }
  }

void detectCable(){
  if (Ethernet.linkStatus() == LinkOFF) {
    Serial.println("Ethernet cable is not connected.");
  }else{
        Serial.println("Ethernet cable is connected!");
    }
}

void showIP(){
  Serial.print("Device IP adress: ");
  Serial.print(Ethernet.localIP());
  Serial.println();
  }


void udpSetup() {
  Ethernet.init(CS_PIN);    // You can use Ethernet.init(pin) to configure the CS pin
  Ethernet.begin(mac, ip);  // start the Ethernet
  detectHardware();
  detectCable();
  Udp.begin(localPort); //start UDP
}
