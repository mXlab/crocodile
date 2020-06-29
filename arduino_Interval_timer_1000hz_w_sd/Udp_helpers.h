
void sendMessage( char reply[] ){
//    send a reply to the IP address and port that sent us the packet we received
    Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
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
    Serial.print("Received packet of size ");
    Serial.println(packetSize);
    Serial.print("From ");
    IPAddress remote = Udp.remoteIP();
    for (int i=0; i < 4; i++) {
      Serial.print(remote[i], DEC);
      if (i < 3) {
        Serial.print(".");
      }
    }
    Serial.print(", port ");
    Serial.println(Udp.remotePort());

    // read the packet into packetBufffer
    Udp.read(packetBuffer, UDP_TX_PACKET_MAX_SIZE);
    Serial.println("Contents:");
    Serial.println(packetBuffer);
  }
}

void parseMessage(){
  

   
    if(packetBuffer[0] == 67 && packetBuffer[1] == 111 && packetBuffer[2] == 110){ //parse for "Con" message
    ComputerIP = Udp.remoteIP();
    ComputerPort = Udp.remotePort();
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
      int dd = d.substring(3,4).toInt();
      int mm = d.substring(5,6).toInt();
      int yy = d.substring(7,8).toInt();
      r.setDate(dd,mm,yy);
      sendMessage( "rcvdD");
      
      }else if(packetBuffer[0] == 73 && packetBuffer[1] == 70 && packetBuffer[2] == 114){
        
      Serial.println("Received Sample Rate");
      String rate =  extractMessage();
      r.setRecRate(rate.toInt() );
      sendMessage( "rcvdR");
       
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
