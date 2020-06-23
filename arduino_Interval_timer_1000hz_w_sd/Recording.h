#ifndef RECORDING_H
#define RECORDING_H


//Header Example 


class Recording{
  
  public:
   //methods

   
    //Variables
    char date[8]; //date of the recording formated dd/mm/yy
    char location[32] ; // location where the recording took place --- Can allocate more space if needed
    char subjectName[32] ; //name of the subject ------ Can allocate more space if needed 
    String signals[10]; // format ["heart, "gsr1" , "gsr2" ,"resp"]
    int rate ; //refresh rate of the sensors
    int mode ; //current mode the object is in 0 = not recording 1 =  recording 2 = ending recording
    int numSignals;


    //Need RTC TO IMPLEMENT THIS
    int startHour ; //starting hour of the recording
    int startMin ; //starting min of the recording
    int startSec ; //starting sec of the recording

    int endHour ; //end hour of the recording
    int endMin ; //end min of the recording
    int endSec ; //end sec of the recording
 
    char dataBuff[100];
    
  

  
  




////////////////////////METHODS/////////////////////////////
 Recording(int _numSignals){
  numSignals = _numSignals;
  
  mode = 0 ; 
  }; //constructor

bool isReadyToStop(){

  if( mode == 2){ 
   return true;
  
  }else{
    return false;
    }
}

bool isRecording(){

  if( mode == 1 ){
   return true;
  
  }else{
    return false;
    }
}

void readyToStartAgain(){
  mode = 0;
  }

 void startRecording() {//start the recording
    mode = 1;
  }; 

void stopRecording(){ //stop the recording
   mode = 2;
}; 


    void setDate(){//manually set date of recording
      //Serial.println("working");
      }; 

  

String channelNames(){// format the sampled data to csv
      String label = "timestamp"; 
      for( int i = 0 ; i < numSignals ; i++){
         label.concat(",");
         label.concat(signals[i]);
        }
      
      return label ;
      
      };  

  String formatData(int data[4]){// format the sampled data to csv
      // each log of data will be formated like this :
      // timestamp,signal1,signal2,signal3,signal4
      // 987753,,1023,1023,1023
      
      sprintf(dataBuff ,"%d,%d,%d,%d" ,data[0],data[1],data[2],data[3]);
      return dataBuff ;
      };  


//--------------------------------------------------------------

     void setLocation(String _Location){//set subject name
      _Location.toCharArray(location, 32);
      }; 

    String getLocation(){//return subject name as string
      return location;
      }; 
      
//--------------------------------------------------------------

    void setSubjectName(String _SubjectName){//set subject name
      _SubjectName.toCharArray(subjectName, 32);
      }; 

    String getSubjectName(){//return subject name as string
      return subjectName;
      }; 

//--------------------------------------------------------------
    
     void setSignals(String _Signals[4]){ //set subject name
      for( int i = 0 ; i < numSignals ; i++ ){
         signals[i] = _Signals[i];
      }
      }; 

    String getSignal(int index){//return subject name as string
      return signals[index];
      };

//--------------------------------------------------------------

           void setRecRate(int _Rate){ //set subject name
          rate = _Rate;
      }; 

    int getRecRate(){//return subject name as string
      return rate;
      };

 
}; //end class declaration with semicolon in c++


#endif
