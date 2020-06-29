#ifndef RECORDING_H
#define RECORDING_H


//Header Example 
//################################################################
//#Name: Etienne Montenegro                         Date: 23/06/20
//#Duration: 1:00:00                            Location: Montreal
//#Start time: 13:40:34                         End Time: 13:40:34
//#Signals: heart , gsr1 , gsr2 , resp        Sample Rate: 1000 Hz
//################################################################

class Recording{
  
  public:
   //methods

    //Variables

    //Header arrays
    String headerLine1= "#";
    String headerLine2 = "#";
    String headerLine3 = "#";
    String headerLine4 = "#";
    
    String nameTitle = "Name: ";
    String dateTitle = "Date: ";
    String durationTitle = "Duration: ";
    String locationTitle = "Location: ";
    String startTimeTitle = "Start time: ";
    String endTimeTitle = "End Time: ";
    String signalsTitle = "Signals: ";
    String sampleRateTitle = "Sample Rate: ";
    

    String date; //date of the recording formated dd/mm/yy
    String location ; // location where the recording took place --- Can allocate more space if needed
    String subjectName ; //name of the subject ------ Can allocate more space if needed 
    String signals[10]; // format ["heart, "gsr1" , "gsr2" ,"resp"]
   
    //MODIFIY THIS WHEN RTC IS IMPLEMENTED
    String duration = "1:00:00";  
    String startTime = "15:32:43";
    String endTime = "16:32:43";
    
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
    
    bool headerPrinted; 

////////////////////////METHODS/////////////////////////////
 Recording(int _numSignals){
  numSignals = _numSignals;
  
  mode = 0 ; 
  headerPrinted = false;
  }; //constructor


//FORMATTING THE HEADER COULD BE DONE WIT ONE BUFFER ONLY IF CLEARED BEETWEN EACH PRINT

String formatHeader1(){
  int spaceLength = 64 -nameTitle.length() - subjectName.length() - dateTitle.length() - date.length();
  String spaces;

  for( int i = 0 ; i < spaceLength ; i++ ){
    spaces.concat(" ");
    }
  headerLine1.concat(nameTitle);
  headerLine1.concat(subjectName);
  headerLine1.concat(spaces);
  headerLine1.concat(dateTitle);
  headerLine1.concat(date);

  return headerLine1;
  }

String formatHeader2(){
  int spaceLength = 64 -durationTitle.length() - duration.length() - locationTitle.length() - location.length();
  String spaces;

  for( int i = 0 ; i < spaceLength ; i++ ){
    spaces.concat(" ");
    }
  headerLine2.concat(durationTitle);
  headerLine2.concat(duration);
  headerLine2.concat(spaces);
  headerLine2.concat(locationTitle);
  headerLine2.concat(location);

  return headerLine2;
  }

String formatHeader3(){
  int spaceLength = 64 -startTimeTitle.length() - startTime.length() - endTimeTitle.length() - endTime.length();
  String spaces;

  for( int i = 0 ; i < spaceLength ; i++ ){
    spaces.concat(" ");
    }
  headerLine3.concat(startTimeTitle);
  headerLine3.concat(startTime);
  headerLine3.concat(spaces);
  headerLine3.concat(endTimeTitle);
  headerLine3.concat(endTime);

  return headerLine3;
  }

String formatHeader4(){
  //61 instead of 64 because adding " Hz" at the end

 int signalsLength = 0 ; 

 for( int i = 0 ; i < numSignals ; i++ ){
  signalsLength += getSignal(i).length();
 }
  
  int spaceLength = 61 -signalsTitle.length() - signalsLength - (numSignals - 1) - sampleRateTitle.length() - String(rate).length();
  String spaces;

  for( int i = 0 ; i < spaceLength ; i++ ){
    spaces.concat(" ");
    }
  headerLine4.concat(signalsTitle);
  for( int i = 0 ; i < numSignals ; i++ ){
  headerLine4.concat(getSignal(i));
  if( i != numSignals-1) headerLine4.concat(",");
  }
  headerLine4.concat(spaces);
  headerLine4.concat(sampleRateTitle);
  headerLine4.concat(rate);
  headerLine4.concat(" Hz");

  return headerLine4;
  }


//---------------------------------------------------------------
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

//passing int instead of unsigned long could cause to truncate the timestamp
//try to pass two argument ( unsigned long timestamp  , int data[3] ) 
  String formatData(unsigned long timestamp,int data[3]){// format the sampled data to csv
      // each log of data will be formated like this :
      // timestamp,signal1,signal2,signal3,signal4
      // 987753,,1023,1023,1023
      
      sprintf(dataBuff ,"%ld,%d,%d,%d" ,timestamp,data[0],data[1],data[2]);
      return dataBuff ;
      };  


//--------------------------------------------------------------

     void setLocation(String _Location){//set subject name
      location = _Location;
      }; 

    String getLocation(){//return subject name as string
      return location;
      }; 
      
//--------------------------------------------------------------

    void setSubjectName(String _SubjectName){//set subject name
      subjectName =_SubjectName;
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

//--------------------------------------------------------------
    
     void setDate(String dd , String mm , String yy){ //set date
      date = dd + "/" + mm + "/" + yy;
      }; 

    String getDate(){//return the date as string
      return date;
      };


 
}; //end class declaration with semicolon in c++


#endif
