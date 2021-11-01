#include <IRROBOT_EZController.h>

IRROBOT_EZController Easy(&Serial1);

#define ID_MAX 11
#define MANUAL_POSITION_VR Easy.VR_1
#define A_POSITION_VR Easy.VR_2
#define B_POSITION_VR Easy.VR_3
#define EXT_ANALOG_VR Easy.VR_4   //VR4 : A0  //VR5 : A2  //VR6 : A3
#define VR_MIN 0
#define VR_MAX 1023
#define VAL_MIN 0
#define VAL_MAX 4095
#define ID_NUM 0
#define PWM_MIN 900 //900
#define PWM_MAX 2100 //2100
#define PWM_VAL map(position_val,VAL_MIN,VAL_MAX,PWM_MIN,PWM_MAX)

short position_val;
int data = 0;
int tempdata = 0;
byte var = 0;

void setup(){
  Easy.begin();
  Easy.MightyZap.begin(32);
  Easy.setStep(ID_MAX, 0, 1023);
  Serial.begin(9600);
  Easy.MightyZap.GoalSpeed(ID_NUM,512);
  Serial.println(Easy.MightyZap.GoalSpeed(ID_NUM));
}

void loop() {
  unsigned char MightyZap_actID = ID_NUM;
  short Ext_analog_val;
  if (Serial.available() > 0)
  {
    data = Serial.parseInt();
    if (data > 4095){
      data = 4095;
    }else if (data < 0){
      data = 0;
    }
    tempdata = data;
  }else{
    data = tempdata;
  }
  
  Easy.MightyZap.GoalPosition(ID_NUM, data);
  Easy.servo_CH1.writeMicroseconds(PWM_VAL);
  Serial.println(data);
//  Serial.print(data);
  delay(1);

}
