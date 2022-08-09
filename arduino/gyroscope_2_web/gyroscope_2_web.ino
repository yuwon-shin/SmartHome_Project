#include <SPI.h>
#include <WiFiNINA.h>
#include <Arduino_LSM6DS3.h>
#include "arduino_secrets.h"
///////보안을 위해 연결하고자 하는 와이파이 SSID와 비밀번호는 arduino_secrets.h 파일에 쓰는것이 좋습니다.
char ssid[] = SECRET_SSID;        // 네트워크 SSID
char pass[] = SECRET_PASS;    // your network password (use for WPA, or use as key for WEP)
float x, y, z;
int degreesX = 0;
int degreesY = 0;
int status = WL_IDLE_STATUS;
WiFiServer server(80);

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println("Hz");

  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    while (true);
  }

  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to Network named: ");
    Serial.println(ssid);

    status = WiFi.begin(ssid, pass);
    delay(10000);
  }
  server.begin();                 
  printWifiStatus();                  
}


void loop() {
  WiFiClient client = server.available();   
  if (client) {                             
    Serial.println("new client");          
    String currentLine = "";                
    while (client.connected()) {            
      if (client.available()) {                             
         if (IMU.accelerationAvailable()) {
            IMU.readAcceleration(x, y, z);
          }
          if (currentLine.length() == 0) {
            client.println("HTTP/1.1 200 OK");
            client.println("Content-type:text/html");
            client.println("Refresh: 1");
            client.println();
            if (x > 0.1) {
              x = 100 * x;
              degreesX = map(x, 0, 97, 0, 90);
              client.print("Tilting up ");
              client.print(degreesX);
              client.println("  degrees<br>");
            }
            if (x < -0.1) {
              x = 100 * x;
              degreesX = map(x, 0, -100, 0, 90);
              client.print("Tilting down ");
              client.print(degreesX);
              client.println("  degrees<br>");
            }
            if (y > 0.1) {
              y = 100 * y;
              degreesY = map(y, 0, 97, 0, 90);
              client.print("Tilting left ");
              client.print(degreesY);
              client.println("  degrees<br>");
            }
            if (y < -0.1) {
              y = 100 * y;
              degreesY = map(y, 0, -100, 0, 90);
              client.print("Tilting right ");
              client.print(degreesY);
              client.println("  degrees<br>");
            }
            if( x <= 0.1 && x >= -0.1 && y <= 0.1 && y >= -0.1){
              client.println("No tilting<br>");
            }
            client.println();
            break;
          } 
      }
    }
    client.stop();
    Serial.println("client disconnected");
  }
}

void printWifiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your board's IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
  // print where to go in a browser:
  Serial.print("To see this page in action, open a browser to http://");
  Serial.println(ip);
}
