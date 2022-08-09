#include <ArduinoBLE.h>
BLEDevice peripheral;

void setup() {
  Serial.begin(9600);
  while (!Serial);
  if (!BLE.begin()) {
    Serial.println("starting Bluetooth® Low Energy module failed!");
    while (1);
  }
  Serial.println("Bluetooth® Low Energy Central - Peripheral Explorer");
  //BLE.scanForName("Inspire 2"); // 특정 name, 혹은 특정 mac address를 대상으로 연결 가능함
  BLE.scan(); // 그냥 범용적(전체적)으로 scan

}

void loop() {
  if (peripheral.connected()) { // 이미 연결이 되어 있다면 rssi만 필요하므로 이것만 출력
    Serial.print("RSSI = ");
    Serial.println(peripheral.rssi());
    delay(500);
    Serial.println(peripheral.connected());
  }
  else {
    peripheral = BLE.available(); //원하는 address만 연결되도록 서칭함
    if (peripheral) {
      Serial.print("Found ");
      Serial.print(peripheral.address());
      Serial.print(" '");
      Serial.print(peripheral.localName());
      Serial.print("' ");
      Serial.print(peripheral.advertisedServiceUuid());
      Serial.println();
      if (peripheral.address() == "4c:59:4c:83:92:fa") { // fitbit: fe:88:62:cb:ba:45, arduino beacon: 58:bf:25:3c:24:96, galazy 8+: 7a:00:2a:a5:75:71(random address라 끄면 다시 검색)
        BLE.stopScan();
        peripheral.connect();
      }
    }
  }
}
