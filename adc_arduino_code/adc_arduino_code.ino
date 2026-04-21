#include <SPI.h>

const byte MCP3004_CS = 10;   // CS/SHDN
const uint16_t VREF_MV = 3300; // change if your VREF is not 5.0V

SPISettings mcp3004SPI(1000000, MSBFIRST, SPI_MODE0);

uint16_t readMCP3004Raw(byte channel)
{
  if (channel > 3) return 0;

  byte tx0 = 0x01;
  byte tx1 = 0x80 | ((channel & 0x03) << 4); // single-ended, CH0-CH3
  byte tx2 = 0x00;

  byte rx0, rx1, rx2;

  SPI.beginTransaction(mcp3004SPI);
  digitalWrite(MCP3004_CS, LOW);

  rx0 = SPI.transfer(tx0);
  rx1 = SPI.transfer(tx1);
  rx2 = SPI.transfer(tx2);

  digitalWrite(MCP3004_CS, HIGH);
  SPI.endTransaction();

  return ((uint16_t)(rx1 & 0x03) << 8) | rx2;
}

uint16_t rawToMillivolts(uint16_t raw)
{
  return (uint32_t)raw * VREF_MV / 1023UL;
}

void setup()
{
  Serial.begin(9600);   // matches your Python visualizer
  SPI.begin();

  pinMode(MCP3004_CS, OUTPUT);
  digitalWrite(MCP3004_CS, HIGH);  // CS idle high

  // Keep SS as output so Uno stays SPI master
  pinMode(10, OUTPUT);
  digitalWrite(10, HIGH);
}

void loop()
{
  // Map MCP3004 channels to the names your Python script expects:
  // CH0 -> PA0
  // CH1 -> PA5
  // CH2 -> PA6
  uint16_t pa0_mv = rawToMillivolts(readMCP3004Raw(0));
  uint16_t pa5_mv = rawToMillivolts(readMCP3004Raw(1));
  uint16_t pa6_mv = rawToMillivolts(readMCP3004Raw(2));

  Serial.print("PA0:");
  Serial.print(pa0_mv);
  Serial.print(",PA5:");
  Serial.print(pa5_mv);
  Serial.print(",PA6:");
  Serial.println(pa6_mv);

  delay(10);
}