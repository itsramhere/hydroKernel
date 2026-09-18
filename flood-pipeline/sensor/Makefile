# Makefile for FreeRTOS POSIX Rain Gauge Sensor Simulator

CC ?= gcc
CFLAGS ?= -Wall -Wextra -O2 -pthread
TARGET = rain_gauge_posix
SRC = main_posix.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
