  # the compiler: gcc for C program, define as g++ for C++
  CC = gcc

  # compiler flags:
  #  -g    adds debugging information to the executable file
  #  -Wall turns on most, but not all, compiler warnings
  #  -fopenmp include OpenMP
  CFLAGS  = -g -Wall

  # file names for parallel and serial
  SERIAL = histogram_s

all: clean histogram_s.o

histogram_s.o:
	$(CC) $(CFLAGS) $(SERIAL).c -o $(SERIAL).o -lm

clean:
	clear && touch trash.o && touch trash.txt && rm *.o && rm *.txt