all: main.exe

LIB=-L -lnlopt
INC=-I
LDFLAGS="-Wl,-rpath,"

main.exe: main.o
	g++ -o main.exe main.o -fopenmp $(LIB) $(LDFLAGS) -pg

main.o: main.cpp
	g++ -o main.o -c main.cpp -std=c++11 -Wall -Wextra -fopenmp $(INC) -lm -Ofast -march=native -pg
    
clean:
	 rm main.o main.exe