CC_SERIAL := gcc
CC_MPI := mpicc

CFLAGS := -O3 -std=c11 -Wall -Wextra -Wshadow -Wconversion -Wno-unused-parameter
LDFLAGS := -lm

.PHONY: all clean

all: sw_serial sw_serial_lw sw_serial_mc sw_serial_schemes sw_mpi sw_mpi_lw sw_mpi_mc

sw_serial: shallow_water_conservative_serial.c
	$(CC_SERIAL) $(CFLAGS) -o $@ $< $(LDFLAGS)

sw_serial_lw: shallow_water_conservative_serial_lw.c
	$(CC_SERIAL) $(CFLAGS) -o $@ $< $(LDFLAGS)

sw_serial_mc: shallow_water_conservative_serial_mc.c
	$(CC_SERIAL) $(CFLAGS) -o $@ $< $(LDFLAGS)

sw_serial_schemes: shallow_water_conservative_serial_schemes.c
	$(CC_SERIAL) $(CFLAGS) -o $@ $< $(LDFLAGS)

sw_mpi: shallow_water_conservative_mpi.c
	$(CC_MPI) $(CFLAGS) -o $@ $< $(LDFLAGS)

sw_mpi_lw: shallow_water_conservative_mpi_lw.c
	$(CC_MPI) $(CFLAGS) -o $@ $< $(LDFLAGS)

sw_mpi_mc: shallow_water_conservative_mpi_mc.c
	$(CC_MPI) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f sw_serial sw_serial_lw sw_serial_mc sw_serial_schemes sw_mpi sw_mpi_lw sw_mpi_mc *.o *.csv
