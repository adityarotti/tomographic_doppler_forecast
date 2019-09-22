modules=./modules
subroutines=./modules/subroutines

######################## GFORTRAN ##########################
FC=gfortran -fbounds-check -ffixed-line-length-none #-Wall -Wextra -Wconversion
F77flags=-c -fopenmp -fPIC
F90flags= -DGFORTRAN -fno-second-underscore -fopenmp -fPIC
############################################################

all:bips
bips:bipsobj
	f2py -c --fcompiler=gfortran *.o -m doppler_forecast $(modules)/doppler_forecast.f90

bipsobj:
	$(FC) -c $(F90flags) $(subroutines)/wigner/*.f 	

clean:
	$(RM) -r *.o *.mod *~ *dSYM

cleanall:
	$(RM) *.o *.mod *.so *.pyf *~
