modules=./
subroutines=./subroutines

######################## GFORTRAN ##########################
FC=gfortran -fbounds-check -ffixed-line-length-none #-Wall -Wextra -Wconversion
F77flags=-c -fopenmp -fPIC
F90flags= -DGFORTRAN -fno-second-underscore -fopenmp -fPIC
############################################################

all:bips
bips:bipsobj
	#f2py -c --fcompiler=gfortran *.o -m cross_doppler_cov $(modules)/cross_doppler_cov.f90
	f2py -c --fcompiler=gfortran *.o -m auto_doppler_cov $(modules)/auto_doppler_cov.f90

bipsobj:
	$(FC) -c $(F90flags) $(subroutines)/wigner/*.f 	

clean:
	$(RM) -r *.o *.mod *~ *dSYM

cleanall:
	$(RM) *.o *.mod *.so *.pyf *~
