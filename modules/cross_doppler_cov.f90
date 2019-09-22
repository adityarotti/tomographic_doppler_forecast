module cross_doppler_forecast

implicit none

contains
!########################################################################
subroutine est_doppler_error(cl1,cl2,clc,lmax,bnu,dw,crosscov)
implicit none

integer*4, parameter :: ndim=5000
real*8, parameter :: pi=3.1415d0
integer*4, intent(in) :: lmax
real*8, intent(in) :: cl1(0:4,0:lmax),cl2(0:4,0:lmax),clc(0:3,0:lmax)
real*8, intent(in) :: bnu,dw
integer*4 :: i,il1,il2,ier
real*8 :: l1,l2,l2min,l2max,wig3j(ndim),nvar_inv,neb,ncd,ncross
real*8 :: G1,cov1,GbyC1,G2,cov2,GbyC2,cov
real*8, intent(out) :: crosscov

neb=0.d0 ; ncd=0.d0 ; ncross=0.d0
do il1=2,lmax
    l1=float(il1)
    call drc3jj(1.d0,l1,0.d0,0.d0,l2min,l2max,wig3j,ndim,ier)
    do i=1,int(min(l2max,float(lmax))-l2min)+1
       il2=int(l2min)+i-1 ; l2=float(il2) !; print*, il2
       if (il2.gt.1) then
		   ! Auto 1
		   ! Modulation
           G1=(cl1(0,il1) + cl1(0,il2))
           ! Aberration
           G1=G1*bnu -  dw*(cl1(1,il1)*F(l1,1.d0,l2)+cl1(1,il2)*F(l2,1.d0,l1))
           G1=G1*wig3j(i)*sqrt((2.d0*l1+1.d0)*(2.d0*l2+1.d0))/sqrt(4.d0*pi)
           cov1=(cl1(2,il1)*cl1(3,il2) + cl1(2,il2)*cl1(3,il1) + 2.d0*cl1(4,il1)*cl1(4,il2))/4.
           GbyC1=G1/cov1
           neb=neb + GbyC1*G1

           ! Auto 2
		   ! Modulation
           G2=(cl2(0,il1) + cl2(0,il2))
           ! Aberration
           G2=G2*bnu -  dw*(cl2(1,il1)*F(l1,1.d0,l2)+cl2(1,il2)*F(l2,1.d0,l1))
           G2=G2*wig3j(i)*sqrt((2.d0*l1+1.d0)*(2.d0*l2+1.d0))/sqrt(4.d0*pi)
           cov2=(cl2(2,il1)*cl2(3,il2) + cl2(2,il2)*cl2(3,il1) + 2.d0*cl2(4,il1)*cl2(4,il2))/4.
           GbyC2=G2/cov2
           ncd=ncd + GbyC2*G2

           cov=clc(0,il1)*clc(1,il2) + clc(0,il2)*clc(1,il1)
           cov=(cov + clc(2,il1)*clc(3,il2) + clc(2,il2)*clc(3,il1))/2.
           ncross = ncross + GbyC1*GbyC2*cov
       endif
    enddo
enddo

crosscov=ncross/neb/ncd

end subroutine est_doppler_error
!########################################################################

!########################################################################
function F(l1,L,l2)
implicit none
real*8, intent(in) :: l1,L,l2
real*8 :: F

F=0.5*(l1*(l1+1.d0) + L*(L+1.d0) - l2*(l2+1.d0))

end
!########################################################################

end module cross_doppler_forecast
