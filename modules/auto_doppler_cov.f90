module auto_doppler_forecast

implicit none

contains
!########################################################################
subroutine est_doppler_error(cl,lmax,bnu,dw,nvar)
implicit none

integer*4, parameter :: ndim=5000
real*8, parameter :: pi=3.1415d0
integer*4, intent(in) :: lmax
real*8, intent(in) :: cl(0:4,0:lmax),bnu,dw
integer*4 :: i,il1,il2,ier
real*8 :: l1,l2,l2min,l2max,wig3j(ndim),nvar_inv
real*8 :: G1,cov1,GbyC1
real*8, intent(out) :: nvar

nvar_inv=0.d0
do il1=2,lmax
    l1=float(il1)
    call drc3jj(1.d0,l1,0.d0,0.d0,l2min,l2max,wig3j,ndim,ier)
    do i=1,int(min(l2max,float(lmax))-l2min)+1
       il2=int(l2min)+i-1 ; l2=float(il2) !; print*, il2
       if (il2.gt.1) then
           G1=(cl(0,il1) + cl(0,il2)) !Modulation
           G1=G1*bnu -  dw*(cl(1,il1)*F(l1,1.d0,l2)+cl(1,il2)*F(l2,1.d0,l1)) !Aberration
           G1=G1*wig3j(i)*sqrt((2.d0*l1+1.d0)*(2.d0*l2+1.d0))/sqrt(4.d0*pi)
           cov1=(cl(2,il1)*cl(3,il2) + cl(2,il2)*cl(3,il1) + 2.d0*cl(4,il1)*cl(4,il2))/4.
           GbyC1=G1/cov1
           nvar_inv=nvar_inv + GbyC1*G1
       endif
    enddo
enddo

nvar=1.d0/nvar_inv

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

end module auto_doppler_forecast
