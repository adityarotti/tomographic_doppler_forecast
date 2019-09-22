        subroutine clebsch(NDIM,l,l1,l2,m,m1min,m1max,cleb)
        integer ier,i,m2
        real*8 THRCOF(NDIM),l,l1,l2,m,m1min,m1max ! Wigner 3j variables
        real*8 cleb(NDIM)

        CALL DRC3JM (l, l1, l2, -m, m1min, m1max, THRCOF, NDIM, IER)

        do i=1,int(m1max-m1min)+1
        m2=int(m1min)+i-1
        if (mod(int(l1-l2+m),2).eq.0) then
            cleb(i)= (sqrt(2.*l+1.))*THRCOF(i)
        else
            cleb(i)= (-1)*(sqrt(2.*l+1.))*THRCOF(i)
        endif
!            cleb(i)= (-1)**int(l-l1-(m+m2))*(sqrt(2.*l2+1.))*THRCOF(i)
!            cleb(i)= THRCOF(i)
        enddo

        return
        end 
