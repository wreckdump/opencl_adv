void dcmp_2pc_3d(float in_oarr[], float in_parr0[], float in_parr1[], int in_nz, int in_ny, int in_nx)
{
   int   i,j,k;
   int   oidx0, oidx1, pidx;
   int   nxy   =  in_nx*in_ny,
         sx    =  in_nx/2,
         pz    =  in_nz+2,
         py    =  in_ny+2,
         px    =  sx+2,
         pxy   =  px*py,
         pz0   =  0,
         pz1   =  pz-1,
         py0   =  0,
         py1   =  py-1,
         px0   =  0,
         px1   =  px-1;

   // zeroing outer shell : north & south
   for (j=0;j<py;j++)
      for (i=0;i<px;i++)
      {
         oidx0          =  i + j * px + pz0 * pxy; // out south
         oidx1          =  i + j * px + pz1 * pxy; // out north
         in_parr0[oidx0] =  0.0;
         in_parr0[oidx1] =  0.0;
         in_parr1[oidx0] =  0.0;
         in_parr1[oidx1] =  0.0;
      }

   // zeroing outer shell : west & east
   for (j=0;j<py;j++)
      for (k=0;k<pz;k++)
      {
         oidx0          =  px0 + j * px + k * pxy; // out west
         oidx1          =  px1 + j * px + k * pxy; // out east
         in_parr0[oidx0] =  0.0;
         in_parr0[oidx1] =  0.0;
         in_parr1[oidx0] =  0.0;
         in_parr1[oidx1] =  0.0;
      }

   // zeroing outer shell : up & down
   for (k=0;k<pz;k++)
      for (i=0;i<px;i++)
      {
         oidx0          =  i + py0 * px + k * pxy; // out up
         oidx1          =  i + py1 * px + k * pxy; // out down
         in_parr0[oidx0] =  0.0;
         in_parr0[oidx1] =  0.0;
         in_parr1[oidx0] =  0.0;
         in_parr1[oidx1] =  0.0;
      }

   // writing the inner part
   for (k=0;k<in_nz;k++)
      for (j=0;j<in_ny;j++)
         for (i=0;i<sx;i++)
         {
            pidx           =  (i+1)  + (j+1) * px    + (k+1) * pxy;
            oidx0          =   i     +  j    * in_nx +  k    * nxy;
            oidx1          =  (i+sx) +  j    * in_nx +  k    * nxy;
            in_parr0[pidx] =  in_oarr[oidx0];
            in_parr1[pidx] =  in_oarr[oidx1];
         }
}


void get_binfo_3d(float in_arr[], int in_pz, int in_py, int in_px, float n[], float s[], float w[], float e[], float u[], float d[])
{
   // input array should be 0 padded 3d array

   // indexing variables
   int   i,j,k;

   // iidx : input index
   // oidx : output index
   int   iidx0, iidx1, oidx;

   // frequently used constants for indexing
   int   px1  =  1,
         px2  =  in_px-2,
         py1  =  1,
         py2  =  in_py-2,
         pz1  =  1,
         pz2  =  in_pz-2,
         pxy  =  in_px * in_py;


   // 3D north & south (size : (in_py-2) * (in_px-2))
   for (j=0;j<py2;j++)
      for (i=0;i<px2;i++)
      {
         oidx     =  i + j * px2;
         iidx0    =  (i+1) + (j+1) * in_px + pz1 * pxy; // 3d south
         iidx1    =  (i+1) + (j+1) * in_px + pz2 * pxy; // 3d north
         s[oidx]  =  in_arr[iidx0];
         n[oidx]  =  in_arr[iidx1];
      }


   // 3D west & east (size : (in_py-2) * (in_pz-2))
   for (j=0;j<py2;j++)
      for (k=0;k<pz2;k++)
      {
         oidx     =  k + j * pz2;
         iidx0    =  px1 + (j+1) * in_px + (k+1) * pxy; // 3d west
         iidx1    =  px2 + (j+1) * in_px + (k+1) * pxy; // 3d east
         w[oidx]  =  in_arr[iidx0];
         e[oidx]  =  in_arr[iidx1];
      }


   // 3D up & down (size : (in_pz-2) * (in_px-2))
   for (k=0;k<pz2;k++)
      for (i=0;i<px2;i++)
      {
         oidx     =  i + k * px2;
         iidx0    =  (i+1) + py1 * in_px + (k+1) * pxy; // 3d up
         iidx1    =  (i+1) + py2 * in_px + (k+1) * pxy; // 3d down
         u[oidx]  =  in_arr[iidx0];
         d[oidx]  =  in_arr[iidx1];
      }
}


void bexchg_3d(float in_arr[], int in_pz, int in_py, int in_px, float n[], float s[], float w[], float e[], float u[], float d[])
{
   // input array should be 0 padded 3d array

   // indexing variables
   int   i,j,k;

   // iidx : input index
   // oidx : output index
   int   iidx, oidx0, oidx1;

   // frequently used constants for indexing
   int   px0  =  0,
         px1  =  in_px-1,
         px2  =  in_px-2,
         py0  =  0,
         py1  =  in_py-1,
         py2  =  in_py-2,
         pz0  =  0,
         pz1  =  in_pz-1,
         pz2  =  in_pz-2,
         pxy  =  in_px * in_py;


   // 3D onorth is isouth (size : (in_py-2) * (in_px-2))
   // 3D osouth is inorth (size : (in_py-2) * (in_px-2))
   for (j=0;j<py2;j++)
      for (i=0;i<px2;i++)
      {
         iidx           =  i + j * px2;
         oidx0          =  (i+1) + (j+1) * in_px + pz1 * pxy; // out north
         oidx1          =  (i+1) + (j+1) * in_px + pz0 * pxy; // out south
         in_arr[oidx0]  =  s[iidx];
         in_arr[oidx1]  =  n[iidx];
      }


   // 3D owest is ieast (size : (in_py-2) * (in_pz-2))
   // 3D oeast is iwest (size : (in_py-2) * (in_pz-2))
   for (j=0;j<py2;j++)
      for (k=0;k<pz2;k++)
      {
         iidx           =  k + j * pz2;
         oidx0          =  px0 + (j+1) * in_px + (k+1) * pxy; // out west
         oidx1          =  px1 + (j+1) * in_px + (k+1) * pxy; // out east
         in_arr[oidx0]  =  e[iidx];
         in_arr[oidx1]  =  w[iidx];
      }


   // 3D oup   is idown (size : (in_pz-2) * (in_px-2))
   // 3D odown is iup   (size : (in_pz-2) * (in_px-2))
   for (k=0;k<pz2;k++)
      for (i=0;i<px2;i++)
      {
         iidx           =  i + k * px2;
         oidx0          =  (i+1) + py0 * in_px + (k+1) * pxy; // out up
         oidx1          =  (i+1) + py1 * in_px + (k+1) * pxy; // out down
         in_arr[oidx0]  =  d[iidx];
         in_arr[oidx1]  =  u[iidx];
      }
}


void arr_init_3d(float in_arr[], int in_nz, int in_ny, int in_nx, int in_iz, int in_iy, int in_ix, int in_oz, int in_oy, int in_ox)
{
   int i,j,k,idx;

   for (k=0;k<in_nz;k++)
      for (j=0;j<in_ny;j++)
         for (i=0;i<in_nx;i++)
         {
            idx         =  i + j * in_nx + k * in_nx * in_ny;
            in_arr[idx] =  0.0;
         }

   for (k=0;k<in_iz;k++)
      for (j=0;j<in_iy;j++)
         for (i=0;i<in_ix;i++)
         {
            idx         =  (i+in_ox) + (j+in_oy) * in_nx + (k+in_oz) * in_nx * in_ny;
            in_arr[idx] =  1000.0;
         }
}


void arr_init_3d0(float in_arr[], int in_nz, int in_ny, int in_nx)
{
   int i,j,k,idx;

   for (k=0;k<in_nz;k++)
      for (j=0;j<in_ny;j++)
         for (i=0;i<in_nx;i++)
         {
            idx         =  i + j * in_nx + k * in_nx * in_ny;
            in_arr[idx] =  idx;
         }
}


void dpad0_cat_3d(float in_carr[], float in_parr0[], float in_parr1[], int in_nz, int in_ny, int in_nx)
{
   int   i,j,k;
   int   oidx0, oidx1, pidx;
   int   nxy   =  in_nx*in_ny,
         sz    =  in_nz,
         sy    =  in_ny,
         sx    =  in_nx/2,
         pz    =  in_nz+2,
         py    =  in_ny+2,
         px    =  sx+2,
         pxy   =  px*py;

   for (k=0;k<sz;k++)
      for (j=0;j<sy;j++)
         for (i=0;i<sx;i++)
         {
            pidx           =  (i+1)  + (j+1) * px    + (k+1) * pxy;
            oidx0          =   i     +  j    * in_nx +  k    * nxy;
            oidx1          =  (i+sx) +  j    * in_nx +  k    * nxy;
            in_carr[oidx0] =  in_parr0[pidx];
            in_carr[oidx1] =  in_parr1[pidx];
         }
}


void print_3darr_fl(float in_arr[], int in_nz, int in_ny, int in_nx, const char* in_name)
{
   int i,j,k;

   printf("\n===============%10s     ===============\n\n",in_name);

   for (k=0;k<in_nz;k++)
   {
      for (j=0;j<in_ny;j++)
      {
         for (i=0;i<in_nx;i++)
         {
            printf("%5.0f ",in_arr[i + j * in_nx + k * in_nx * in_ny]);
         }
         printf("\n");
      }
      printf("\n");
   }

   printf("---------------------------------------------\n");

}


void print_2darr_fl(float in_arr[], int in_sy, int in_sx, const char* in_name)
{
   int i,j;

   printf("\n===============%10s     ===============\n",in_name);

   for (j=0;j<in_sy;j++)
   {
      for (i=0;i<in_sx;i++)
      {
         printf("%5.0f ",in_arr[i + j * in_sx]);
      }
      printf("\n");
   }

   printf("---------------------------------------------\n");

}


void print_1darr_fl(float in_arr[], int in_sx, const char* in_name)
{
   int i;

   printf("\n===============%10s===============\n",in_name);

   for (i=0;i<in_sx;i++)
   {
      printf("%5.0f\n",in_arr[i]);
   }

   printf("\n----------------------------------------\n");
}



