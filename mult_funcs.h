void dcmp_data_2pc(float in_oarr[], float in_sarr0[], float in_sarr1[], int in_ny, int in_nx, int in_sy, int in_sx)
{
   int i,j;

   for (j=0;j<in_sy;j++)
      for (i=0;i<in_sx;i++)
      {
         in_sarr0[i + j * in_sx]  =  in_oarr[ i        + j * in_nx];
         in_sarr1[i + j * in_sx]  =  in_oarr[(i+in_sx) + j * in_nx];
      }
}


void pad0(float in_npd[], float in_pdd[], int in_sy, int in_sx)
{
   int i,j;

   for (j=0;j<in_sy;j++)
      for (i=0;i<in_sx;i++)
         in_pdd[(i+1) + (j+1) * (in_sx+2)] =  in_npd[i + j * in_sx];
}


void dpad0(float in_pdd[], float in_dpd[], int in_sy, int in_sx)
{
   int i,j;

   for (j=0;j<in_sy;j++)
      for (i=0;i<in_sx;i++)
         in_dpd[i + j * in_sx] =  in_pdd[(i+1) + (j+1) * (in_sx+2)];
}


void get_we_binfo(float in_arr[], int in_sy, int in_sx, float in_e[], float in_w[])
{
   int i;

   for (i=0;i<in_sy;i++)
   {
      in_e[i]  =  in_arr[(i+2) * (in_sx+2) - 2]; // east
      in_w[i]  =  in_arr[(i+1) * (in_sx+2) + 1]; // west
   }
}


void bexchg(float in_arr[], int in_sy, int in_sx, float in_e[], float in_w[])
{
   int i;

   // north & south
   for (i=0;i<in_sx;i++)
   {
      // north to south
      in_arr[i + (in_sx+2) * (in_sy+1) + 1]  = 
      in_arr[i + (in_sx+2)             + 1];

      // south to north
      in_arr[i                         + 1]  = 
      in_arr[i + (in_sx+2) *  in_sy    + 1];
   }

   // east & west
   for (i=0;i<in_sy;i++)
   {
      in_arr[(i+2) * (in_sx+2) - 1] =  in_w[i]; // 'east' from other process's west
      in_arr[(i+1) * (in_sx+2)    ] =  in_e[i]; // 'west' from other process's east
   }
}


void arr_init(float in_arr[], int in_sy, int in_sx, int in_oy, int in_ox, int in_iy, int in_ix)
{
   int i,j;

   for (j=0;j<in_sy;j++)
      for (i=0;i<in_sx;i++)
         in_arr[i + j * in_sx] =  0.0;

   for (j=0;j<in_iy;j++)
      for (i=0;i<in_ix;i++)
         in_arr[(i+in_ox) + (j+in_oy) * in_sx] =  100.0;
}


void arr_init0(float in_arr[], int in_sy, int in_sx)
{
   int i,j;

   for (j=0;j<in_sy;j++)
      for (i=0;i<in_sx;i++)
         in_arr[i + j * in_sx] =  i + j * in_sx;
}


void ups_adv_bn(float in_arr0[], float in_arr1[], int in_sy, int in_sx, float in_u, float in_v, float in_c)
{
   int   i   , j    ,
         c_yi, c_xi ,
         m_yi, m_xi ,
         idx , idx_i, idx_j;

   for (j=0;j<in_sy;j++)
      for (i=0;i<in_sx;i++)
      {
         c_xi  =   i          % in_sx;
         c_yi  =   j          % in_sy;
         m_xi  =  (i+in_sx-1) % in_sx;
         m_yi  =  (j+in_sy-1) % in_sy;
         
         idx   =  c_xi + c_yi * in_sx;
         idx_i =  m_xi + c_yi * in_sx;
         idx_j =  c_xi + m_yi * in_sx;
         
         in_arr1[idx]  =  in_arr0[idx]
                       -  in_u * in_c * (in_arr0[idx] - in_arr0[idx_i])
                       -  in_v * in_c * (in_arr0[idx] - in_arr0[idx_j]);
      }
}


void ups_adv_nb(float in_arr0[], float in_arr1[], int in_sy, int in_sx, float in_u, float in_v, float in_c)
{
   int   i   , j    ,
         rx  , ry   ,
         c_yi, c_xi ,
         m_yi, m_xi ,
         idx , idx_i, idx_j;
   
   rx    =  in_sx + 2;
   ry    =  in_sy + 2;
   
   for (j=1;j<ry;j++)
      for (i=1;i<rx;i++)
      {
         c_xi  =   i   ;
         c_yi  =   j   ;
         m_xi  =  (i-1);
         m_yi  =  (j-1);
         
         idx   =  c_xi + c_yi * rx;
         idx_i =  m_xi + c_yi * rx;
         idx_j =  c_xi + m_yi * rx;
         
         in_arr1[idx]  =  in_arr0[idx]
                       -  in_u * in_c * (in_arr0[idx] - in_arr0[idx_i])
                       -  in_v * in_c * (in_arr0[idx] - in_arr0[idx_j]);
      }
}


void dpad0_cat_arr(float in_carr[], float in_parr0[], float in_parr1[], int in_sy, int in_sx)
{
   int i,j,
       px   =  in_sx + 2,
       rx   =  in_sx * 2;

   for (j=0;j<in_sy;j++)
      for (i=0;i<in_sx;i++)
      {
         in_carr[ i        + j * rx]  =  in_parr0[(i+1) + (j+1) * px];
         in_carr[(i+in_sx) + j * rx]  =  in_parr1[(i+1) + (j+1) * px];
      }
}


void print_2darr_fl(float in_arr[], int in_sy, int in_sx, const char* in_name)
{
   int i,j;

   printf("\n===============%10s===============\n",in_name);

   for (j=0;j<in_sy;j++)
   {
      for (i=0;i<in_sx;i++)
      {
         printf("%5.0f ",in_arr[i + j * in_sx]);
      }
      printf("\n");
   }

   printf("\n----------------------------------------\n");

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
