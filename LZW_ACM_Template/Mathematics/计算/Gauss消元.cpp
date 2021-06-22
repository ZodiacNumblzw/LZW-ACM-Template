//Gauss��Ԫ        O(n^3)
double a[N][N];
int  Gauss(int n,int m){
    int col,i,mxr,j,row;
    for(row=col=1;row<=n&&col<=m;row++,col++){
        mxr = row;
        for(i=row+1;i<=n;i++)
            if(fabs(a[i][col])>fabs(a[mxr][col]))
                mxr = i;
        if(mxr != row) swap(a[row],a[mxr]);
        if(fabs(a[row][col]) < eps){
            row--;
            continue;
        }
        for(i=1;i<=n;i++)///���������Ǿ���
            if(i!=row&&fabs(a[i][col])>eps)
                for(j=m;j>=col;j--)
                    a[i][j]-=a[row][j]/a[row][col]*a[i][col];
    }
    row--;
    for(int i = row;i>=1;i--){///�ش��ɶԽǾ���
        for(int j = i + 1;j <= row;j++){
                a[i][m] -= a[j][m] * a[i][j];
        }
        a[i][m] /= a[i][i];
    }
    return row;          //������
}