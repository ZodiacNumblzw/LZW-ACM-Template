//暴力求解原根
int G(int s)
{
    int q[1010]={0};
    for(int i=2;i<=s-2;i++) if ((s-1)%i==0) q[++q[0]]=i;
    for (int i=2;;i++)
    {
        bool B=1;
        for (int j=1;j<=q[0]&&B;j++) if (quick(i,q[j],s)==1) B=0;
        if (B) return i;
    }
    return -1;
}