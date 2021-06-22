//马拉车算法(Manacher)  找出一个字符串中的最长回文子串 O(n)
int n,hw[maxn<<1],ans;              //hw记录每个点的回文半径，hw[i]-1即为以i为中心的最长回文长度（去掉'#'），n为字符串长度
char a[maxn],s[maxn<<1];       //a为原字符串，s为扩展后的字符串
void change()            //将相邻两个字符之间插上'#'
{
    s[0]=s[1]='#';
    for(int i=0; i<n; i++)
    {
        s[i*2+2]=a[i];
        s[i*2+3]='#';
    }
    n=n*2+2;
    s[n]=0;
}
int manacher()       //马拉车算法
{
    ans=1;
    change();
    int maxright=0,mid;        //maxright记录当前可以拓展到的最右的回文串的最右边界(不可达边界，maxright-1才是可达边界)，mid记录这个最长回文串的中心点
    for(int i=1; i<n; i++)
    {
        if(i<maxright)
            hw[i]=min(hw[(mid<<1)-i],hw[mid]+mid-i);  //((mid<<1)-i)是i关于mid对称的那个点   hw[mid]+mid-i是i到maxright的最长距离
        else
            hw[i]=1;
        for(; s[i+hw[i]]==s[i-hw[i]]; ++hw[i]); //暴力扩展maxright的外部部分
        if(hw[i]+i>maxright)
        {
            maxright=hw[i]+i;
            mid=i;
        }
       // cout<<s[i]<<' '<<hw[i]<<endl;
        ans=max(ans,hw[i]);
    }
    return ans-1;                 //返回最长回文串的长度
}

signed main()
{
    scanf("%s",a);
    n=strlen(a);
    printf("%d\n",manacher());
    return 0;
}

//PS:manacher算法求得最长回文串的回文半径hw[i]以及该回文串的中心位置i，
//则s[i-(hw[i]-1)+1]~s[i+(hw[i]-1)-1]中去掉#的就是最长回文子串，
//原位置为a[(i-(hw[i]-1)+1)/2-1]~a[(i+(hw[i]-1)-1)/2-1].
//a[(i-hw[i])/2]~a[(i+hw[i])/2-2]