//�������㷨(Manacher)  �ҳ�һ���ַ����е�������Ӵ� O(n)
int n,hw[maxn<<1],ans;              //hw��¼ÿ����Ļ��İ뾶��hw[i]-1��Ϊ��iΪ���ĵ�����ĳ��ȣ�ȥ��'#'����nΪ�ַ�������
char a[maxn],s[maxn<<1];       //aΪԭ�ַ�����sΪ��չ����ַ���
void change()            //�����������ַ�֮�����'#'
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
int manacher()       //�������㷨
{
    ans=1;
    change();
    int maxright=0,mid;        //maxright��¼��ǰ������չ�������ҵĻ��Ĵ������ұ߽�(���ɴ�߽磬maxright-1���ǿɴ�߽�)��mid��¼�������Ĵ������ĵ�
    for(int i=1; i<n; i++)
    {
        if(i<maxright)
            hw[i]=min(hw[(mid<<1)-i],hw[mid]+mid-i);  //((mid<<1)-i)��i����mid�ԳƵ��Ǹ���   hw[mid]+mid-i��i��maxright�������
        else
            hw[i]=1;
        for(; s[i+hw[i]]==s[i-hw[i]]; ++hw[i]); //������չmaxright���ⲿ����
        if(hw[i]+i>maxright)
        {
            maxright=hw[i]+i;
            mid=i;
        }
       // cout<<s[i]<<' '<<hw[i]<<endl;
        ans=max(ans,hw[i]);
    }
    return ans-1;                 //��������Ĵ��ĳ���
}

signed main()
{
    scanf("%s",a);
    n=strlen(a);
    printf("%d\n",manacher());
    return 0;
}

//PS:manacher�㷨�������Ĵ��Ļ��İ뾶hw[i]�Լ��û��Ĵ�������λ��i��
//��s[i-(hw[i]-1)+1]~s[i+(hw[i]-1)-1]��ȥ��#�ľ���������Ӵ���
//ԭλ��Ϊa[(i-(hw[i]-1)+1)/2-1]~a[(i+(hw[i]-1)-1)/2-1].
//a[(i-hw[i])/2]~a[(i+hw[i])/2-2]